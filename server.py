"""Dron autónomo: servidor local del simulador.

    python server.py            -> http://127.0.0.1:7873

Python simula todo (física a 200 Hz, sensores, filtro de Kalman, planificación, control); el navegador solo
dibuja. Escucha únicamente en 127.0.0.1.

Fluidez: la simulación corre en un HILO PROPIO, hasta 1 s (de tiempo simulado) por delante de lo que está
mostrando el navegador, y guarda un fotograma cada 0,05 s. El navegador pide los fotogramas nuevos y los reproduce
con un pequeño colchón. Así, si un paso tarda más de la cuenta (una replanificación), el vídeo no se detiene.
El ritmo lo marca el reloj de reproducción del navegador: si pausas, la simulación se para 1 s por delante.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dron.mission import MODES
from dron.params import PROFILES
from dron.sensors import NOISE_LEVELS, RAIN
from dron.search import SEARCH_MODES
from dron.sim import MAP_MODES, Simulation
from dron.swarm import SWARM_MAX, Swarm
from dron.target import MOTIONS
from dron.world import DENSITIES, GOAL_KINDS, LEVELS, TERRAINS

HERE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 7873))
STATIC = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"),
          "/app.js": ("app.js", "application/javascript")}
FRAME_DT = 0.05     # s simulados entre fotogramas
LOOKAHEAD = 1.0     # s simulados que la simulación puede ir por delante del navegador


class Session:
    def __init__(self):
        self.sim = None
        self.results = []
        self.frames = []
        self.clock = 0.0          # tiempo de reproducción del navegador
        self.lock = threading.Lock()
        self.gen = 0              # cambia en cada reset: el hilo anterior se detiene solo
        self._recorded = False

    def reset(self, cfg):
        drones = int(cfg.get("drones", 1) or 1)
        search = cfg.get("search", "no")
        team = drones > 1 and (search != "no" or (cfg.get("motion") == "huye" and cfg.get("mode") == "carrera"))
        make = (lambda **kw: Swarm(drones, **kw)) if team else Simulation
        sim = make(
            profile=cfg.get("profile", "px4"), level=cfg.get("level", "mixto"),
            seed=int(cfg["seed"]) if str(cfg.get("seed", "")).strip() else None,
            wind_speed=float(cfg.get("wind_speed", 0)), wind_dir=float(cfg.get("wind_dir", 0)),
            gusts=int(cfg.get("gusts", 0)), noise=cfg.get("noise", "realista"),
            precision_landing=bool(cfg.get("precision_landing", True)),
            collision_prevention=bool(cfg.get("collision_prevention", True)),
            terrain=cfg.get("terrain", "plano"), density=cfg.get("density", "normal"),
            goal_kind=cfg.get("goal_kind", "suelo"), mode=cfg.get("mode", "aterrizar"), rain=cfg.get("rain", "no"),
            motion=cfg.get("motion", "fija"), map_mode=cfg.get("map_mode", "desconocido"),
            search=cfg.get("search", "no"))
        with self.lock:
            self.gen += 1
            self.sim, self.frames, self.clock, self._recorded = sim, [], 0.0, False
            first = self._frame(True)
            gen = self.gen
        threading.Thread(target=self._run, args=(gen,), daemon=True).start()
        return first

    def _run(self, gen):
        """Simula por delante del navegador y va guardando fotogramas."""
        while True:
            with self.lock:
                if gen != self.gen or self.sim is None or self.sim.done:
                    return
                ahead = self.sim.t - self.clock
            if ahead > LOOKAHEAD:
                time.sleep(0.005)
                continue
            with self.lock:
                if gen != self.gen:
                    return
                self.sim.step(FRAME_DT)
                self.frames.append(self._frame(False))
                if len(self.frames) > 400:
                    self.frames = self.frames[-400:]

    def poll(self, since, clock):
        """Fotogramas posteriores a `since`; `clock` es por dónde va la reproducción en el navegador."""
        with self.lock:
            if self.sim is None:
                raise ValueError("No hay vuelo: pulsa Nuevo vuelo")
            self.clock = max(self.clock, float(clock))
            out = [f for f in self.frames if f["t"] > since][:60]
            return {"frames": out}

    def _frame(self, full):
        s = self.sim
        f = s.frame(full)
        if s.done and not self._recorded:
            self._recorded = True
            land = float(((s.drone.p[0] - s.mission.pad[0]) ** 2 + (s.drone.p[1] - s.mission.pad[1]) ** 2) ** 0.5)
            self.results.append({"status": s.status, "time": s.t, "land": land, "profile": s.prof.key,
                                 "mode": s.mode, "seed": s.world.seed, "found_t": s.mission.found_t})
        f["results"] = self.results[-200:]
        return f


session = Session()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # conexiones persistentes: el navegador reutiliza la conexión en cada consulta

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in STATIC:
            name, ctype = STATIC[self.path]
            with open(os.path.join(HERE, "ui", name), "rb") as f:
                return self._send(200, f.read(), ctype)
        if self.path == "/api/options":
            return self._send(200, {"profiles": {k: p.to_dict() for k, p in PROFILES.items()},
                                    "levels": list(LEVELS), "noise": list(NOISE_LEVELS), "terrains": list(TERRAINS),
                                    "densities": list(DENSITIES), "goal_kinds": list(GOAL_KINDS),
                                    "modes": list(MODES), "rain": list(RAIN), "motions": list(MOTIONS),
                                    "map_modes": list(MAP_MODES), "searches": list(SEARCH_MODES),
                                    "swarm_max": SWARM_MAX})
        self._send(404, {"error": "No encontrado"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/reset":
                return self._send(200, session.reset(req))
            if self.path == "/api/frames":
                return self._send(200, session.poll(float(req.get("since", -1)), float(req.get("clock", 0))))
            self._send(404, {"error": "No encontrado"})
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # que la página muestre el error en vez de quedarse colgada
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Dron autónomo en http://%s:%d  (Ctrl+C para parar)" % (HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nParado.")


if __name__ == "__main__":
    main()
