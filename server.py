"""Dron autónomo: servidor local del simulador.

    python server.py            -> http://127.0.0.1:7873

Python simula todo (física a 200 Hz, sensores, filtro de Kalman, planificación, control); el navegador solo
dibuja. Escucha únicamente en 127.0.0.1.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dron.mission import MODES
from dron.params import PROFILES
from dron.sensors import NOISE_LEVELS, RAIN
from dron.sim import Simulation
from dron.target import MOTIONS
from dron.world import DENSITIES, GOAL_KINDS, LEVELS, TERRAINS

HERE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 7873))
STATIC = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"),
          "/app.js": ("app.js", "application/javascript")}


class Session:
    def __init__(self):
        self.sim = None
        self.results = []

    def reset(self, cfg):
        self.sim = Simulation(
            profile=cfg.get("profile", "px4"), level=cfg.get("level", "mixto"),
            seed=int(cfg["seed"]) if str(cfg.get("seed", "")).strip() else None,
            wind_speed=float(cfg.get("wind_speed", 0)), wind_dir=float(cfg.get("wind_dir", 0)),
            gusts=int(cfg.get("gusts", 0)), noise=cfg.get("noise", "realista"),
            precision_landing=bool(cfg.get("precision_landing", True)),
            collision_prevention=bool(cfg.get("collision_prevention", True)),
            terrain=cfg.get("terrain", "plano"), density=cfg.get("density", "normal"),
            goal_kind=cfg.get("goal_kind", "suelo"), mode=cfg.get("mode", "aterrizar"), rain=cfg.get("rain", "no"),
            motion=cfg.get("motion", "fija"))
        self._recorded = False
        return self._frame(True)

    def step(self, dt):
        if self.sim is None:
            raise ValueError("No hay vuelo: pulsa Nuevo vuelo")
        self.sim.step(min(max(float(dt), 0.005), 1.0))
        return self._frame(False)

    def _frame(self, full):
        f = self.sim.frame(full)
        s = self.sim
        if s.done and not self._recorded:
            self._recorded = True
            land = float(((s.drone.p[0] - s.mission.pad[0]) ** 2 + (s.drone.p[1] - s.mission.pad[1]) ** 2) ** 0.5)
            self.results.append({"status": s.status, "time": s.t, "land": land, "profile": s.prof.key,
                                 "mode": s.mode, "seed": s.world.seed})
        f["results"] = self.results[-200:]
        return f


session, lock = Session(), threading.Lock()


class Handler(BaseHTTPRequestHandler):
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
                                    "modes": list(MODES), "rain": list(RAIN), "motions": list(MOTIONS)})
        self._send(404, {"error": "No encontrado"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                if self.path == "/api/reset":
                    return self._send(200, session.reset(req))
                if self.path == "/api/step":
                    return self._send(200, session.step(req.get("dt", 0.05)))
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
