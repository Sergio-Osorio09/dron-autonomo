"""Simulación completa de un vuelo: une mundo, viento, física, sensores, estimación, misión y control.

Bucle a 200 Hz (dt = 5 ms), en este orden cada paso:
    viento → física (estado REAL) → IMU → predicción del filtro → sensores lentos y correcciones del filtro
    → [mapa: barridos insertados desde la posición ESTIMADA] → misión (referencia) → control (con el estado
    ESTIMADO) → órdenes a motores y actitud

Modo de mapa (`map_mode`):
  * "conocido"    (fase 1) el planificador usa la rejilla de holgura del mundo real;
  * "desconocido" (fase 2) el dron empieza sin saber nada y construye un mapa de ocupación con los telémetros y
    la cámara de profundidad (mapping.py). El planificador, la altura mínima de crucero y la altura del despegue
    salen de ese mapa. El telémetro inferior ya no corrige la altura absoluta (haría falta conocer el terreno):
    la altura sale del GPS y el barómetro, como en PX4 cuando no hay estimación del terreno.
  El mundo real se sigue usando para la física, los sensores y la evaluación, y para generar un mundo con camino.
"""
import math
import random
from typing import Dict, Optional

import numpy as np

from .control import PositionController
from .dynamics import Multirotor
from .estimator import Estimator
from .mapping import OccupancyMap
from .mission import GATE_RADIUS, PAD_RADIUS, Mission
from .params import PROFILES
from .planning import ClearanceGrid, astar, MARGIN
from .sensors import NOISE_LEVELS, RAIN, Sensors
from .wind import Wind
from .world import LENGTH, WIDTH, make_world

MAP_MODES = ("conocido", "desconocido")
DT = 0.005
MAX_TIME = 150.0


def make_flyable_world(level, seed, radius, goal_kind="suelo", terrain="plano", density="normal", race=False,
                       motion="fija"):
    """Mundo en el que A* encuentra camino de la salida a la meta (prueba semillas consecutivas)."""
    seed = seed if seed is not None else random.randrange(1 << 30)
    for k in range(30):
        w = make_world(level, seed + k, goal_kind, terrain, density, race, motion)
        grid = ClearanceGrid(w)
        s = (w.start[0], w.start[1], w.start[2] + 2.5)
        g = (w.goal[0], w.goal[1], w.goal[2] + (1.0 if race and w.goal_support else 0.0 if race else 2.5))
        if astar(grid, s, g, radius + MARGIN) is not None:
            return w, grid
    raise RuntimeError("No se encontró un mundo con camino: prueba otra semilla o menos densidad")


class Simulation:
    def __init__(self, profile: str = "px4", level: str = "mixto", seed: Optional[int] = None,
                 wind_speed: float = 0.0, wind_dir: float = 0.0, gusts: int = 0, noise: str = "realista",
                 precision_landing: bool = True, collision_prevention: bool = True, terrain: str = "plano",
                 density: str = "normal", goal_kind: str = "suelo", mode: str = "aterrizar", rain: str = "no",
                 motion: str = "fija", map_mode: str = "conocido"):
        if map_mode not in MAP_MODES:
            raise ValueError("Modo de mapa desconocido %r" % map_mode)
        if profile not in PROFILES:
            raise ValueError("Perfil desconocido %r" % profile)
        if noise not in NOISE_LEVELS:
            raise ValueError("Nivel de ruido desconocido %r" % noise)
        if rain not in RAIN:
            raise ValueError("Lluvia desconocida %r" % rain)
        self.prof = PROFILES[profile]
        self.world, self.grid = make_flyable_world(level, seed, self.prof.radius, goal_kind, terrain, density,
                                                   mode == "carrera", motion)
        seed = self.world.seed
        self.config = {"profile": profile, "level": level, "seed": seed, "wind_speed": wind_speed,
                       "wind_dir": wind_dir, "gusts": gusts, "noise": noise, "precision_landing": precision_landing,
                       "collision_prevention": collision_prevention, "terrain": terrain, "density": density,
                       "goal_kind": self.world.goal_kind, "mode": mode, "rain": rain,
                       "motion": self.world.motion.kind if self.world.motion else "fija", "map_mode": map_mode}
        self.mode = mode
        self.collision_prevention = collision_prevention
        yaw = random.Random(seed).uniform(-math.pi, math.pi)
        self.drone = Multirotor(self.prof, self.world.start, yaw)
        self.drone.drag_mult = RAIN[rain]["drag"]
        self.wind = Wind(wind_speed, wind_dir, gusts, seed, self.world)
        self.sensors = Sensors(self.prof, noise, seed + 1, rain)
        self.est = Estimator(self.drone.p.copy(), self.prof.gps_sigma, NOISE_LEVELS[noise])
        self.ctrl = PositionController(self.prof)
        self.map = OccupancyMap() if map_mode == "desconocido" else None
        self.sensors.depth_on = self.map is not None
        self.mission = Mission(self.world, self.grid, self.prof, precision_landing, mode, self.sensors.range,
                               self.map)
        self.mission.yaw = yaw
        self.mission.est = self.est
        self.t = 0.0
        self.status = "flying"   # flying | success | missed | crash | timeout
        self.cause = None
        self.wind_now = np.zeros(3)
        self.ref = (self.drone.p.copy(), np.zeros(3), np.zeros(3))
        self.max_speed = 0.0
        self.history = []
        self._last_hist = -1.0
        self._next_target = 0.0
        self.distance = 0.0
        self._map_sent = -1
        self._map_sent_t = -9.0
        self._depth_pts = None

    @property
    def done(self) -> bool:
        return self.status != "flying"

    def step(self, duration: float):
        n = max(1, int(round(duration / DT)))
        for _ in range(n):
            if self.done:
                break
            self._tick()

    def _tick(self):
        d, est = self.drone, self.est
        self.t += DT
        airspeed = float(np.linalg.norm(d.v - self.wind_now))
        self.wind_now = np.array(self.wind.step(d.p, airspeed, DT))
        d.step(DT, self.wind_now, self.world)
        if d.crashed:
            self.status, self.cause = "crash", d.crashed
            return
        est.predict(self.sensors.imu(d.a, DT), DT)
        m = self.mission
        r = self.sensors.read(self.t, d.p, d.v, d.yaw, self.world, (m.pad[0], m.pad[1], m.pad_z))
        if self.world.moving and self.t >= self._next_target:  # el vehículo emite su posición a 5 Hz
            self._next_target = self.t + 0.2
            gx, gy, _ = self.world.goal_at(self.t)
            s = 0.3 * max(self.sensors.m, 0.1)
            r["target"] = (gx + self.sensors.rng.gauss(0, s), gy + self.sensors.rng.gauss(0, s))
            if self.map is not None:  # el rastreador GNSS también da la altura (1,5 veces menos precisa)
                r["target_z"] = self.world.goal_at(self.t)[2] + self.sensors.rng.gauss(0, 1.5 * s)
        if "gps_pos" in r:
            est.gps(r["gps_pos"], r["gps_vel"])
        if "baro" in r:
            est.baro(r["baro"])
        if self.map is not None:  # el mapa se construye desde donde el dron CREE estar
            if "rays" in r:
                self.map.insert(est.p, [x["dir"] for x in r["rays"]], [x["dist"] for x in r["rays"]], self.sensors.range)
            if "depth" in r:
                dep = r["depth"]
                self.map.insert(est.p, dep["dirs"], dep["dist"], dep["max"])
                hit = np.isfinite(dep["dist"]) & (dep["dist"] < dep["max"] - 1e-3)
                self._depth_pts = (d.p + dep["dirs"][hit] * dep["dist"][hit, None]).round(1)
        elif "rays" in r:
            down = next(x for x in r["rays"] if x["label"] == "down")
            est.range_down(down["dist"], self.world.surface(est.p[0], est.p[1]))
        if m.phase == "aterrizaje" and d.on_ground:
            m.phase = "aterrizado"
            dist = math.hypot(*(d.p[:2] - m.pad))
            on_pad = dist < PAD_RADIUS and abs(d.p[2] - d.gear - m.pad_z) < 0.3
            self.status = "success" if on_pad else "missed"
            self.cause = None if on_pad else "fuera de la plataforma (%.1f m)" % dist
            d.cmd_thrust = 0.0
            return
        gate = m.gate
        if self.world.moving:  # la puerta real va sobre el vehículo
            gx, gy, gz = self.world.goal_at(self.t)
            gate = np.array([gx, gy, gz + 1.0])
            self.gate_now = gate
        if m.phase in ("carrera", "reintento", "persecución") and np.linalg.norm(d.p - gate) < GATE_RADIUS + (0.3 if self.world.moving else 0):
            m.phase = "meta"
            self.status = "success"
            return
        p_ref, v_ref, a_ref, yaw_ref = m.reference(self.t, est.p.copy(), est.v.copy(), r)
        if m.plan_error:
            self.status, self.cause = "crash", m.plan_error
            return
        if m.phase in ("crucero", "carrera", "persecución", "reintento"):
            # altura mínima sobre la superficie (debajo y 0,8 s por delante): no rozar el borde de una meseta
            ahead = est.p + est.v * 0.8
            if self.map is None:
                floor = max(self.world.surface(est.p[0], est.p[1]), self.world.surface(ahead[0], ahead[1])) + 1.0
            else:  # la superficie que ha visto (debajo, y delante hasta 2 m por encima de su altura)
                floor = max(self.map.surface(est.p[0], est.p[1], est.p[2]),
                            self.map.surface(ahead[0], ahead[1], est.p[2] + 2.0)) + 1.0
            if p_ref[2] < floor:
                p_ref = p_ref.copy()
                p_ref[2] = floor
                v_ref = v_ref.copy()
                v_ref[2] = max(v_ref[2], 0.0)
        self.ref = (p_ref, v_ref, a_ref)
        rays = None
        if self.collision_prevention:
            # geovalla (como Geofence de PX4): los bordes de la arena son paredes virtuales que los telémetros no ven
            ex, ey = est.p[0], est.p[1]
            fence = [{"label": "geovalla", "dir": dv, "dist": dd} for dv, dd in (
                ((1.0, 0.0, 0.0), LENGTH - ex), ((-1.0, 0.0, 0.0), ex), ((0.0, 1.0, 0.0), WIDTH - ey), ((0.0, -1.0, 0.0), ey))
                if dd < self.sensors.range]
            rays = self.sensors.last_rays + fence
        t_vec = self.ctrl.update(est.p.copy(), est.v.copy(), p_ref, v_ref, a_ref, DT, rays)
        d.cmd_z = t_vec / np.linalg.norm(t_vec)
        d.cmd_thrust = float(t_vec @ d.z_body)
        if yaw_ref is not None:
            d.cmd_yaw = yaw_ref
        speed = math.hypot(d.v[0], d.v[1])
        self.distance += float(np.linalg.norm(d.v)) * DT
        self.max_speed = max(self.max_speed, speed)
        if self.t - self._last_hist >= 0.1:
            self._last_hist = self.t
            self.history.append({"t": round(self.t, 2), "speed": round(speed, 2),
                                 "ref_speed": round(math.hypot(v_ref[0], v_ref[1]), 2),
                                 "alt": round(d.p[2], 2), "err": round(float(np.linalg.norm(est.p - d.p)), 3),
                                 "wind": round(float(np.linalg.norm(self.wind_now[:2])), 2)})
            self.history = self.history[-600:]
        if self.t > MAX_TIME:
            self.status, self.cause = "timeout", "tiempo agotado"

    # ------------------------------------------------------------------ para la interfaz / evaluación
    def battery(self) -> float:
        return max(0.0, 100.0 * (1 - self.drone.energy_wh / self.prof.battery_wh))

    def frame(self, full: bool = False) -> Dict:
        d, m = self.drone, self.mission
        xb, yb, zb = d.attitude()
        out = {
            "t": round(self.t, 3), "status": self.status, "cause": self.cause, "phase": m.phase,
            "pos": d.p.round(3).tolist(), "vel": d.v.round(3).tolist(), "acc": d.a.round(3).tolist(),
            "est": self.est.p.round(3).tolist(), "est_err": float(np.linalg.norm(self.est.p - d.p)),
            "x_body": xb.round(4).tolist(), "z_body": zb.round(4).tolist(), "yaw": d.yaw,
            "tilt_deg": math.degrees(d.tilt), "thrust_g": d.thrust / 9.81,
            "speed": float(math.hypot(d.v[0], d.v[1])), "vz": float(d.v[2]),
            "ref": self.ref[0].round(3).tolist(), "ref_speed": float(math.hypot(*self.ref[1][:2])),
            "wind": self.wind_now.round(2).tolist(), "wind_mean": list(self.wind.mean(d.p)),
            "mode": self.mode, "gate": self.mission.gate.round(3).tolist(), "pad_z": self.mission.pad_z,
            "goal_now": list(self.world.goal_at(self.t)), "moving": self.world.moving,
            "target_vel": list(self.world.motion.velocity(self.t)) if self.world.moving else [0.0, 0.0],
            "intercept": None if self.mission.intercept is None else self.mission.intercept.round(2).tolist(),
            "target_track": [list(map(lambda v: round(v, 2), q)) for q in self.world.motion.track(self.t, self.t + 15, 1.0)]
            if self.world.moving else None,
            "gps": None if self.sensors.last_gps is None else self.sensors.last_gps.round(2).tolist(),
            "rays": [{"label": x["label"], "dist": round(x["dist"], 2),
                      "end": (d.p + np.array(x["dir"]) * x["dist"]).round(2).tolist()} for x in self.sensors.last_rays],
            "battery": self.battery(), "power_w": d.power(), "replans": m.replans,
            "pad": m.pad.round(3).tolist(), "pad_est": m.pad_est.round(3).tolist(),
            "max_speed": self.max_speed, "rejected": self.est.rejected, "ekf_resets": self.est.resets,
            "distance": self.distance, "map_mode": self.config["map_mode"],
        }
        if self.map is not None:
            out["map_replans"] = m.map_replans
            out["explored"] = self.map.explored_fraction()
            out["blocked"] = m.blocked is not None
            if self._depth_pts is not None:
                out["depth_pts"] = self._depth_pts.tolist()
                self._depth_pts = None
            # el mapa entero, como mucho una vez por segundo y solo si ha cambiado
            if full or (self.map.version != self._map_sent and self.t - self._map_sent_t >= 1.0) or self.done:
                out["map"] = self.map.to_dict()
                self._map_sent, self._map_sent_t = self.map.version, self.t
        if full:
            out["world"] = self.world.to_dict()
            out["profile"] = self.prof.to_dict()
            out["config"] = self.config
            out["wind_field"] = self.wind.field() if self.wind.speed > 0 else None
            out["history"] = self.history
        if m.traj is not None and (full or getattr(self, "_sent_traj", None) is not m.traj):
            out["trajectory"] = m.traj.to_dict()
            self._sent_traj = m.traj
        return out
