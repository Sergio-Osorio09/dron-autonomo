"""Varios drones que se reparten la búsqueda (fase 5).

Cada dron es una simulación completa (física, sensores, filtro de Kalman, mapa, control y misión) y todos avanzan a
la vez, paso a paso, en el mismo mundo. Lo que comparten es lo que compartirían por radio:
  * la CREENCIA de la búsqueda (un solo mapa de probabilidad): cada imagen de cualquier dron la actualiza;
  * su posición estimada y el punto al que van (10 veces por segundo en un dron real; aquí, a cada paso).

Reparto del trabajo (search.py):
  * barrido     franjas intercaladas: la k-ésima para el dron k mod n;
  * fronteras y bayesiana: cada dron solo elige en su celda de VORONOI (lo que tiene más cerca que ningún otro dron:
                cobertura de Cortés, Martínez, Karatas y Bullo, 2004) y nunca a menos de 8 m del punto que ya ha
                elegido otro (subasta voraz: el primero que lo pide se lo queda; CBBA es la versión con consenso).

Evitar chocar entre ellos:
  * separación vertical: cada dron busca en su capa de altura (+2 m por dron), como la separación por altitud del
    control de tráfico aéreo;
  * evitación reactiva: la posición compartida de los demás entra en Collision Prevention como un obstáculo más, y
    como lo hacen los dos, cada uno cede su parte (la idea de "reciprocidad" de ORCA, van den Berg et al., 2011,
    sin su programa lineal).
Si dos drones se tocan de verdad (distancia < suma de radios), los dos se estrellan.

La misión acaba cuando el dron que ha encontrado la meta aterriza (o la cruza en carrera); los demás esperan.

PERSECUCIÓN EN EQUIPO (fases 4 + 5, movimiento "huye"): el vehículo huye del dron más cercano; si un dron lo ve,
todos actualizan su filtro (avistamiento compartido por radio); el primero va al punto de intercepción y los demás
se abren a los lados (cerco). Gana el equipo cuando cualquiera cruza la puerta. En persecución-evasión con
obstáculos, un solo perseguidor no siempre puede atrapar a quien se esconde; varios pueden cerrarle las salidas.
"""
import math
from typing import List

import numpy as np

from .sim import Simulation

SWARM_MAX = 4
LAYER = 2.0          # m de separación vertical entre las capas de búsqueda


def _starts(world, n, radius):
    """Posiciones de salida: la del mundo y otras a 3-4 m, en suelo llano y libre."""
    x0, y0, _ = world.start
    out = [world.start]
    k = 0
    while len(out) < n and k < 200:
        ang, r = 2.4 * k, 3.0 + 0.5 * (k // 8)
        k += 1
        x, y = x0 + r * math.cos(ang), y0 + r * math.sin(ang)
        z = world.terrain.height(x, y)
        if world.terrain.slope(x, y) > 0.2 or world.clearance((x, y, z + 0.5), 2.0, ground=False) < radius + 0.8:
            continue
        if all(math.hypot(x - q[0], y - q[1]) > 2.5 for q in out):
            out.append((x, y, z))
    return out


class Swarm:
    def __init__(self, drones: int = 2, **cfg):
        if not 1 <= drones <= SWARM_MAX:
            raise ValueError("Número de drones entre 1 y %d" % SWARM_MAX)
        if cfg.get("search", "no") == "no" and cfg.get("motion") != "huye":
            raise ValueError("El enjambre sirve para buscar o para perseguir al que huye")
        lead = Simulation(**cfg)
        starts = _starts(lead.world, drones, lead.prof.radius)
        self.sims: List[Simulation] = [lead]
        for k in range(1, len(starts)):
            shared = {"world": lead.world, "grid": lead.grid, "searcher": lead.searcher, "start": starts[k], "id": k,
                      "agl": LAYER * k}
            self.sims.append(Simulation(**dict(cfg, seed=lead.world.seed), shared=shared))
        self.missions = [s.mission for s in self.sims]
        for m in self.missions:
            m.team = self
        self.config = dict(lead.config, drones=len(self.sims))
        self.status, self.cause = "flying", None

    # --- lo que usan el servidor y la evaluación, como con una sola Simulation
    @property
    def lead(self) -> Simulation:
        """El dron que ha cruzado la meta o la ha encontrado (o el primero)."""
        for s in self.sims:
            if s.status == "success":
                return s
        k = getattr(self.sims[0].searcher, "found_by", None)
        return self.sims[k or 0]

    def __getattr__(self, name):   # drone, mission, prof, world, mode, searcher, distance... del dron principal
        if name in ("sims", "missions"):
            raise AttributeError(name)
        return getattr(self.lead, name)

    @property
    def t(self):
        return self.sims[0].t

    @property
    def done(self) -> bool:
        return self.status != "flying"

    def step(self, duration: float):
        from .sim import DT
        for _ in range(max(1, int(round(duration / DT)))):
            if self.done:
                break
            self._tick()

    def _tick(self):
        for s in self.sims:   # "radio": cada uno recibe la posición estimada de los demás
            s.neighbors = [(o.est.p.copy(), o.prof.radius) for o in self.sims if o is not s]
        motion = self.sims[0].world.motion
        if motion is not None and motion.kind == "huye":   # el vehículo ve a todos los drones
            motion.threats = [s.drone.p.copy() for s in self.sims]
        for s in self.sims:
            if not s.done:
                s._tick()
        for i, a in enumerate(self.sims):   # choque entre drones (con las posiciones REALES)
            for b in self.sims[i + 1:]:
                if np.linalg.norm(a.drone.p - b.drone.p) < a.prof.radius + b.prof.radius:
                    for s in (a, b):
                        s.status, s.cause = "crash", "choque entre drones"
        sightings = [m.last_meas for m in self.missions if m.last_meas is not None]
        if sightings:          # avistamiento compartido: el más reciente llega a todos
            t, xy, z = max(sightings, key=lambda q: q[0])
            for m in self.missions:
                m.share_sighting(t, xy, z)
        lead = self.lead
        if lead.status == "success":
            self.status, self.cause = "success", None
        elif lead.done:
            self.status, self.cause = lead.status, lead.cause
        elif any(s.status == "crash" for s in self.sims):
            s = next(s for s in self.sims if s.status == "crash")
            self.status, self.cause = "crash", s.cause
        elif all(s.done for s in self.sims):
            self.status, self.cause = self.sims[0].status, self.sims[0].cause

    def frame(self, full: bool = False):
        f = self.sims[0].frame(full)   # la cámara sigue siempre al dron 1; los demás van en "swarm"
        f["status"], f["cause"] = self.status, self.cause
        f["swarm"] = []
        for s in self.sims:
            xb, _, zb = s.drone.attitude()
            goal = s.mission.search_goal
            f["swarm"].append({"id": s.id, "pos": s.drone.p.round(2).tolist(), "est": s.est.p.round(2).tolist(),
                               "x_body": xb.round(3).tolist(), "z_body": zb.round(3).tolist(),
                               "phase": s.mission.phase, "goal": None if goal is None else goal.round(1).tolist(),
                               "speed": round(float(np.linalg.norm(s.drone.v)), 2),
                               "status": s.status})
        f["lead"] = self.lead.id
        f["found_t"] = self.lead.mission.found_t
        if full:
            f["config"] = self.config
        return f
