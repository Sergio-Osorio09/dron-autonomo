"""Fases 4 y 5: objetivo que huye y enjambre de drones."""
import math

import numpy as np

from dron.sim import Simulation
from dron.swarm import Swarm
from dron.target import EvaderMotion
from dron.world import make_world


def test_evader_flees_and_hides():
    w = make_world("ciudad", 3, race=True, motion="huye")
    ev = w.motion
    assert isinstance(ev, EvaderMotion)
    p0 = np.array(ev.position(0.0))
    drone = np.array([p0[0] + 12.0, p0[1], w.terrain.height(p0[0] + 12, p0[1]) + 6.0])  # a 12 m y a la vista
    for k in range(1, 81):     # 4 s con el dron quieto
        ev.step(k * 0.05, drone)
    p1 = np.array(ev.position(4.0))
    assert np.linalg.norm(p1 - drone[:2]) > np.linalg.norm(p0 - drone[:2]) + 5.0   # se aleja
    assert max(math.hypot(*ev.velocity(t)) for t in np.arange(0, 4, 0.1)) <= EvaderMotion.VMAX + 0.3
    z = w.terrain.height(*p1) + 0.5
    assert all(o.distance((p1[0], p1[1], z)) > 1.0 for o in w.obstacles)          # sin atravesar obstáculos


def test_chase_an_evader_with_the_camera_only():
    s = Simulation(profile="mini", level="ciudad", seed=2, mode="carrera", motion="huye", terrain="colinas",
                   wind_speed=3)
    while not s.done:
        s.step(0.5)
    assert s.status == "success", (s.status, s.cause)
    assert s.mission.lost_count >= 1          # se escondió al menos una vez y lo volvió a encontrar


def test_swarm_shares_the_search_and_splits_the_area():
    sw = Swarm(3, profile="px4", level="mixto", seed=11, search="bayesiana")
    assert len(sw.sims) == 3 and all(s.searcher is sw.sims[0].searcher for s in sw.sims)
    starts = [s.drone.p.copy() for s in sw.sims]
    assert min(np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(starts) for b in starts[i + 1:]) > 2.0
    min_sep = math.inf
    goals_seen = set()
    while not sw.done:
        sw.step(0.25)
        ps = [s.drone.p for s in sw.sims]
        min_sep = min(min_sep, min(np.linalg.norm(a - b) for i, a in enumerate(ps) for b in ps[i + 1:]))
        gs = [m.search_goal for m in sw.missions if m.search_goal is not None]
        if len(gs) == 3:
            goals_seen.add(min(np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(gs) for b in gs[i + 1:]) > 4.0)
    assert sw.status == "success", (sw.status, sw.cause)
    assert min_sep > 2 * sw.sims[0].prof.radius          # nunca se tocan
    assert True in goals_seen                            # se reparten la zona (no van al mismo punto)
    assert sum(m.phase == "espera" for m in sw.missions) == 2


def test_team_pursuit_catches_what_one_drone_cannot():
    kw = dict(profile="px4", level="mixto", seed=1001, mode="carrera", motion="huye", terrain="colinas",
              wind_speed=4, gusts=1)
    team = Swarm(3, **kw)
    min_sep = math.inf
    while not team.done:
        team.step(0.5)
        ps = [s.drone.p for s in team.sims]
        min_sep = min(min_sep, min(np.linalg.norm(a - b) for i, a in enumerate(ps) for b in ps[i + 1:]))
    assert team.status == "success" and team.t < 120, (team.status, team.cause, team.t)
    assert min_sep > 2 * team.sims[0].prof.radius
