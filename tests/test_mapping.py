"""Fase 2: el dron construye su mapa con los sensores y vuela sin conocer el mundo."""
import math

import numpy as np

from dron.mapping import CELL, LearnedGrid, OccupancyMap
from dron.sensors import depth_dirs, ray_dirs
from dron.sim import Simulation
from dron.world import make_world


def _scan_poses(w, n, rng):
    out = []
    while len(out) < n:
        x, y = rng.uniform(5, 85), rng.uniform(5, 55)
        o = np.array([x, y, w.surface(x, y) + 2.5])
        if w.clearance(tuple(o), 2) > 0.5:
            out.append(o)
    return out


def test_vectorized_rays_match_the_scalar_ones():
    w = make_world("mixto", 3, terrain="precipicios")
    rng = np.random.default_rng(0)
    for o in _scan_poses(w, 10, rng):
        D = rng.normal(size=(100, 3))
        D /= np.linalg.norm(D, axis=1)[:, None]
        assert np.allclose(w.rays(o, D, 15), [w.ray(tuple(o), tuple(d), 15) for d in D])


def test_map_converges_to_the_real_world():
    w = make_world("mixto", 3, terrain="colinas")
    m = OccupancyMap()
    rng = np.random.default_rng(1)
    poses = _scan_poses(w, 40, rng)
    for o in poses:
        for yaw in np.linspace(0, 2 * math.pi, 4, endpoint=False):
            D = depth_dirs(yaw)
            m.insert(o, D, w.rays(o, D, 15), 15)
    # precisión: toda celda ocupada tiene algo real (obstáculo o terreno) a menos de media diagonal de celda
    P = (np.argwhere(m.occ) + 0.5) * CELL
    geo = [min(w.clearance(tuple(p), 5, ground=False), abs(p[2] - w.terrain.height(p[0], p[1]))) for p in P]
    assert len(P) > 1000 and max(geo) < 0.5 * math.sqrt(3) * CELL + 1e-6
    # exhaustividad: los ecos de un barrido NUEVO (otros rumbos) caen en celdas que el mapa ya tiene ocupadas, o
    # al lado (una celda ocupada solo en parte puede quedar "libre": los rayos que pasan por su hueco la vacían)
    from scipy.ndimage import binary_dilation
    solid = m.solid()
    near = binary_dilation(solid, np.ones((3, 3, 3)))
    inside, beside, total = 0, 0, 0
    for o in poses:
        D = np.array([d for _, d in ray_dirs(0.05)])
        d = w.rays(o, D, 15)
        H = o + D[d < 15] * d[d < 15, None]
        idx = np.floor(np.maximum(H, 0) / CELL).astype(int)
        idx = idx[(idx < m.shape).all(axis=1)]   # los rayos no ven los bordes de la arena: pueden salirse
        inside += int(solid[tuple(idx.T)].sum())
        beside += int(near[tuple(idx.T)].sum())
        total += len(idx)
    assert inside / total > 0.8 and beside / total > 0.97, (inside / total, beside / total)


def test_esdf_clearance_of_a_single_occupied_cell():
    solid = np.zeros((90, 60, 35), bool)
    solid[40, 30, 5] = True
    g = LearnedGrid(solid)
    assert abs(g.clr[43, 30, 5] - 2.5) < 1e-9          # 3 celdas - media celda
    assert abs(g.clearance((43.5, 30.5, 5.5)) - 2.5) < 1e-9
    assert g.clearance((40.5, 30.5, 5.5)) < 0                # dentro del obstáculo
    assert g.clr[10, 10, 20] == 10.5                         # lo más cercano es el borde de la arena


def test_unknown_map_starts_empty_and_the_first_plan_goes_through_unseen_trees():
    s = Simulation(profile="mini", level="bosque", seed=7, terrain="colinas", map_mode="desconocido")
    first = None
    min_clr = math.inf
    while not s.done:
        s.step(0.1)
        if first is None and s.mission.traj is not None:
            first = s.mission.traj
        min_clr = min(min_clr, s.world.clearance(tuple(s.drone.p), 3, ground=False))
    through = sum(s.world.clearance(tuple(p), 1, ground=False) < s.prof.radius for p in first.P)
    assert through > 0                     # no conocía los árboles: el primer plan los atravesaba
    assert s.mission.map_replans > 0       # los vio y replanificó
    assert s.status == "success" and min_clr > s.prof.radius


def test_missions_with_unknown_map():
    for kw in (dict(profile="px4", level="mixto", seed=11, wind_speed=5, gusts=1),
               dict(profile="matrice", level="ciudad", seed=3, goal_kind="azotea"),
               dict(profile="px4", level="bosque", density="extrema", seed=7),
               dict(profile="px4", mode="carrera", motion="rápido", terrain="colinas", seed=2)):
        s = Simulation(map_mode="desconocido", **kw)
        while not s.done:
            s.step(1.0)
        assert s.status == "success", (kw, s.status, s.cause)
