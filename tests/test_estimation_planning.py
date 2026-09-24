import numpy as np

from dron.estimator import Estimator
from dron.params import PROFILES
from dron.planning import ClearanceGrid, plan
from dron.sim import Simulation
from dron.world import make_world


def test_kalman_filter_beats_raw_gps():
    rng = np.random.default_rng(0)
    est = Estimator(np.zeros(3), 0.75, 1.0)
    p, v = np.zeros(3), np.array([3.0, 1.0, 0.0])
    raw, filt = [], []
    for k in range(4000):  # 20 s a 200 Hz con aceleración nula y GPS a 10 Hz
        p = p + v * 0.005
        est.predict(rng.normal(0, 0.1, 3), 0.005)
        if k % 20 == 0:
            g = p + rng.normal(0, 0.75, 3)
            est.gps(g, v + rng.normal(0, 0.1, 3))
            if k > 400:
                raw.append(np.linalg.norm(g - p))
                filt.append(np.linalg.norm(est.p - p))
    assert np.mean(filt) < 0.6 * np.mean(raw)


def test_innovation_gate_rejects_outliers():
    est = Estimator(np.zeros(3), 0.5, 1.0)
    est.baro(0.1)
    before = est.rejected
    est.baro(50.0)  # imposible: la puerta la descarta
    assert est.rejected == before + 1 and abs(est.p[2]) < 1.0


def test_plan_is_clear_and_respects_speed_limits():
    prof = PROFILES["mini"]
    w = make_world("bosque", 5, terrain="colinas")
    g = ClearanceGrid(w)
    tr = plan(w, g, prof, (w.start[0], w.start[1], w.start[2] + 2.5), (w.goal[0], w.goal[1], w.goal[2] + 2.5))
    assert tr is not None
    assert tr.speed.max() <= prof.v_cruise + 1e-6 and tr.speed[0] == 0 and tr.speed[-1] == 0
    assert all(w.clearance(tuple(p), 5, ground=True) > prof.radius for p in tr.P)
    acc = np.diff(tr.speed ** 2) / (2 * np.diff(tr.s))
    assert np.abs(acc).max() <= prof.acc_hor + 1e-6


def test_full_mission_lands_on_the_pad():
    for prof in ("mini", "matrice"):
        s = Simulation(profile=prof, level="mixto", seed=11, wind_speed=5, gusts=1, noise="realista")
        while not s.done:
            s.step(1.0)
        assert s.status == "success", (prof, s.status, s.cause)


def test_goals_on_roof_peak_and_in_the_air():
    for kw in (dict(level="ciudad", goal_kind="azotea"), dict(terrain="montañoso", goal_kind="cima"),
               dict(goal_kind="aire", mode="carrera", terrain="colinas")):
        s = Simulation(profile="matrice", seed=3, wind_speed=3, **kw)
        assert s.world.goal_kind == kw["goal_kind"]
        while not s.done:
            s.step(1.0)
        assert s.status == "success", (kw, s.status, s.cause)


def test_race_is_faster_than_landing():
    times = {}
    for mode in ("aterrizar", "carrera"):
        s = Simulation(profile="matrice", seed=4, mode=mode, noise="ideal")
        while not s.done:
            s.step(1.0)
        assert s.status == "success"
        times[mode] = s.t
    assert times["carrera"] < times["aterrizar"]


def test_terrain_rays_and_surface():
    w = make_world("mixto", 7, terrain="precipicios")
    x, y = 45.0, 30.0
    top = w.surface(x, y)
    assert abs(w.ray((x, y, top + 20), (0, 0, -1), 40) - 20) < 0.3 or top < w.terrain.height(x, y) + 20
