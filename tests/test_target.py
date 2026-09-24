import math

import numpy as np

from dron.sim import Simulation
from dron.target import TargetTracker, intercept_point
from dron.world import make_world


def test_target_motion_speeds_and_obstacle_free_track():
    for kind, vmax in (("suave", 1.0), ("medio", 2.5), ("rápido", 5.0)):
        w = make_world("mixto", 5, "suelo", "colinas", race=True, motion=kind)
        assert w.moving
        speeds = [math.hypot(*w.motion.velocity(t)) for t in np.arange(0, 60, 0.5)]
        assert max(speeds) <= vmax * 1.05 and np.mean(speeds) > 0.3 * vmax
        for t in np.arange(0, 60, 1.0):  # el vehículo nunca atraviesa obstáculos
            x, y, z = w.goal_at(t)
            assert all(o.distance((x, y, z - 0.1)) > 1.0 for o in w.obstacles)


def test_kalman_tracker_estimates_velocity():
    rng = np.random.default_rng(1)
    tr = TargetTracker()
    for k in range(50):  # 10 s a 5 Hz, objetivo a (2, -1) m/s con ruido de 0,3 m
        t = k * 0.2
        tr.update(t, np.array([2 * t, -t]) + rng.normal(0, 0.3, 2))
    _, v = tr.state_at(10.0)
    assert np.linalg.norm(v - [2, -1]) < 0.6  # ajustado para objetivos que maniobran: responde rápido, algo más ruidoso


def test_intercept_point_is_reachable_in_time():
    p, T = intercept_point((0, 0), (20, 0), (0, 3), 6.0)
    assert abs(np.linalg.norm(p) - 6.0 * T) < 1e-6 and abs(p[1] - 3 * T) < 1e-6


def test_race_against_moving_targets():
    for motion in ("suave", "rápido", "variable"):
        s = Simulation(profile="px4", seed=2, mode="carrera", motion=motion, terrain="colinas", wind_speed=3)
        while not s.done:
            s.step(1.0)
        assert s.status == "success", (motion, s.status, s.cause)
