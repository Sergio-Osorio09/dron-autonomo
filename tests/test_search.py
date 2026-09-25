"""Fase 3: búsqueda de una meta de la que solo se conoce la zona."""
import math

import numpy as np

from dron.search import CELL, Searcher, in_cone, p_detect
from dron.sensors import Sensors
from dron.params import PROFILES
from dron.sim import Simulation
from dron.world import Box, Terrain, World
import random

AREA = (20.0, 60.0, 10.0, 40.0)


def flat_searcher(strategy="bayesiana"):
    return Searcher(AREA, strategy, lambda x, y: 0.0, lambda o, pts: np.ones(len(pts), bool))


def test_detection_model():
    assert abs(float(p_detect(10.0)) - 0.9) < 1e-9          # cerca: 0,9
    assert float(p_detect(50.0)) == 0.0                     # lejos: nada
    assert float(p_detect(20.0)) > float(p_detect(30.0)) > 0
    down_front = np.array([math.cos(math.radians(60)), 0.0, -math.sin(math.radians(60))])
    assert in_cone(down_front, 0.0)[0] and not in_cone(np.array([-1.0, 0.0, 0.0]), 0.0)[0]


def test_bayesian_update_lowers_what_it_sees_and_keeps_a_distribution():
    s = flat_searcher()
    p0 = s.P.copy()
    s.glimpse(np.array([30.0, 25.0, 8.0]), 0.0, None)     # mira hacia +x, hacia abajo
    assert abs(s.P.sum() - 1.0) < 1e-9
    i = int(np.argmin(abs(s.cx - 35.0)))
    j = int(np.argmin(abs(s.cy - 25.0)))
    far = (int(np.argmin(abs(s.cx - 58.0))), int(np.argmin(abs(s.cy - 12.0))))
    assert s.P[i, j] < 0.2 * p0[i, j]                       # lo que ha visto sin encontrarlo baja
    assert s.P[far] > p0[far]                               # y lo que no ha visto sube
    assert 0 < s.covered() < 0.5


def test_occlusion_blocks_detection():
    wall = Box(30, 31, 0, 60, 20.0)
    w = World(Terrain("plano", random.Random(0)), [wall], (5, 5, 0.0), (40.0, 30.0, 0.05), "suelo", "ciudad", 0)
    sen = Sensors(PROFILES["px4"], "ideal", 1)
    behind = sum(sen.detect(np.array([22.0, 30.0, 8.0]), 0.0, w)["rel"] is not None for _ in range(200))
    w.obstacles = []
    clear = sum(sen.detect(np.array([22.0, 30.0, 8.0]), 0.0, w)["rel"] is not None for _ in range(200))
    expected = 200 * float(p_detect(math.hypot(18.0, 7.95)))   # ≈ 0,72 a 19,7 m
    assert behind == 0 and abs(clear - expected) < 25, (behind, clear, expected)


def test_lawnmower_covers_the_area():
    s = flat_searcher("barrido")
    pts = [s.next_goal(np.array([20.0, 10.0, 8.0]))[:2] for _ in range(6)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert min(xs) < 25 and max(xs) > 55                    # pasadas a lo largo del lado largo (x)
    assert len(set(np.round(ys, 1))) >= 2                   # en varias franjas
    assert all(AREA[2] <= y <= AREA[3] for y in ys)


def test_rejected_goals_are_not_proposed_again():
    s = flat_searcher("bayesiana")
    g = s.next_goal(np.array([20.0, 10.0, 8.0]))
    s.reject(g)
    for _ in range(3):
        assert np.linalg.norm(s.next_goal(np.array([20.0, 10.0, 8.0]))[:2] - g[:2]) > 4.0


def test_search_missions_find_and_land():
    for strategy, mm in (("barrido", "conocido"), ("fronteras", "desconocido"), ("bayesiana", "desconocido")):
        s = Simulation(profile="px4", level="mixto", seed=11, search=strategy, map_mode=mm)
        x0, x1, y0, y1 = s.area
        assert x0 <= s.world.goal[0] <= x1 and y0 <= s.world.goal[1] <= y1
        assert s.mission.pad_est is None                    # no sabe dónde está
        while not s.done:
            s.step(1.0)
        assert s.status == "success" and s.mission.found_t is not None, (strategy, s.status, s.cause)
        assert np.linalg.norm(s.searcher.detections[-1]["pos"][:2] - s.world.goal[:2]) < 2 * CELL
