import math

import numpy as np

from dron.control import PositionController
from dron.dynamics import Multirotor
from dron.params import G, PROFILES
from dron.wind import Wind
import random

from dron.world import Box, Terrain, World

EMPTY = World(Terrain("plano", random.Random(0)), [], (10, 10, 0.0), (60, 30, 0.05), "suelo", "mixto", 0)


def hover(prof, z=5.0):
    d = Multirotor(prof, (35, 20, 0.0))
    d.p[2], d.on_ground = z, False
    return d


def test_drag_gives_the_manufacturer_top_speed():
    for prof in PROFILES.values():
        d = hover(prof)
        tilt = prof.tilt_max
        d.cmd_z = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        d.z_body = d.cmd_z.copy()
        d.cmd_thrust = d.thrust = G / math.cos(tilt)
        d.p[0] = 1.0
        for _ in range(6000):
            d.step(0.005, np.zeros(3), EMPTY)
            d.p[0] = 1.0  # que no se salga de la arena
        assert abs(d.v[0] - prof.v_max) < 0.05 * prof.v_max, (prof.key, d.v[0])


def test_hover_needs_one_g_and_tilt_is_limited():
    prof = PROFILES["px4"]
    d, c = hover(prof), PositionController(prof)
    for _ in range(2000):
        t = c.update(d.p.copy(), d.v.copy(), np.array([35, 20, 5.0]), np.zeros(3), np.zeros(3), 0.005)
        d.cmd_z, d.cmd_thrust = t / np.linalg.norm(t), float(t @ d.z_body)
        d.step(0.005, np.zeros(3), EMPTY)
    assert abs(d.thrust - G) < 0.05 and np.linalg.norm(d.p - [35, 20, 5]) < 0.02
    t = c.update(d.p.copy(), d.v.copy(), np.array([60, 20, 5.0]), np.zeros(3), np.zeros(3), 0.005)
    assert math.degrees(math.acos(t[2] / np.linalg.norm(t))) <= prof.tilt_max_deg + 1e-6


def test_integral_term_rejects_steady_wind():
    prof = PROFILES["mini"]
    d, c = hover(prof), PositionController(prof)
    wind = np.array([6.0, 0.0, 0.0])
    for _ in range(8000):
        t = c.update(d.p.copy(), d.v.copy(), np.array([35, 20, 5.0]), np.zeros(3), np.zeros(3), 0.005)
        d.cmd_z, d.cmd_thrust = t / np.linalg.norm(t), float(t @ d.z_body)
        d.step(0.005, wind, EMPTY)
    assert np.linalg.norm(d.p[:2] - [35, 20]) < 0.1   # sin integral se quedaría desplazado a sotavento
    assert d.tilt > math.radians(3)                   # se inclina contra el viento


def test_hard_touchdown_crashes_soft_one_lands():
    prof = PROFILES["px4"]
    for vz, expect in ((-3.0, "terreno"), (-0.7, None)):
        d = hover(prof, 0.5)
        d.v[2] = vz
        d.cmd_thrust = d.thrust = G  # empuje de vuelo estacionario: toca el suelo a la velocidad que lleva
        for _ in range(200):
            d.step(0.005, np.zeros(3), EMPTY)
        assert d.crashed == expect


def test_dryden_gusts_have_zero_mean_and_scale_with_level():
    stds = []
    for level in (1, 3):
        w = Wind(8.0, 0.0, level, seed=1)
        xs = [w.step((30, 30, 10.0), 8.0, 0.02)[1] for _ in range(60000)]  # componente lateral: solo ráfaga (20 min)
        stds.append(float(np.std(xs)))
        assert abs(float(np.mean(xs))) < 0.5 * stds[-1]
    assert stds[1] > 2 * stds[0] > 0


def test_landing_on_a_roof_is_not_a_crash_but_hitting_a_wall_is():
    prof = PROFILES["px4"]
    roof = Box(30, 40, 15, 25, 8.0)
    w = World(Terrain("plano", random.Random(0)), [roof], (5, 5, 0.0), (35, 20, 8.05), "azotea", "ciudad", 0)
    d = Multirotor(prof, (35, 20, 9.0))
    d.on_ground, d.v[2] = False, -0.7
    d.cmd_thrust = d.thrust = G
    for _ in range(600):
        d.step(0.005, np.zeros(3), w)
    assert d.crashed is None and d.on_ground and abs(d.p[2] - (8.0 + d.gear)) < 1e-6
    d = Multirotor(prof, (27, 20, 3.0))
    d.on_ground, d.v[0] = False, 1.0
    d.cmd_thrust = d.thrust = G
    for _ in range(1000):
        d.step(0.005, np.zeros(3), w)
    assert d.crashed == "obstáculo"


def test_wind_is_sheltered_behind_a_building_and_faster_on_its_roof():
    from dron.wind import Wind
    b = Box(30, 36, 25, 35, 12.0)
    w = World(Terrain("plano", random.Random(0)), [b], (5, 5, 0.0), (80, 30, 0.05), "suelo", "ciudad", 0)
    wind = Wind(8.0, 0.0, 0, 0, w)  # sopla hacia +x
    upwind = np.hypot(*wind.mean((20, 30, 4))[:2])
    lee = np.hypot(*wind.mean((40, 30, 4))[:2])
    roof = np.hypot(*wind.mean((33, 30, 13))[:2])
    free_13 = np.hypot(*Wind(8.0, 0.0, 0, 0, EMPTY).mean((33, 30, 13))[:2])
    assert lee < 0.5 * upwind and roof > free_13


def test_collision_prevention_never_pushes_into_another_obstacle():
    """Un edificio que acaba de dejar atrás no debe empujarlo contra la pared de enfrente (fallo de la fase 1)."""
    from dron.control import CP_ACC_FACTOR, CP_MARGIN, collision_prevention
    prof = PROFILES["matrice"]
    rays = [{"label": "geovalla", "dir": (-1.0, 0.0, 0.0), "dist": 2.6},    # pared delante (va hacia -x)
            {"label": "0", "dir": (1.0, 0.0, 0.0), "dist": 2.0},            # edificio detrás
            {"label": "90", "dir": (0.0, 1.0, 0.0), "dist": 40.0}]
    v_now = np.array([-3.9, 0.0, 0.0])
    v_sp, a = collision_prevention(np.array([1.9, 0.0, 0.0]), np.zeros(3), rays, prof, v_now)
    room = 2.6 - (prof.radius + CP_MARGIN + 0.3 * 3.9)
    assert -v_sp[0] <= math.sqrt(2 * CP_ACC_FACTOR * prof.acc_hor * max(room, 0)) + 1e-9   # respeta la pared
    assert a[0] > 0            # y frena de verdad (va demasiado rápido hacia la pared)


def test_collision_prevention_brakes_when_already_too_fast():
    from dron.control import collision_prevention
    prof = PROFILES["px4"]
    rays = [{"label": "0", "dir": (1.0, 0.0, 0.0), "dist": 5.0}, {"label": "180", "dir": (-1.0, 0.0, 0.0), "dist": 15.0}]
    v_sp, a = collision_prevention(np.array([6.0, 0.0, 0.0]), np.zeros(3), rays, prof, np.array([6.0, 0.0, 0.0]))
    assert v_sp[0] < 3.0 and a[0] <= -prof.acc_hor + 1e-9    # pide menos velocidad Y deceleración máxima
