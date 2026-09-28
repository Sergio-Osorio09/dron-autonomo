"""Física de 6 grados de libertad con cuatro motores (dynamics.Multirotor6DOF) y el autopiloto interno de PX4."""
import math

import numpy as np
import pytest

from dron.dynamics import Multirotor6DOF
from dron.params import G, PROFILES
from dron.world import make_world

DT = 0.005
WORLD = make_world("mixto", 5)


def hover(key):
    d = Multirotor6DOF(PROFILES[key], (20.0, 20.0, 0.0), 0.0)
    d.cmd_thrust, d.cmd_z = G * 1.2, np.array([0.0, 0.0, 1.0])
    for _ in range(200):
        d.step(DT, np.zeros(3), WORLD)
    d.cmd_thrust = G
    for _ in range(200):
        d.step(DT, np.zeros(3), WORLD)
    return d


@pytest.mark.parametrize("key", list(PROFILES))
def test_hover_shares_thrust_equally(key):
    d = hover(key)
    assert not d.on_ground and max(d.u) - min(d.u) < 0.02          # los cuatro motores, iguales
    assert abs(sum(d.f) - PROFILES[key].mass * G) < 0.05 * PROFILES[key].mass * G   # sostienen el peso


@pytest.mark.parametrize("key", list(PROFILES))
def test_tilt_step_is_fast_and_does_not_overshoot(key):
    d = hover(key)
    th = math.radians(15)
    d.cmd_thrust, d.cmd_z = G / math.cos(th), np.array([0.0, -math.sin(th), math.cos(th)])
    tilts = []
    for _ in range(200):
        d.step(DT, np.zeros(3), WORLD)
        tilts.append(math.degrees(d.tilt))
    t90 = next(i for i, t in enumerate(tilts) if t > 13.5) * DT
    assert t90 < 0.4 and max(tilts) < 16.0                           # MC_ROLL_P 6,5: ~0,25 s, sin pasarse
    assert d.R[1, 2] < -0.2                                         # se inclina hacia donde se pide (−y)


@pytest.mark.parametrize("key", list(PROFILES))
def test_yaw_turn_with_px4_profile(key):
    d = hover(key)
    d.cmd_yaw = math.pi / 2
    yaws = []
    for _ in range(800):
        d.step(DT, np.zeros(3), WORLD)
        yaws.append(math.degrees(d.yaw))
    assert max(yaws) < 95.0 and abs(yaws[-1] - 90.0) < 3.0          # MPC_YAWRAUTO_ACC: sin pasarse de rumbo


def test_mixer_sacrifices_yaw_before_roll_and_pitch():
    d = hover("px4")
    d.cmd_thrust = G * 1.9                                           # casi todo el empuje: poco margen
    th = math.radians(30)
    d.cmd_z = np.array([math.sin(th), 0.0, math.cos(th)])
    d.cmd_yaw = math.pi
    for _ in range(40):
        d.step(DT, np.zeros(3), WORLD)
    thr, roll, pitch, yawc = d.sticks
    assert all(0.0 <= u <= 1.0 for u in d.u)
    assert abs(yawc) <= 0.05                                         # la guiñada es lo primero que cede
    assert d.tilt > math.radians(10)                                 # la inclinación sí se consigue


def test_payload_makes_the_x650_heavier_and_is_limited_by_max_takeoff_weight():
    """Carga útil: más masa, menos empuje por kilo; el X650 no despega con 4 kg (pasaría de sus 6,3 kg)."""
    from dron.params import with_payload
    p = PROFILES["x650"]
    q = with_payload(p, 2.0)
    assert q.mass == pytest.approx(p.mass + 2.0) and q.twr < p.twr and q.hover_power_w > p.hover_power_w
    assert q.mass * q.twr == pytest.approx(p.mass * p.twr)          # el empuje máximo no cambia
    with pytest.raises(ValueError):
        with_payload(p, 4.0)


def test_x650_does_not_take_off_when_the_forecast_gust_is_above_its_limit():
    """Comprobación de viento (params.gust_peak, como COM_WIND_MAX de PX4): racha prevista = viento × (1 + 0,6 × nivel)."""
    from dron.sim import Simulation
    assert Simulation(profile="x650", wind_speed=10, gusts=1, seed=1).status == "grounded"   # 16 m/s > 13
    assert Simulation(profile="x650", wind_speed=8, gusts=1, seed=1).status == "flying"      # 12,8 m/s
    assert Simulation(profile="x650", wind_speed=12, gusts=0, seed=1).status == "flying"     # sin ráfagas
