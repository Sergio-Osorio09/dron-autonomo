"""Prueba de maniobras: la destreza del autopiloto con la física de 4 motores, como los vuelos de prueba con los que se
ajusta un dron real (escalones, paradas, círculos, giros), en campo abierto y sin obstáculos.

  escalón         de estacionario, la referencia salta 10 m hacia delante: tiempo de subida (10-90 %), cuánto se
                  pasa y cuánto tarda en quedarse a menos de 0,3 m
  parada          volando al 80 % de su velocidad máxima, "¡para!": distancia de frenada y cuánto retrocede
  círculo         seguir un círculo de 8 m de radio a 6 m/s: error medio (RMS) y máximo
  giro de 180°    tiempo hasta quedar a menos de 5° del rumbo pedido

El controlador es el de siempre (control.PositionController: posición → velocidad → aceleración → empuje) con el
estado REAL (se prueba el control, no la estimación) y, con 6 grados de libertad, el autopiloto interno de PX4 y los
4 motores. `--gains archivo.json` prueba unas ganancias entrenadas (un "best" de eval/evolve.py).

    python eval/maniobras.py                      # los tres drones, física de 4 motores y la simple
"""
import argparse
import dataclasses
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dron.control import PositionController  # noqa: E402
from dron.dynamics import Multirotor, Multirotor6DOF  # noqa: E402
from dron.params import PROFILES  # noqa: E402
from dron.world import make_world  # noqa: E402

DT = 0.005


def open_world():
    w = make_world("mixto", 1)
    w.obstacles = []
    w._arr_n = None
    return w


def rig(key, physics, gains=None):
    prof = PROFILES[key]
    if gains and physics == "6dof":
        g = dict(prof.inner)
        g["att_p"] *= gains.get("att_k", 1.0)
        g["rate_p"] *= gains.get("rate_pk", 1.0)
        g["rate_i"] *= gains.get("rate_ik", 1.0)
        g["rate_d"] *= gains.get("rate_dk", 1.0)
        prof = dataclasses.replace(prof, inner=g)
    d = (Multirotor6DOF if physics == "6dof" else Multirotor)(prof, (20.0, 30.0, 0.0), 0.0)
    c = PositionController(prof)
    if gains:
        c.kp[:2] *= gains.get("xy_p_k", 1.0)
        c.kv[:2] *= gains.get("vel_p_k", 1.0)
    return d, c


def fly(d, c, w, ref, T):
    """ref(t) → (p_ref, v_ref, a_ref, yaw_ref). Devuelve las posiciones, velocidades y rumbos."""
    P, V, Y = [], [], []
    for k in range(int(T / DT)):
        t = k * DT
        p_ref, v_ref, a_ref, yaw = ref(t)
        tv = c.update(d.p.copy(), d.v.copy(), p_ref, v_ref, a_ref, DT)
        d.cmd_z = tv / np.linalg.norm(tv)
        d.cmd_thrust = float(tv @ d.z_body)
        if yaw is not None:
            d.cmd_yaw = yaw
        d.step(DT, np.zeros(3), w)
        P.append(d.p.copy())
        V.append(d.v.copy())
        Y.append(d.yaw)
        if d.crashed:
            break
    return np.array(P), np.array(V), np.array(Y)


def takeoff(d, c, w):
    z = np.array([20.0, 30.0, 5.0])
    fly(d, c, w, lambda t: (z, np.zeros(3), np.zeros(3), 0.0), 6.0)
    return z


def m_step(key, physics, gains):
    w = open_world()
    d, c = rig(key, physics, gains)
    z = takeoff(d, c, w)
    goal = z + np.array([10.0, 0.0, 0.0])
    P, _, _ = fly(d, c, w, lambda t: (goal, np.zeros(3), np.zeros(3), 0.0), 8.0)
    x = P[:, 0] - z[0]
    t10 = next(i for i, v in enumerate(x) if v > 1.0) * DT
    t90 = next(i for i, v in enumerate(x) if v > 9.0) * DT
    settle = next((i * DT for i in range(len(x)) if all(abs(x[j] - 10) < 0.3 for j in range(i, len(x)))), None)
    return {"subida": t90 - t10, "se_pasa": max(0.0, x.max() - 10.0), "estable": settle}


def m_stop(key, physics, gains):
    w = open_world()
    d, c = rig(key, physics, gains)
    z = takeoff(d, c, w)
    v = 0.8 * PROFILES[key].v_max
    run = 40.0 / v                               # 40 m a v (la arena mide 90 m: queda sitio para frenar)
    start = np.array([8.0, 30.0, z[2]])

    def ref(t):
        if t < run:
            return start + np.array([v * t, 0.0, 0.0]), np.array([v, 0.0, 0.0]), np.zeros(3), 0.0
        return stop_at, np.zeros(3), np.zeros(3), 0.0
    d.p[:] = start
    d.v[:] = (v, 0.0, 0.0)
    stop_at = start + np.array([v * run, 0.0, 0.0])
    P, V, _ = fly(d, c, w, ref, run + 8.0)
    i0 = int(run / DT)
    xs = P[i0:, 0]
    return {"frenada": xs.max() - xs[0], "retrocede": xs.max() - xs[-1]}


def m_circle(key, physics, gains):
    w = open_world()
    d, c = rig(key, physics, gains)
    z = takeoff(d, c, w)
    R, v = 8.0, 6.0
    om = v / R
    cen = z + np.array([-R, 0.0, 0.0])

    def ref(t):
        a = om * t
        p = cen + R * np.array([math.cos(a), math.sin(a), 0.0])
        vv = v * np.array([-math.sin(a), math.cos(a), 0.0])
        acc = -om * om * R * np.array([math.cos(a), math.sin(a), 0.0])
        return p, vv, acc, None
    d.v[:] = (0.0, v, 0.0)
    P, _, _ = fly(d, c, w, ref, 2 * math.pi / om * 1.5)
    ts = np.arange(len(P)) * DT
    err = [np.linalg.norm(P[i] - ref(ts[i])[0]) for i in range(len(P)) if ts[i] > 2.0]
    return {"error_rms": float(np.sqrt(np.mean(np.square(err)))), "error_max": float(np.max(err))}


def m_yaw(key, physics, gains):
    w = open_world()
    d, c = rig(key, physics, gains)
    z = takeoff(d, c, w)
    _, _, Y = fly(d, c, w, lambda t: (z, np.zeros(3), np.zeros(3), math.pi - 0.01), 6.0)
    err = np.degrees(np.abs((Y - (math.pi - 0.01) + math.pi) % (2 * math.pi) - math.pi))
    return {"giro_180": next((i * DT for i in range(len(err)) if all(err[j] < 5 for j in range(i, len(err)))), None)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="mini,px4,matrice")
    ap.add_argument("--physics", default="6dof,simple")
    ap.add_argument("--gains", default=None, help="JSON con un 'best' de eval/evolve.py (ganancias entrenadas)")
    ap.add_argument("--autotune", action="store_true", help="con las ganancias de dron/autotune.json de cada dron")
    a = ap.parse_args()
    gains = None
    if a.gains:
        with open(a.gains, encoding="utf-8") as f:
            d = json.load(f)
        gains = d.get("best", d)
    fmt = lambda v: "—" if v is None else "%.2f" % v
    print("| dron | física | escalón 10 m: subida · se pasa · estable | parada (80 % v_max): frenada · retrocede "
          "| círculo 8 m a 6 m/s: error RMS · máx | giro 180° |")
    print("|---|---|---|---|---|---|")
    for key in a.profiles.split(","):
        for ph in a.physics.split(","):
            g = gains
            if a.autotune:
                at = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dron", "autotune.json")
                with open(at, encoding="utf-8") as f:
                    g = json.load(f).get(key, {}).get("genes")
            s, p, cc, y = m_step(key, ph, g), m_stop(key, ph, g), m_circle(key, ph, g), m_yaw(key, ph, g)
            print("| %s | %s | %s s · %s m · %s s | %s m · %s m | %s m · %s m | %s s |" % (
                key, ph, fmt(s["subida"]), fmt(s["se_pasa"]), fmt(s["estable"]), fmt(p["frenada"]),
                fmt(p["retrocede"]), fmt(cc["error_rms"]), fmt(cc["error_max"]), fmt(y["giro_180"])), flush=True)


if __name__ == "__main__":
    main()
