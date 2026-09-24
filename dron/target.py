"""Objetivo en movimiento (modo carrera): un vehículo terrestre con la plataforma encima, y cómo lo sigue el dron.

1. MOVIMIENTO (`TargetMotion`): el vehículo recorre el terreno por tramos rectos que no chocan con obstáculos ni
   suben pendientes imposibles, girando hasta 70° en cada cambio de tramo. Como un vehículo real, acelera y frena con
   límites (2 m/s², 4 m/s² en "variable") y reduce la velocidad en las curvas. Tipos (`MOTIONS`):
     * fija      no se mueve;
     * suave     ~1 m/s, giros suaves;
     * medio     ~2,5 m/s;
     * rápido    ~5 m/s;
     * variable  cambia de velocidad (0-6 m/s) cada pocos segundos, a veces se para y gira bruscamente.
   Todo el recorrido se precalcula con la semilla, así es reproducible: `position(t)` y `velocity(t)`.

2. SEGUIMIENTO (`TargetTracker`): el objetivo emite su posición como un rastreador GNSS (5 Hz, ruido de 0,3 m). El
   dron la filtra con un filtro de Kalman de velocidad constante (estado: posición y velocidad en el plano), el
   modelo clásico para seguir blancos en movimiento. Con él predice dónde estará el objetivo.

3. INTERCEPCIÓN (`intercept_point`): el punto donde se encuentran dron y objetivo si el objetivo mantiene su
   velocidad y el dron vuela en línea recta a su velocidad de crucero: se resuelve |p_obj + v_obj·T − p_dron| = v·T.
   Es la base de la "persecución con adelanto" (lead pursuit) que usan los interceptores.
"""
import math
import random
from typing import Optional, Tuple

import numpy as np

MOTIONS = ("fija", "suave", "medio", "rápido", "variable")
SPEEDS = {"suave": 1.0, "medio": 2.5, "rápido": 5.0, "variable": 3.0}
DT = 0.05
HORIZON = 240.0  # s de recorrido precalculado


class TargetMotion:
    def __init__(self, world, start_xy, kind: str, seed: int):
        if kind not in MOTIONS:
            raise ValueError("Movimiento desconocido %r (usa %s)" % (kind, ", ".join(MOTIONS)))
        self.kind = kind
        self.world = world
        rng = random.Random(seed * 7 + 3)
        n = int(HORIZON / DT) + 1
        self.T = np.arange(n) * DT
        P = np.zeros((n, 2))
        P[0] = start_xy
        if kind == "fija":
            P[:] = start_xy
            self.P, self.V = P, np.zeros((n, 2))
            self.waypoints = [tuple(start_xy)]
            return
        vmax = SPEEDS[kind]
        acc = 4.0 if kind == "variable" else 2.0
        wps = self._waypoints(start_xy, rng, kind)
        self.waypoints = wps
        seg, pos, speed = 1, np.array(start_xy, float), 0.0
        target_speed, next_change = vmax, 0.0
        for k in range(1, n):
            t = k * DT
            if kind == "variable" and t >= next_change:  # cambios bruscos de ritmo
                r = rng.random()
                target_speed = 0.0 if r < 0.15 else rng.uniform(0.5, 6.0)
                next_change = t + rng.uniform(1.5, 5.0)
            if seg >= len(wps):
                P[k:] = pos
                break
            goal = np.array(wps[seg])
            d = goal - pos
            dist = float(np.linalg.norm(d))
            # frenar antes del giro: velocidad de paso por la esquina según el ángulo
            v_corner = target_speed
            if seg + 1 < len(wps):
                a = np.array(wps[seg]) - np.array(wps[seg - 1])
                b = np.array(wps[seg + 1]) - np.array(wps[seg])
                cosang = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
                v_corner = target_speed * max(0.25, (1 + cosang) / 2)
            v_allowed = min(target_speed, math.sqrt(v_corner ** 2 + 2 * acc * dist))
            speed = min(v_allowed, speed + acc * DT) if speed < v_allowed else max(v_allowed, speed - acc * DT)
            step = speed * DT
            if step >= dist:
                pos = goal.copy()
                seg += 1
            else:
                pos = pos + d / dist * step
            P[k] = pos
        self.P = P
        self.V = np.gradient(P, DT, axis=0)

    def _waypoints(self, p0, rng, kind):
        world = self.world
        from .world import LENGTH, WIDTH
        turn = math.radians(35 if kind == "suave" else 70)

        def clear(p):
            if not (3 < p[0] < LENGTH - 3 and 3 < p[1] < WIDTH - 3):
                return False
            if world.terrain.slope(p[0], p[1]) > 0.4:
                return False
            z = world.terrain.height(p[0], p[1]) + 0.5
            return all(o.distance((p[0], p[1], z)) > 1.6 for o in world.obstacles)

        wps = [tuple(p0)]
        heading = rng.uniform(-math.pi, math.pi)
        total = 0.0
        while total < SPEEDS.get(kind, 3.0) * HORIZON * 1.3 and len(wps) < 400:
            best = None
            for _ in range(24):
                h = heading + rng.uniform(-turn, turn)
                length = rng.uniform(5, 15)
                ok_len = 0.0
                while ok_len < length:
                    q = (wps[-1][0] + math.cos(h) * (ok_len + 0.5), wps[-1][1] + math.sin(h) * (ok_len + 0.5))
                    if not clear(q):
                        break
                    ok_len += 0.5
                if ok_len >= 3 and (best is None or ok_len > best[0]):
                    best = (ok_len, h)
                if best and best[0] >= length:
                    break
            if best is None:  # callejón sin salida: media vuelta
                heading += math.pi
                continue
            ok_len, heading = best
            wps.append((wps[-1][0] + math.cos(heading) * ok_len, wps[-1][1] + math.sin(heading) * ok_len))
            total += ok_len
        return wps

    def _idx(self, t: float) -> Tuple[int, float]:
        f = min(max(t, 0.0), self.T[-1]) / DT
        i = min(int(f), len(self.T) - 2)
        return i, f - i

    def position(self, t: float) -> Tuple[float, float]:
        i, f = self._idx(t)
        p = self.P[i] * (1 - f) + self.P[i + 1] * f
        return float(p[0]), float(p[1])

    def velocity(self, t: float) -> Tuple[float, float]:
        i, f = self._idx(t)
        v = self.V[i] * (1 - f) + self.V[i + 1] * f
        return float(v[0]), float(v[1])

    def track(self, t0: float, t1: float, step: float = 0.5):
        """Recorrido entre t0 y t1 (para dibujar lo que le queda por delante)."""
        return [self.position(t) for t in np.arange(t0, t1, step)]


class TargetTracker:
    """Filtro de Kalman de velocidad constante del objetivo (plano horizontal). Estado: x, y, vx, vy."""

    def __init__(self, q_acc: float = 1.5, r_pos: float = 0.3):
        self.x: Optional[np.ndarray] = None
        self.P = np.eye(4) * 10.0
        self.q, self.r = q_acc, r_pos
        self.t = None

    def update(self, t: float, meas_xy):
        if self.x is None:
            self.x = np.array([meas_xy[0], meas_xy[1], 0.0, 0.0])
            self.t = t
            return
        self.predict_to(t)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + np.eye(2) * self.r ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(meas_xy) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def predict_to(self, t: float):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q ** 2
        self.t = t

    def state_at(self, t: float):
        """(posición, velocidad) predichas para el instante t, sin modificar el filtro."""
        dt = t - self.t
        return self.x[:2] + self.x[2:] * dt, self.x[2:].copy()


def intercept_point(p_drone, p_target, v_target, speed: float, max_t: float = 30.0):
    """(punto de intercepción en el plano, tiempo T) resolviendo |p_obj + v_obj·T − p_dron| = speed·T."""
    r = np.asarray(p_target[:2]) - np.asarray(p_drone[:2])
    v = np.asarray(v_target[:2])
    a = v @ v - speed ** 2
    b = 2 * (r @ v)
    c = r @ r
    T = None
    if abs(a) < 1e-9:
        T = -c / b if b < 0 else None
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            roots = [(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)]
            pos = [x for x in roots if x > 0]
            T = min(pos) if pos else None
    if T is None or T > max_t:
        T = max_t if T is None else T
        T = min(T, max_t)
    return np.asarray(p_target[:2]) + v * T, T
