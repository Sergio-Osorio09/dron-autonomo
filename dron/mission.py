"""Máquina de estados de la misión (como los modos automáticos de PX4: Takeoff → Mission → Land).

Dos modos:

ATERRIZAR   en tierra → despegue → crucero → aproximación → aterrizaje → aterrizado
  * despegue: sube en vertical a MPC_TKO_SPEED hasta 2,5 m sobre el terreno.
  * crucero: sigue la trayectoria planificada hasta 2,5 m por encima de la plataforma (que puede estar en el suelo,
    en una azotea o en una cima). Si el error de seguimiento pasa de 3 m, replanifica desde donde está.
    La holgura con los obstáculos crece con la incertidumbre de posición: 0,6 m + 2σ del filtro (máx. 2,5 m).
  * aproximación: se estabiliza encima. Con aterrizaje de precisión corrige con la cámara inferior (como IR-LOCK).
  * aterrizaje: baja a MPC_LAND_SPEED; si el viento lo desplaza más de 35 cm, deja de bajar y se realinea.
    ÉXITO si queda posado sobre la plataforma.

CARRERA     en tierra → despegue → carrera → [persecución] → meta
  * despega hasta 1,5 m y vuela lo más rápido posible (90 % de la velocidad del fabricante) una trayectoria que NO
    frena al final: cruza la meta como una puerta. ÉXITO al pasar a menos de 1 m del punto meta (1 m sobre la plataforma,
    o un punto en el aire). Si la pasa de largo, vuelve a por ella.
  * OBJETIVO EN MOVIMIENTO: el vehículo emite su posición (rastreador GNSS, 5 Hz); un filtro de Kalman de velocidad
    constante estima dónde está y hacia dónde va (target.py). El dron vuela hacia el PUNTO DE INTERCEPCIÓN (donde
    estará el objetivo cuando llegue) y replanifica cada segundo si ese punto se mueve. A menos de 12 m, y SOLO si
    hay línea de visión libre (si no, la persecución en línea recta atravesaría el árbol que el vehículo rodea), pasa
    a PERSECUCIÓN PREDICTIVA: apunta a la posición predicha 0,5 s por delante y va a la velocidad del objetivo más
    una velocidad de cierre, hasta cruzar la puerta que lleva encima.

En ambos modos la velocidad se limita a la que permite frenar dentro del alcance de los telémetros
(v ≤ √(2·a·(alcance − radio − 1 m))): la regla de oro de los drones autónomos. Con lluvia el alcance baja y el dron
vuela más despacio; en carrera, un dron con sensores de 12 m no puede ir a 14 m/s.
"""
import math

import numpy as np

from .planning import plan, segment_clear
from .target import TargetTracker, intercept_point

CRUISE_AGL = 2.5
PAD_RADIUS = 0.75
GATE_RADIUS = 1.0
MODES = ("aterrizar", "carrera")


class Mission:
    def __init__(self, world, grid, prof, precision: bool = True, mode: str = "aterrizar", sensor_range: float = None):
        if mode not in MODES:
            raise ValueError("Modo de misión desconocido %r" % mode)
        self.world, self.grid, self.prof, self.precision, self.mode = world, grid, prof, precision, mode
        self.phase = "en tierra"
        g = world.goal
        self.pad = np.array(g[:2])                       # centro de la plataforma (o del punto en el aire)
        self.pad_z = g[2]                                # altura de la plataforma (o del punto)
        self.gate = np.array([g[0], g[1], g[2] + (1.0 if world.goal_support else 0.0)])  # punto a cruzar en carrera
        self.moving = world.moving
        self.tracker = TargetTracker() if self.moving else None
        self.intercept = None
        self.last_plan_t = -1.0
        self.speed_now = 0.0
        self.los_t, self.los_clear = -1.0, False
        self.pad_est = self.pad.copy()
        self.traj = None
        self.t0 = 0.0
        self.hold = None
        self.replans = 0
        self.yaw = None
        self.plan_error = None
        self.est = None       # filtro de Kalman (lo asigna la simulación): su covarianza fija el margen de seguridad
        self.margin = 0.6
        # no volar más rápido de lo que dejan ver los sensores: poder frenar dentro del alcance de los telémetros
        rng = sensor_range or prof.sensor_range
        self.v_sense = math.sqrt(2 * prof.acc_hor * max(rng - prof.radius - 1.0, 1.0))

    def gate_at(self, t):
        """Puerta a cruzar en el instante t: la real si el objetivo está quieto, la estimada si se mueve."""
        if not self.moving:
            return self.gate
        p, _ = self.tracker.state_at(t)
        return np.array([p[0], p[1], self.world.terrain.height(p[0], p[1]) + self.world.goal[2]
                         - self.world.terrain.height(*self.world.goal[:2]) + 1.0])

    def _plan(self, t, p_est):
        # margen que crece con la incertidumbre de posición del filtro (2 sigmas), como hacen los planificadores
        # que tienen en cuenta la covarianza
        sigma = math.sqrt(max(self.est.P[0, 0] + self.est.P[1, 1], 0.0)) if self.est is not None else 0.0
        self.margin = min(0.6 + 2.0 * sigma, 2.5)
        if self.mode == "carrera":
            v = end = min(0.9 * self.prof.v_max, self.v_sense)
            goal = self.gate
            if self.moving:  # volar hacia donde ESTARÁ el objetivo
                p_t, v_t = self.tracker.state_at(t)
                xy, _ = intercept_point(p_est, p_t, v_t, 0.8 * v)
                self.last_plan_t = t
                old, old_t0 = self.traj, self.t0
                for cand in (xy, p_t):  # si el punto de intercepción no es alcanzable, ir a por la posición actual
                    goal = np.array([cand[0], cand[1], self.world.terrain.height(cand[0], cand[1]) + 2.0])
                    self.traj = plan(self.world, self.grid, self.prof, p_est, goal, v, end, self.margin,
                                     self.speed_now)
                    if self.traj is not None:
                        self.intercept = goal
                        self.t0 = t
                        self.phase = "carrera"
                        return True
                self.traj, self.t0 = old, old_t0  # sin camino ahora: seguir con la trayectoria anterior
                if old is None:
                    self.plan_error = "sin camino"
                    return False
                return True
        else:
            goal = np.array([self.pad_est[0], self.pad_est[1], self.pad_z + CRUISE_AGL])
            v, end = min(self.prof.v_cruise, self.v_sense), 0.0
        new = plan(self.world, self.grid, self.prof, p_est, goal, v, end, self.margin, self.speed_now)
        if new is None:
            if self.traj is None:  # ni siquiera hay un primer plan
                self.plan_error = "sin camino"
                return False
            return False  # replanificar falló (p. ej. desplazado junto a una pared): seguir con la trayectoria anterior
        self.traj = new
        self.t0 = t
        self.phase = "carrera" if self.mode == "carrera" else "crucero"
        return True

    def reference(self, t, est_p, est_v, readings):
        """(p_ref, v_ref, a_ref, yaw_ref) para el control."""
        prof = self.prof
        self.speed_now = min(float(np.linalg.norm(est_v)), 0.9 * prof.v_max)
        if self.moving and "target" in readings:
            self.tracker.update(t, readings["target"])
        if "pad_rel" in readings and self.precision:  # la cámara ve la plataforma: mejor estimación relativa
            self.pad_est += (est_p[:2] + readings["pad_rel"] - self.pad_est) * 0.2
        zero = np.zeros(3)
        if self.phase == "en tierra":
            self.phase = "despegue"
            self.hold = est_p.copy()
            self.hold_ground = self.world.surface(est_p[0], est_p[1])
            self.t0 = t
        if self.phase == "despegue":
            alt = 1.5 if self.mode == "carrera" else CRUISE_AGL
            top = self.hold_ground + alt
            speed = prof.v_up if self.mode == "carrera" else prof.tko_speed
            z = min(self.hold[2] + speed * (t - self.t0), top)
            if est_p[2] > top - 0.3:
                self._plan(t, est_p)
            return np.array([self.hold[0], self.hold[1], z]), np.array([0, 0, speed if z < top else 0.0]), zero, self.yaw
        if self.moving and self.phase in ("carrera", "reintento", "persecución"):
            gate = self.gate_at(t)
            if t - self.los_t > 0.2:  # ¿línea de visión libre hasta el objetivo? (5 veces por segundo)
                self.los_t = t
                self.los_clear = segment_clear(self.world, est_p, gate, self.prof.radius + 0.8, 0.4)
            if np.linalg.norm(gate[:2] - est_p[:2]) < 12.0 and self.los_clear:  # cerca y a la vista: persecución
                self.phase = "persecución"
                p_t, v_t = self.tracker.state_at(t + 0.5)
                aim = np.array([p_t[0], p_t[1], gate[2]])
                d = aim - est_p
                dist = float(np.linalg.norm(d))
                close = min(self.v_sense, max(2.0, 0.8 * dist))  # velocidad de cierre
                v_ref = np.array([v_t[0], v_t[1], 0.0]) + d / max(dist, 1e-6) * close
                if math.hypot(v_ref[0], v_ref[1]) > 1.0:
                    self.yaw = math.atan2(v_ref[1], v_ref[0])
                return aim, v_ref, np.zeros(3), self.yaw
            if self.phase == "persecución":  # se ha alejado: volver a interceptar
                self._plan(t, est_p)
            elif t - self.last_plan_t > 1.0:  # el punto de intercepción cambia: replanificar cada segundo
                p_t, v_t = self.tracker.state_at(t)
                xy, _ = intercept_point(est_p, p_t, v_t, 0.8 * min(0.9 * prof.v_max, self.v_sense))
                if self.intercept is None or np.linalg.norm(xy - self.intercept[:2]) > 2.0 or self.phase == "reintento":
                    self._plan(t, est_p)
                else:
                    self.last_plan_t = t
        if self.phase in ("crucero", "carrera"):
            p, v, a = self.traj.sample(t - self.t0)
            if np.linalg.norm(p - est_p) > 3.0:  # nos hemos desviado mucho: replanificar desde aquí
                self.replans += 1
                if self._plan(t, est_p):
                    p, v, a = self.traj.sample(0.0)
            if math.hypot(v[0], v[1]) > 1.0:
                self.yaw = math.atan2(v[1], v[0])
            if t - self.t0 >= self.traj.duration:
                self.phase = "reintento" if self.mode == "carrera" else "aproximación"
                self.t0 = t
            return p, v, a, self.yaw
        if self.phase == "reintento":  # carrera: la pasó de largo, vuelve a por ella
            return self.gate, zero, zero, self.yaw
        target = np.array([self.pad_est[0], self.pad_est[1], self.pad_z + CRUISE_AGL])
        if self.phase == "aproximación":
            err = math.hypot(*(target[:2] - est_p[:2]))
            if (err < 0.3 and np.linalg.norm(est_v) < 0.4) or t - self.t0 > 6.0:
                self.phase = "aterrizaje"
                self.t0 = t
                self.hold = est_p.copy()
            return target, zero, zero, self.yaw
        if self.phase == "aterrizaje":
            err = math.hypot(*(target[:2] - est_p[:2]))
            if err > 0.35 and est_p[2] > self.pad_z + 0.6:  # desplazado por el viento: dejar de bajar y realinear
                self.hold[2] = est_p[2]
                self.t0 = t
                return np.array([target[0], target[1], est_p[2]]), zero, zero, self.yaw
            z = max(self.hold[2] - prof.land_speed * (t - self.t0), self.pad_z - 0.5)
            return np.array([target[0], target[1], z]), np.array([0, 0, -prof.land_speed]), zero, self.yaw
        return est_p, zero, zero, self.yaw
