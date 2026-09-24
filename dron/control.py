"""Control de posición en cascada, como el de PX4 (mc_pos_control):

    posición --P--> velocidad pedida --PID--> aceleración pedida --> vector de empuje --> inclinación + empuje

  * Se sigue una referencia de trayectoria con prealimentación (feed-forward): p_ref, v_ref y a_ref. El P de
    posición y el PID de velocidad solo corrigen el error; la mayor parte de la aceleración viene de a_ref.
  * Ganancias por defecto de PX4 (MPC_XY_P, MPC_XY_VEL_*_ACC, MPC_Z_P, MPC_Z_VEL_*_ACC).
  * El término integral es lo que compensa el viento constante: sin él, el dron se quedaría "a sotavento".
    Tiene anti-windup: deja de integrar cuando la salida está saturada.
  * El vector de empuje se limita a la inclinación máxima dando prioridad a la vertical (no caer es lo primero),
    igual que PX4.
  * Todo usa el estado ESTIMADO (filtro de Kalman), nunca el real: el ruido de los sensores afecta al vuelo.
  * Collision Prevention (como la de PX4): con las distancias de los telémetros, que son RELATIVAS y no dependen del
    GPS, se limita la velocidad pedida hacia cada obstáculo para poder frenar siempre a tiempo
    (v_hacia ≤ √(2·a·(d - d_seguridad))), y si está demasiado cerca se aleja. Es lo que evita chocar cuando el GPS
    se equivoca en medio metro y el planificador pasaba justo al lado de un árbol.
"""
import math

import numpy as np

from .params import G, Profile


class PositionController:
    def __init__(self, profile: Profile):
        self.prof = profile
        g = profile.gains
        self.kp = np.array([g["xy_p"], g["xy_p"], g["z_p"]])
        self.kv = np.array([g["xy_vel_p"], g["xy_vel_p"], g["z_vel_p"]])
        self.ki = np.array([g["xy_vel_i"], g["xy_vel_i"], g["z_vel_i"]])
        self.kd = np.array([g["xy_vel_d"], g["xy_vel_d"], g["z_vel_d"]])
        self.integ = np.zeros(3)
        self.prev_v = None
        self.dv = np.zeros(3)
        self.saturated = False
        self.cp_active = False
        self._cp_hits = 0

    def reset(self):
        self.integ[:] = 0.0
        self.prev_v = None

    def update(self, p, v, p_ref, v_ref, a_ref, dt, rays=None):
        prof = self.prof
        # posición -> velocidad
        v_sp = v_ref + self.kp * (p_ref - p)
        if rays:
            v_sp, a_ref = collision_prevention(v_sp, a_ref.copy(), rays, prof, v)
            self.cp_active = self._cp_hits > 0
        else:
            self.cp_active = False
        h = math.hypot(v_sp[0], v_sp[1])
        if h > prof.v_max:
            v_sp[:2] *= prof.v_max / h
        v_sp[2] = float(np.clip(v_sp[2], -prof.v_down, prof.v_up))
        # velocidad -> aceleración (PID; derivada sobre la medida, filtrada)
        e = v_sp - v
        if self.prev_v is not None:
            self.dv += ((v - self.prev_v) / dt - self.dv) * min(1.0, dt / 0.05)
        self.prev_v = v.copy()
        if not self.saturated:
            self.integ += e * dt
        lim = np.array([G * math.tan(prof.tilt_max)] * 2 + [G * 0.5])
        self.integ = np.clip(self.integ, -lim / np.maximum(self.ki, 1e-6), lim / np.maximum(self.ki, 1e-6))
        a_sp = a_ref + self.kv * e + self.ki * self.integ - self.kd * self.dv
        # aceleración -> vector de empuje, con límite de inclinación y prioridad vertical
        t = a_sp + np.array([0.0, 0.0, G])
        t[2] = max(t[2], 0.1 * G)
        t_max = prof.twr * G
        t[2] = min(t[2], t_max)
        max_h = min(t[2] * math.tan(prof.tilt_max), math.sqrt(max(t_max ** 2 - t[2] ** 2, 0.0)))
        th = math.hypot(t[0], t[1])
        self.saturated = th > max_h
        if self.saturated:
            t[:2] *= max_h / th
        return t  # vector de empuje deseado (m/s²) en ejes del mundo


def collision_prevention(v_sp, a_ref, rays, prof, v_now=None):
    """Limita la velocidad pedida en la dirección de cada rayo que ve un obstáculo cercano (PX4 CollisionPrevention).

    rays: [{"dir": (x, y, z), "dist": d, "label": ...}] de los telémetros (distancias medidas, con su ruido).
    Los rayos que miran hacia abajo (abajo y frontal-inferior) no cuentan: casi siempre ven el SUELO, y el suelo
    lo gestionan el despegue y el aterrizaje (si no, el dron no podría aterrizar). Los obstáculos bajos delante
    los ve el anillo horizontal.
    """
    # distancia de seguridad dinámica: margen fijo + lo que recorre durante el tiempo de reacción (0,3 s)
    speed = float(np.linalg.norm(v_now)) if v_now is not None else 0.0
    d_safe = prof.radius + 0.8 + 0.3 * speed
    acc = 0.5 * prof.acc_hor  # frenada conservadora: el dron tarda en responder (inercia de actitud y del control)
    reach = max((r["dist"] for r in rays), default=prof.sensor_range)  # alcance efectivo (menor con lluvia)
    v_cap = math.sqrt(2 * acc * max(reach - d_safe, 0.5))                # poder frenar dentro de lo que se ve
    n = np.linalg.norm(v_sp)
    if n > v_cap:
        v_sp = v_sp * (v_cap / n)
    for r in rays:
        u = np.array(r["dir"])
        if u[2] < -0.3 or r["dist"] >= reach - 1e-6:
            continue
        room = r["dist"] - d_safe
        along = float(v_sp @ u)
        v_lim = math.sqrt(2 * acc * max(room, 0.0))
        if room < 0:                      # demasiado cerca: alejarse un poco
            v_lim = -min(1.5, -room * 3.0)
        if along > v_lim:
            v_sp = v_sp - (along - v_lim) * u
            a_along = float(a_ref @ u)
            if a_along > 0:
                a_ref = a_ref - a_along * u
    return v_sp, a_ref
