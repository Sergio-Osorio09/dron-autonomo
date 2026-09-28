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

from . import tuning
from .params import G, Profile

# valores de fábrica (los de cada dron salen de tuning.params: pueden estar ajustados)
XY_MARGIN = 0.3                              # MPC_THR_XY_MARG por defecto
# PREALIMENTACIÓN DEL ARRASTRE (Faessler, Franchi y Scaramuzza, RA-L 2018: planitud diferencial con arrastre). El
# arrastre frena al dron con k·|v|·v (a 8 m/s, ~4,4 m/s² en el X650, casi toda la aceleración del planificador); sin
# prealimentarlo, el PID de velocidad solo lo compensa cuando ya va por detrás (el clásico, un 30 % del tiempo, con
# 1,5 m/s de error medio). Se suma de antemano el que corresponde a la velocidad pedida, con el coeficiente del modelo
# del dron (en uno real, identificado en vuelo). DRON_DRAG_FF=0: sin ella
DRAG_FF = float(__import__("os").environ.get("DRON_DRAG_FF", 1.0))
CP_ACC_FACTOR = tuning.FACTORY["cp_acc"]    # fracción de la aceleración con la que frena Collision Prevention
CP_MARGIN = tuning.FACTORY["cp_margin"]     # m de seguridad además del radio
CP_REACTION = tuning.FACTORY["cp_react"]    # s de reacción: la distancia de seguridad crece con la velocidad


def cp_reaction(prof: Profile, P: dict) -> float:
    """Tiempo de reacción de Collision Prevention: el ajustado, pero nunca menos de lo que tarda la frenada en llegar
    a su aceleración (rampa limitada por el tirón: a/jerk, la mitad en recorrido). Con 0,1 s y frenando al 90 %,
    el Matrice (4 m/s², tirón 5 m/s³) se metía en la copa de un árbol persiguiendo a 4 m/s."""
    return max(P["cp_react"], 0.5 * P["cp_acc"] * prof.acc_hor / prof.jerk)


def cp_speed_limit(prof: Profile, reach: float, P=None) -> float:
    """Velocidad máxima con la que Collision Prevention deja volar cuando los sensores ven hasta `reach` m: la que
    cumple v = √(2·a·(reach − radio − margen − reacción·v)), con a = CP_ACC_FACTOR · aceleración. El planificador usa
    este mismo límite; si planificara más rápido, el dron se quedaría atrás de la referencia y replanificaría."""
    P = P or tuning.params(prof.key)
    a = P["cp_acc"] * prof.acc_hor
    room = max(reach - prof.radius - P["cp_margin"], 0.5)
    react = cp_reaction(prof, P)
    return -a * react + math.sqrt((a * react) ** 2 + 2 * a * room)


class PositionController:
    def __init__(self, profile: Profile, P=None):
        self.prof = profile
        self.P = P               # parámetros de seguridad (los del piloto); None: tuning.params
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
        self.clear_ahead = 0.0   # m que el dron VE libres hasta la meta (sprint de carrera): amplían el límite global
        self.dive = 1.0          # factor sobre la bajada máxima (carrera: picado en diagonal hacia la meta)
        self.ram = None          # embestida (mission.RAM): (inclinación máxima en rad, velocidad máxima) o None

    def reset(self):
        self.integ[:] = 0.0
        self.prev_v = None

    def update(self, p, v, p_ref, v_ref, a_ref, dt, rays=None):
        prof = self.prof
        # posición -> velocidad
        v_sp = v_ref + self.kp * (p_ref - p)
        if rays:
            v_sp, a_ref = collision_prevention(v_sp, a_ref.copy(), rays, prof, v, self.clear_ahead, self.P)
            self.cp_active = self._cp_hits > 0
        else:
            self.cp_active = False
        tilt_max, v_max = self.ram if self.ram else (prof.tilt_max, prof.v_max)
        h = math.hypot(v_sp[0], v_sp[1])
        if h > v_max:
            v_sp[:2] *= v_max / h
        v_sp[2] = float(np.clip(v_sp[2], -prof.v_down * self.dive, prof.v_up))
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
        if DRAG_FF:
            a_sp = a_sp + DRAG_FF * prof.drag * float(np.linalg.norm(v_sp)) * v_sp
        # aceleración -> vector de empuje, con límite de inclinación y prioridad vertical
        t = a_sp + np.array([0.0, 0.0, G])
        t[2] = max(t[2], 0.1 * G)
        t_max = prof.twr * G
        t[2] = min(t[2], t_max)
        # (en la embestida, la inclinación que aún sostiene el peso, ~55-60°: casi todo el empuje hacia delante)
        max_h = min(t[2] * math.tan(tilt_max), math.sqrt(max(t_max ** 2 - t[2] ** 2, 0.0)))
        # MPC_THR_XY_MARG (0,3): aunque se priorice la subida, se guarda un 30 % del empuje para el control
        # horizontal y la subida cede lo necesario. Sin esto, un Matrice subiendo a tope junto a un edificio se
        # quedaba sin fuerza lateral para apartarse y chocaba
        marg = min(XY_MARGIN * t_max, t[2] * math.tan(tilt_max))
        th = math.hypot(t[0], t[1])
        if th > max_h and max_h < marg:
            max_h = min(th, marg)
            t[2] = min(t[2], math.sqrt(max(t_max ** 2 - max_h ** 2, 0.0)))
        self.saturated = th > max_h
        if self.saturated:
            t[:2] *= max_h / th
        return t  # vector de empuje deseado (m/s²) en ejes del mundo


def collision_prevention(v_sp, a_ref, rays, prof, v_now=None, clear_ahead: float = 0.0, P=None):
    """Limita la velocidad pedida en la dirección de cada rayo que ve un obstáculo cercano (PX4 CollisionPrevention).

    rays: [{"dir": (x, y, z), "dist": d, "label": ...}] de los telémetros (distancias medidas, con su ruido).
    Los rayos que miran hacia abajo (abajo y frontal-inferior) no cuentan: casi siempre ven el SUELO, y el suelo
    lo gestionan el despegue y el aterrizaje (si no, el dron no podría aterrizar). Los obstáculos bajos delante
    los ve el anillo horizontal. Excepción: un rayo con su propia distancia de seguridad (`d_safe`), que la simulación
    añade en crucero con el rayo inferior para no bajar sobre la copa de un árbol.
    """
    # distancia de seguridad dinámica: margen fijo + lo que recorre durante el tiempo de reacción (0,3 s)
    speed = float(np.linalg.norm(v_now)) if v_now is not None else 0.0
    P = P or tuning.params(prof.key)
    cp_margin, cp_react = P["cp_margin"], cp_reaction(prof, P)
    d_safe = prof.radius + cp_margin + cp_react * speed
    acc = P["cp_acc"] * prof.acc_hor  # frenada prudente: el dron tarda en responder (inercia de actitud y control)
    reach = max((r["dist"] for r in rays), default=prof.sensor_range)  # alcance efectivo (menor con lluvia)
    # poder frenar dentro de lo que se ve; en el sprint de carrera (clear_ahead = inf) no hay límite global: el
    # pasillo hasta la meta se ve libre y la meta se cruza. Cada rayo sigue limitando hacia su obstáculo
    v_cap = math.inf if clear_ahead == math.inf else math.sqrt(2 * acc * max(max(reach, clear_ahead) - d_safe, 0.5))
    n = math.sqrt(v_sp[0] ** 2 + v_sp[1] ** 2 + v_sp[2] ** 2)
    if n > v_cap:
        v_sp = v_sp * (v_cap / n)
    # (las cuentas por rayo, en floats de Python y no con vectores de numpy de 3 elementos: es lo que más se
    # ejecuta de todo el simulador, 200 veces por segundo y ~100 rayos, y así va ~4 veces más rápido)
    vx, vy, vz = float(v_sp[0]), float(v_sp[1]), float(v_sp[2])
    ax, ay, az = float(a_ref[0]), float(a_ref[1]), float(a_ref[2])
    wx, wy, wz = (0.0, 0.0, 0.0) if v_now is None else (float(v_now[0]), float(v_now[1]), float(v_now[2]))
    sprint = clear_ahead == math.inf
    if n > 1e-6:
        dx, dy, dz = vx / n, vy / n, vz / n
    # y la dirección en la que se MUEVE de verdad: con inercia (un Matrice de 6,5 kg), tras pasar la meta de largo
    # seguía 8 m/s hacia un edificio que no estaba en la dirección pedida y el sprint lo ignoraba
    nw = math.sqrt(wx * wx + wy * wy + wz * wz)
    if nw > 1e-6:
        mx, my, mz = wx / nw, wy / nw, wz / nw
    corridor = prof.radius + P.get("corridor", 0.6) + 0.05 * speed   # medio ancho del pasillo del sprint
    base = prof.radius + cp_margin
    active = []
    for r in rays:
        ux, uy, uz = r["dir"]
        own = r.get("d_safe")
        dist = r["dist"]
        if (uz < -0.3 and own is None) or dist >= reach - 1e-6:
            continue
        if sprint and own is None and n > 1e-6:
            # sprint de carrera: cada rayo es una pared perpendicular a él (el modelo de PX4, muy prudente: un
            # árbol a 6 m y 30° del rumbo frenaba al dron de 15 a 7 m/s). Solo cuenta si el obstáculo está DENTRO
            # del pasillo que va a barrer el dron (a menos de radio + 0,6 m + 0,05 s·v de su línea de avance)
            qx, qy, qz = ux * dist, uy * dist, uz * dist
            ahead = qx * dx + qy * dy + qz * dz
            off = ahead <= 0 or math.sqrt((qx - ahead * dx) ** 2 + (qy - ahead * dy) ** 2
                                          + (qz - ahead * dz) ** 2) > corridor
            if off and nw > 1e-6:
                ahead_m = qx * mx + qy * my + qz * mz
                off = ahead_m <= 0 or math.sqrt((qx - ahead_m * mx) ** 2 + (qy - ahead_m * my) ** 2
                                                + (qz - ahead_m * mz) ** 2) > corridor
            if off:
                continue
        # el margen por tiempo de reacción solo cuenta con la velocidad HACIA ese obstáculo: si se aleja de un
        # edificio que acaba de pasar, no hay que "huir" de él (antes eso lo empujaba contra la pared de enfrente)
        closing = wx * ux + wy * uy + wz * uz
        safe = own if own is not None else base + cp_react * max(closing, 0.0)
        room = dist - safe
        a_r = r.get("acc", acc)            # un rayo puede pedir su propia frenada (la geovalla: la prudente)
        active.append((ux, uy, uz, room, a_r, closing))
        along = vx * ux + vy * uy + vz * uz
        v_lim = math.sqrt(2 * a_r * max(room, 0.0))
        if room < 0:                      # demasiado cerca: alejarse un poco
            v_lim = -min(1.5, -room * 3.0)
        if along > v_lim:
            k = along - v_lim
            vx, vy, vz = vx - k * ux, vy - k * uy, vz - k * uz
            a_along = ax * ux + ay * uy + az * uz
            if a_along > 0:
                ax, ay, az = ax - a_along * ux, ay - a_along * uy, az - a_along * uz
    # segunda pasada, solo recortando: ningún "alejarse" de un obstáculo puede meterlo en otro
    for ux, uy, uz, room, a_r, closing in active:
        v_lim = math.sqrt(2 * a_r * max(room, 0.0))
        along = vx * ux + vy * uy + vz * uz
        if along > v_lim:
            k = along - v_lim
            vx, vy, vz = vx - k * ux, vy - k * uy, vz - k * uz
        # si YA va más rápido hacia el obstáculo de lo que permite frenar a tiempo, no basta con pedir menos
        # velocidad (el PID de velocidad frenaría flojo): se añade la deceleración necesaria para parar en el
        # espacio que queda, hasta la aceleración máxima del dron (un Matrice a 6 m/s chocaba contra el borde)
        if closing > v_lim + 0.3:
            need = (closing ** 2 - v_lim ** 2) / (2 * max(room, 0.3))
            k = min(need, prof.acc_hor) + max(ax * ux + ay * uy + az * uz, 0.0)
            ax, ay, az = ax - k * ux, ay - k * uy, az - k * uz
    return np.array([vx, vy, vz]), np.array([ax, ay, az])
