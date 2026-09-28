"""Física del multicóptero: masa puntual con actitud, empuje, arrastre, viento y suelo.

Es el modelo que usan los simuladores de planificación (p. ej. los del FAST Lab o Flightmare en su modo simple):
el bucle interno de actitud del autopiloto es mucho más rápido que el de posición, así que se modela como una
respuesta de primer orden y la física de verdad está en la traslación.

    a = T · z_cuerpo + (0, 0, -g) - k · |v_aire| · v_aire          v_aire = v - viento

  * T: aceleración colectiva del empuje (m/s²), sigue a la orden con la constante de tiempo de los motores y está
    limitada por la relación empuje/peso.
  * z_cuerpo: eje "arriba" del dron. Gira hacia el eje pedido con la constante de tiempo de actitud y una velocidad
    angular máxima. Inclinarse es la ÚNICA forma de acelerar en horizontal: por eso acelerar, frenar y resistir
    el viento tienen límites realistas.
  * k: arrastre cuadrático deducido de la velocidad máxima del fabricante (ver params.py).
  * Superficie (terreno con relieve o azotea): tocarla despacio es posarse; bajando a más de 2 m/s o yendo a más de
    3 m/s en horizontal es estrellarse (p. ej. contra la pared de un acantilado). Posarse sobre una azotea no es
    chocar con el edificio.
"""
import math
from typing import Optional

import numpy as np

from .params import G, Profile
from .world import LENGTH, WIDTH, Box, World

HARD_VZ, HARD_VXY = 2.0, 3.0
MAX_TILT_RATE = math.radians(300)  # rad/s (Matrice 350 RTK: 300°/s en cabeceo)


def rotate_towards(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    """Gira el vector unitario a hacia b como mucho max_angle radianes (sobre el círculo máximo)."""
    c = float(np.clip(a @ b, -1.0, 1.0))
    ang = math.acos(c)
    if ang <= max_angle or ang < 1e-9:
        return b.copy()
    axis = np.cross(a, b)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return a.copy()
    axis /= n
    t = max_angle
    return a * math.cos(t) + np.cross(axis, a) * math.sin(t) + axis * (axis @ a) * (1 - math.cos(t))


def body_axes(z_body: np.ndarray, yaw: float):
    """(x_cuerpo, y_cuerpo, z_cuerpo) a partir del eje z y el rumbo (como bodyzToAttitude de PX4)."""
    y_c = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    x_b = np.cross(y_c, z_body)
    n = np.linalg.norm(x_b)
    x_b = x_b / n if n > 1e-6 else np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_b = np.cross(z_body, x_b)
    return x_b, y_b, z_body


class Multirotor:
    def __init__(self, profile: Profile, pos, yaw: float = 0.0):
        self.prof = profile
        self.gear = 0.4 * profile.radius       # altura del centro sobre la superficie cuando está posado
        self.p = np.array([pos[0], pos[1], pos[2] + self.gear], dtype=float)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.z_body = np.array([0.0, 0.0, 1.0])
        self.yaw = yaw
        self.thrust = 0.0                      # m/s²
        self.on_ground = True
        self.crashed: Optional[str] = None
        # órdenes del controlador
        self.cmd_z = np.array([0.0, 0.0, 1.0])
        self.cmd_thrust = 0.0
        self.cmd_yaw = yaw
        self.energy_wh = 0.0
        self.drag_mult = 1.0                   # >1 con lluvia

    @property
    def tilt(self) -> float:
        return math.acos(float(np.clip(self.z_body[2], -1, 1)))

    def power(self) -> float:
        """Potencia eléctrica (W): la de vuelo estacionario escalada por (T/g)^1,5 (teoría del momento) + 10 %."""
        return self.prof.hover_power_w * (0.1 + 0.9 * (max(self.thrust, 0.0) / G) ** 1.5)

    def step(self, dt: float, wind: np.ndarray, world: World):
        if self.crashed:
            return
        prof = self.prof
        # motores y actitud (bucle interno rápido, primer orden)
        t_cmd = float(np.clip(self.cmd_thrust, 0.0, prof.twr * G))
        self.thrust += (t_cmd - self.thrust) * min(1.0, dt / prof.tau_motor)
        z_cmd = self.cmd_z / np.linalg.norm(self.cmd_z)
        ang = math.acos(float(np.clip(self.z_body @ z_cmd, -1, 1)))
        self.z_body = rotate_towards(self.z_body, z_cmd, min(ang * dt / prof.tau_att, MAX_TILT_RATE * dt))
        self.z_body /= np.linalg.norm(self.z_body)
        dyaw = (self.cmd_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += float(np.clip(dyaw, -math.radians(prof.yaw_rate_deg) * dt, math.radians(prof.yaw_rate_deg) * dt))
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        # traslación
        v_air = self.v - wind
        a = self.thrust * self.z_body + np.array([0.0, 0.0, -G]) - prof.drag * self.drag_mult * np.linalg.norm(v_air) * v_air
        if self.on_ground:
            if a[2] <= 0:            # posado: el suelo sostiene al dron
                self.v[:] = 0.0
                self.a[:] = 0.0
                self.energy_wh += self.power() * dt / 3600
                return
            self.on_ground = False
        self.a = a
        self.v += a * dt
        self.p += self.v * dt
        self.energy_wh += self.power() * dt / 3600
        # superficie: terreno o azotea
        surf = world.surface(self.p[0], self.p[1])
        if self.p[2] <= surf + self.gear:
            on_terrain = surf <= world.terrain.height(self.p[0], self.p[1]) + 0.01
            wall = surf + self.gear - self.p[2] > 0.25  # entra de lado en una pared (acantilado, edificio)
            if wall or self.v[2] < -HARD_VZ or math.hypot(self.v[0], self.v[1]) > HARD_VXY:
                self.crashed = ("terreno" if on_terrain else "edificio") if wall else ("terreno" if on_terrain else "azotea")
                return
            self.p[2] = surf + self.gear
            self.a = self.a - self.v / dt  # el golpe de parar en seco lo siente la IMU (si no, el filtro diverge)
            self.v[:] = 0.0
            self.on_ground = True
        # obstáculos y bordes
        x, y = self.p[0], self.p[1]
        r = prof.radius
        if x < r or y < r or x > LENGTH - r or y > WIDTH - r:
            self.crashed = "borde"
        elif self._hits_obstacle(world, r):
            self.crashed = "obstáculo"

    def _hits_obstacle(self, world: World, r: float) -> bool:
        p = tuple(self.p)
        for ob in world.near(p[0], p[1], r):     # solo los cercanos (antes, todos en cada paso de 5 ms)
            x0, x1, y0, y1 = ob.footprint()
            if p[0] < x0 - r or p[0] > x1 + r or p[1] < y0 - r or p[1] > y1 + r:
                continue
            if isinstance(ob, Box) and x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and p[2] >= ob.top - 1e-3:
                continue  # encima de la azotea: es superficie, no choque
            if ob.distance(p) < r:
                return True
        return False

    def attitude(self):
        return body_axes(self.z_body, self.yaw)


SQ2 = math.sqrt(0.5)
# MC_YAW_WEIGHT de PX4: el rumbo se corrige con 0,4 veces MC_YAW_P, más despacio que la inclinación y que su propio
# bucle de velocidad de giro (con la ganancia entera los dos bucles se peleaban y el dron oscilaba de rumbo)
YAW_WEIGHT = 0.4
YAW_ACC = 60.0     # °/s², MPC_YAWRAUTO_ACC por defecto
TILT_FF = float(__import__("os").environ.get("DRON_TILT_FF", 1.0))   # prealimentación del giro (ver step)
TILT_FF_MAX = float(__import__("os").environ.get("DRON_TILT_FF_MAX", 1.0))   # rad/s como mucho


class Multirotor6DOF(Multirotor):
    """Cuadricóptero de 6 grados de libertad con CUATRO empujes independientes, como un dron real (y como el modelo
    del X500 de PX4 en Gazebo). El autopiloto interno es el de PX4:

      posición → velocidad → aceleración (control.py) → vector de empuje pedido (cmd_z, cmd_thrust) y rumbo (cmd_yaw)
      → ACTITUD (mc_att_control: error de inclinación y de rumbo → velocidades angulares, P = MC_ROLL_P 6,5 y
        MC_YAW_P 2,8, limitadas a MC_ROLLRATE_MAX 220°/s y MC_YAWRATE_MAX 200°/s)
      → VELOCIDAD ANGULAR (mc_rate_control: PID por eje → par normalizado; MC_ROLLRATE_P/I/D 0,15/0,2/0,003)
      → MEZCLADOR (cuádruple en X: cada motor = empuje ± 0,707·balanceo ± 0,707·cabeceo ± guiñada, con la prioridad
        de PX4: si un motor satura, primero se sacrifica la guiñada, luego el empuje y por último el balanceo y el
        cabeceo, que son los que mantienen el dron en el aire)
      → 4 MOTORES (primer orden: más rápidos acelerando que frenando, timeConstantUp/Down) → empujes f_i
      → SÓLIDO RÍGIDO: par = Σ brazo × empuje (balanceo y cabeceo) + Σ ±k_m·f_i (guiñada: el par de reacción de
        cada hélice; dos giran en un sentido y dos en el otro), J·ω̇ = τ − ω × Jω, y la traslación con el empuje total
        a lo largo del eje del cuerpo, la gravedad y el arrastre.

    Es lo que hace un piloto con las dos palancas: la izquierda da empuje (arriba/abajo) y guiñada (girar), la
    derecha cabeceo (adelante/atrás) y balanceo (lados). Para acelerar hacia delante hay que cabecear (inclinarse) Y a
    la vez dar más empuje para no perder altura, y para frenar, cabecear al revés antes de llegar: dominar la
    inercia. `sticks` guarda esas cuatro órdenes (normalizadas) y `u` la salida de cada motor.

    Todo el bucle interno corre a 1 kHz (5 pasos por cada paso de 5 ms de la simulación), como el de PX4.
    """
    SUB = 5

    def __init__(self, profile: Profile, pos, yaw: float = 0.0):
        super().__init__(profile, pos, yaw)
        c = profile.arm * SQ2
        # (x, y, sentido): 0 delante-derecha y 1 detrás-izquierda giran antihorario (su par de reacción es horario:
        # −z); 2 delante-izquierda y 3 detrás-derecha, horario (+z). Como el X500 (rotor_0..3)
        self.mot = ((c, -c, -1.0), (-c, c, -1.0), (c, c, 1.0), (-c, -c, 1.0))
        self.mix = tuple((math.copysign(SQ2, y), -math.copysign(SQ2, x), s) for x, y, s in self.mot)
        self.f = [0.0, 0.0, 0.0, 0.0]            # empuje de cada motor (N)
        self.u = [0.0, 0.0, 0.0, 0.0]            # salida normalizada de cada motor (0-1)
        self._level(yaw)                         # orientación R (por filas, floats)
        self.R = np.array([self.r[0:3], self.r[3:6], self.r[6:9]])
        self.w = [0.0, 0.0, 0.0]                 # velocidad angular en ejes del cuerpo (rad/s)
        self.integ = [0.0, 0.0, 0.0]
        self.w_prev = [0.0, 0.0, 0.0]
        self.yaw_sp, self.yaw_rate_sp = yaw, 0.0
        self.sticks = (0.0, 0.0, 0.0, 0.0)       # empuje, balanceo, cabeceo, guiñada (las dos palancas)
        self.tau_mult = 1.0

    @property
    def f_max(self) -> float:
        return self.prof.twr * self.prof.mass * G / 4.0

    def attitude(self):
        return self.R[:, 0].copy(), self.R[:, 1].copy(), self.R[:, 2].copy()

    def _inner(self, dt: float, zd):
        """Un paso del autopiloto interno (actitud → velocidad angular → mezclador) y de los motores. Todo en floats de
        Python (la orientación `self.r` son las 9 componentes de R por filas): es lo que más se ejecuta, 1000 veces
        por segundo simulado, y con matrices de numpy de 3×3 iba ~3 veces más lento."""
        prof, g = self.prof, self.prof.inner
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = self.r
        zdx, zdy, zdz = zd
        # --- actitud: inclinación hacia el eje pedido (eje del cuerpo z = columna 2 de R) y rumbo hacia el pedido
        cx = r12 * zdz - r22 * zdy
        cy = r22 * zdx - r02 * zdz
        cz = r02 * zdy - r12 * zdx
        sn = math.sqrt(cx * cx + cy * cy + cz * cz)
        ang = math.atan2(sn, r02 * zdx + r12 * zdy + r22 * zdz)
        k = ang / sn if sn > 1e-9 else 0.0
        ex = (r00 * cx + r10 * cy + r20 * cz) * k     # error en ejes del cuerpo (Rᵀ·eje·ángulo)
        ey = (r01 * cx + r11 * cy + r21 * cz) * k
        yaw = math.atan2(r10, r00)
        eyaw = (self.yaw_sp - yaw + math.pi) % (2 * math.pi) - math.pi
        rmax, ymax = self._rmax, self._ymax
        ap = g["att_p"]
        # (guiñada: más la velocidad a la que gira el rumbo pedido, como el feedforward de PX4; sin ella el dron
        # iba siempre por detrás, el integrador se cargaba y se pasaba de rumbo: el Matrice llegaba a 125° de 90)
        ffx, ffy = getattr(self, "_tilt_ff", (0.0, 0.0))
        sp = (max(-rmax, min(rmax, ap * ex + ffx)), max(-rmax, min(rmax, ap * ey + ffy)),
              max(-ymax, min(ymax, YAW_WEIGHT * g["yaw_p"] * eyaw + self.yaw_rate_sp)))
        # --- velocidad angular: PID (derivada sobre la medida) → par normalizado
        w, wp, it = self.w, self.w_prev, self.integ
        out = [0.0, 0.0, 0.0]
        for i in range(3):
            e = sp[i] - w[i]
            if i < 2:
                kp, ki, kd = g["rate_p"], g["rate_i"], g["rate_d"]
            else:
                kp, ki, kd = g["yawrate_p"], g["yawrate_i"], 0.0
            v = it[i] + ki * e * dt
            it[i] = -0.3 if v < -0.3 else (0.3 if v > 0.3 else v)          # MC_RR_INT_LIM 0,3
            out[i] = kp * e + it[i] - kd * (w[i] - wp[i]) / dt
        self.w_prev = [w[0], w[1], w[2]]
        roll, pitch, yawc = out
        fm = self._fmax
        thr = self.cmd_thrust * prof.mass / (4.0 * fm)
        thr = 0.0 if thr < 0.0 else (1.0 if thr > 1.0 else thr)
        # --- mezclador con la prioridad de PX4: balanceo y cabeceo > empuje > guiñada
        rp = [mr * roll + mp * pitch for mr, mp, _ in self.mix]
        hi, lo = max(rp), min(rp)
        if hi - lo > 1.0:                        # ni con el empuje ideal cabe: se reduce el balanceo/cabeceo
            s = 1.0 / (hi - lo)
            rp = [v * s for v in rp]
            hi, lo = hi * s, lo * s
        thr = min(max(thr, -lo), 1.0 - hi)       # se mueve el empuje lo justo para que quepan
        room = min(1.0 - thr - hi, thr + lo)
        yawc = max(-room, min(room, yawc))       # la guiñada, con lo que sobre
        u = self.u
        for i in range(4):
            v = thr + rp[i] + self.mix[i][2] * yawc
            u[i] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        self.sticks = (thr, roll, pitch, yawc)
        # --- motores: primer orden, más rápidos acelerando que frenando
        f = self.f
        up, dn = self._aup * dt, self._adn * dt
        for i in range(4):
            target = u[i] * fm
            a = up if target > f[i] else dn
            f[i] += (target - f[i]) * (a if a < 1.0 else 1.0)

    def _body(self, dt: float):
        """Sólido rígido: pares de los cuatro motores → velocidad angular → orientación (Rodrigues, en floats)."""
        prof = self.prof
        Ix, Iy, Iz = prof.inertia
        f0, f1, f2, f3 = self.f
        (x0, y0, s0), (x1, y1, s1), (x2, y2, s2), (x3, y3, s3) = self.mot
        tx = y0 * f0 + y1 * f1 + y2 * f2 + y3 * f3
        ty = -(x0 * f0 + x1 * f1 + x2 * f2 + x3 * f3)
        tz = prof.k_m * (s0 * f0 + s1 * f1 + s2 * f2 + s3 * f3)
        wx, wy, wz = self.w
        wx += (tx - (Iz - Iy) * wy * wz) / Ix * dt
        wy += (ty - (Ix - Iz) * wz * wx) / Iy * dt
        wz += (tz - (Iy - Ix) * wx * wy) / Iz * dt
        self.w = [wx, wy, wz]
        n = math.sqrt(wx * wx + wy * wy + wz * wz)
        th = n * dt
        if th > 1e-12:                           # R ← R·exp([ω]× dt)
            kx, ky, kz = wx / n, wy / n, wz / n
            s, c1 = math.sin(th), 1.0 - math.cos(th)
            # E = I + s·K + c1·K² (K = [k]×)
            e00 = 1.0 - c1 * (ky * ky + kz * kz)
            e11 = 1.0 - c1 * (kx * kx + kz * kz)
            e22 = 1.0 - c1 * (kx * kx + ky * ky)
            e01 = -s * kz + c1 * kx * ky
            e10 = s * kz + c1 * kx * ky
            e02 = s * ky + c1 * kx * kz
            e20 = -s * ky + c1 * kx * kz
            e12 = -s * kx + c1 * ky * kz
            e21 = s * kx + c1 * ky * kz
            r00, r01, r02, r10, r11, r12, r20, r21, r22 = self.r
            self.r = [r00 * e00 + r01 * e10 + r02 * e20, r00 * e01 + r01 * e11 + r02 * e21,
                      r00 * e02 + r01 * e12 + r02 * e22,
                      r10 * e00 + r11 * e10 + r12 * e20, r10 * e01 + r11 * e11 + r12 * e21,
                      r10 * e02 + r11 * e12 + r12 * e22,
                      r20 * e00 + r21 * e10 + r22 * e20, r20 * e01 + r21 * e11 + r22 * e21,
                      r20 * e02 + r21 * e12 + r22 * e22]

    def _level(self, yaw):
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.r = [cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0]

    def step(self, dt: float, wind: np.ndarray, world: World):
        if self.crashed:
            return
        prof = self.prof
        # rumbo pedido con velocidad Y ACELERACIÓN limitadas (MPC_YAWRAUTO_MAX y MPC_YAWRAUTO_ACC, 60°/s²): acelera,
        # gira y frena antes de llegar. Sin el límite de aceleración, el Matrice (poco par de guiñada) se pasaba ~20°
        # al parar un giro de 100°/s: frenarlo le lleva esos 20°
        dyaw = (self.cmd_yaw - self.yaw_sp + math.pi) % (2 * math.pi) - math.pi
        acc = math.radians(YAW_ACC)
        want = math.copysign(min(math.radians(prof.yaw_rate_deg), math.sqrt(2 * acc * abs(dyaw))), dyaw)
        self.yaw_rate_sp += max(-acc * dt, min(acc * dt, want - self.yaw_rate_sp))
        self.yaw_sp = (self.yaw_sp + self.yaw_rate_sp * dt + math.pi) % (2 * math.pi) - math.pi
        # constantes de este paso
        g = prof.inner
        self._rmax, self._ymax = math.radians(g["rate_max"]), math.radians(g["yawrate_max"])
        self._fmax = self.f_max
        self._aup = 1.0 / (prof.tau_up * self.tau_mult)
        self._adn = 1.0 / (prof.tau_down * self.tau_mult)
        cz = self.cmd_z
        nz = math.sqrt(cz[0] * cz[0] + cz[1] * cz[1] + cz[2] * cz[2])
        zd = (cz[0] / nz, cz[1] / nz, cz[2] / nz)
        # PREALIMENTACIÓN DEL GIRO (planitud diferencial, Mellinger y Kumar 2011): el eje de empuje pedido gira a
        # ω = z_prev × z_nuevo / dt; se da esa velocidad angular de antemano al lazo de velocidad angular (en ejes del
        # cuerpo, filtrada ~30 ms), en vez de esperar a que haya error de actitud. DRON_TILT_FF=0: sin ella
        zp = getattr(self, "_zd_prev", None)
        fx = fy = 0.0
        if TILT_FF and zp is not None and dt > 0:
            wx_w = (zp[1] * zd[2] - zp[2] * zd[1]) / dt
            wy_w = (zp[2] * zd[0] - zp[0] * zd[2]) / dt
            wz_w = (zp[0] * zd[1] - zp[1] * zd[0]) / dt
            r = self.r      # R por filas: ejes del cuerpo = Rᵀ · mundo
            fx = r[0] * wx_w + r[3] * wy_w + r[6] * wz_w
            fy = r[1] * wx_w + r[4] * wy_w + r[7] * wz_w
        a = min(1.0, dt / 0.03)
        ox, oy = getattr(self, "_tilt_ff", (0.0, 0.0))
        # (acotada: un salto brusco del eje pedido, p. ej. un escalón, no es una trayectoria que seguir y su
        # "derivada" daría un latigazo que hace pasarse de inclinación)
        lim = TILT_FF_MAX
        fx = max(-lim, min(lim, TILT_FF * fx))
        fy = max(-lim, min(lim, TILT_FF * fy))
        self._tilt_ff = (ox + (fx - ox) * a, oy + (fy - oy) * a)
        self._zd_prev = zd
        h = dt / self.SUB
        k_drag = prof.drag * self.drag_mult
        wx_, wy_, wz_ = float(wind[0]), float(wind[1]), float(wind[2])
        px, py, pz = float(self.p[0]), float(self.p[1]), float(self.p[2])
        vx, vy, vz = float(self.v[0]), float(self.v[1]), float(self.v[2])
        ax = ay = az = 0.0
        m = prof.mass
        for _ in range(self.SUB):
            self._inner(h, zd)
            total = self.f[0] + self.f[1] + self.f[2] + self.f[3]
            if self.on_ground:
                if total / m * self.r[8] <= G:     # posado: el suelo lo sostiene, nivelado
                    self.w = [0.0, 0.0, 0.0]
                    self.integ = [0.0, 0.0, 0.0]
                    self._level(self.yaw_sp)
                    vx = vy = vz = 0.0
                    ax = ay = az = 0.0
                    continue
                self.on_ground = False
            self._body(h)
            r = self.r
            ux, uy, uz = vx - wx_, vy - wy_, vz - wz_
            dk = k_drag * math.sqrt(ux * ux + uy * uy + uz * uz)
            T = total / m
            ax = T * r[2] - dk * ux
            ay = T * r[5] - dk * uy
            az = T * r[8] - G - dk * uz
            vx += ax * h
            vy += ay * h
            vz += az * h
            px += vx * h
            py += vy * h
            pz += vz * h
        self.p[:] = (px, py, pz)
        self.v[:] = (vx, vy, vz)
        self.a = np.array([ax, ay, az])
        self._orthonormalize()
        r = self.r
        self.R = np.array([r[0:3], r[3:6], r[6:9]])
        self.z_body = np.array([r[2], r[5], r[8]])
        self.yaw = math.atan2(r[3], r[0])
        self.thrust = sum(self.f) / m
        self.energy_wh += self.power() * dt / 3600
        if self.on_ground:
            return
        self._contact(world, dt)

    def _orthonormalize(self):
        """Gram-Schmidt sobre las columnas x e y de R (los redondeos la desvían muy poco a cada paso)."""
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = self.r
        n = math.sqrt(r00 * r00 + r10 * r10 + r20 * r20)
        r00, r10, r20 = r00 / n, r10 / n, r20 / n
        d = r01 * r00 + r11 * r10 + r21 * r20
        r01, r11, r21 = r01 - d * r00, r11 - d * r10, r21 - d * r20
        n = math.sqrt(r01 * r01 + r11 * r11 + r21 * r21)
        r01, r11, r21 = r01 / n, r11 / n, r21 / n
        r02, r12, r22 = r10 * r21 - r20 * r11, r20 * r01 - r00 * r21, r00 * r11 - r10 * r01
        self.r = [r00, r01, r02, r10, r11, r12, r20, r21, r22]

    def _contact(self, world: World, dt: float):
        prof = self.prof
        surf = world.surface(self.p[0], self.p[1])
        if self.p[2] <= surf + self.gear:
            on_terrain = surf <= world.terrain.height(self.p[0], self.p[1]) + 0.01
            wall = surf + self.gear - self.p[2] > 0.25
            if wall or self.v[2] < -HARD_VZ or math.hypot(self.v[0], self.v[1]) > HARD_VXY:
                self.crashed = ("terreno" if on_terrain else "edificio") if wall else ("terreno" if on_terrain else "azotea")
                return
            self.p[2] = surf + self.gear
            self.a = self.a - self.v / dt
            self.v[:] = 0.0
            self.on_ground = True
        x, y = self.p[0], self.p[1]
        r = prof.radius
        if x < r or y < r or x > LENGTH - r or y > WIDTH - r:
            self.crashed = "borde"
        elif self._hits_obstacle(world, r):
            self.crashed = "obstáculo"
