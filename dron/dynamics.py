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
        for ob in world.obstacles:
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
