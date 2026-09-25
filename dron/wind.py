"""Viento: medio con cizalladura, ráfagas Dryden y EFECTO DE LOS OBSTÁCULOS Y DEL TERRENO.

1. Viento medio a 6,1 m sobre el suelo, con el perfil logarítmico de MIL-F-8785C:
   W(h) = W20 · ln(h / z0) / ln(20 ft / z0), con h = altura SOBRE EL TERRENO y z0 = 0,15 ft.

2. Ráfagas con el modelo de turbulencia Dryden a baja altitud (MIL-F-8785C / MIL-HDBK-1797), con fórmulas en pies:
       L_w = h,  L_u = L_v = h / (0,177 + 0,000823 h)^1,2,  σ_w = 0,1 · W20,  σ_u = σ_v = σ_w / (0,177 + 0,000823 h)^0,4
   Cada componente es un proceso de Gauss-Markov de primer orden con tiempo de correlación L/V.
   Intensidad: 0 ninguna, 1 ligeras, 2 moderadas, 3 fuertes (multiplica σ).

3. Interacción con obstáculos y terreno (modelo de estelas simplificado, en el espíritu de los modelos de ingeniería
   de viento urbano; no es CFD):
     * ESTELA de abrigo detrás de cada edificio o árbol: el viento cae hasta un 80 % (edificios) o un 40 % (árboles,
       que son porosos) justo detrás, se recupera con la distancia (≈ 5 alturas) y la estela se ensancha. Dentro de
       la estela la turbulencia AUMENTA (el aire se arremolina).
     * ACELERACIÓN sobre las azoteas (+25 %) y sobre las cimas y lomas (+3 % por metro de altura del terreno).
     * ABRIGO del relieve: si entre el punto y el viento hay terreno más alto (una montaña, una pared de acantilado),
       el viento cae un 60 % y se vuelve más turbulento.
"""
import math
import random
from typing import Tuple

import numpy as np

from .world import Box, World

FT = 0.3048
GUST_LEVELS = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
TURB_MAX = 2.0   # multiplicador máximo de la turbulencia en estelas (ESTIMADO)


class Wind:
    def __init__(self, speed: float = 0.0, direction_deg: float = 0.0, gusts: int = 0, seed: int = 0,
                 world: World = None):
        """speed: viento medio a 6,1 m sobre el suelo (m/s); direction_deg: hacia dónde SOPLA (0 = +x, 90 = +y)."""
        self.speed = speed
        self.direction = math.radians(direction_deg)
        self.u = np.array([math.cos(self.direction), math.sin(self.direction)])
        self.gusts = gusts
        self.rng = random.Random(seed)
        self.turb = [0.0, 0.0, 0.0]
        self.world = world
        self._prep()

    def _prep(self):
        """Datos de las estelas por obstáculo, vectorizados (dependen de la dirección del viento)."""
        obs = self.world.obstacles if self.world else []
        perp = np.array([-self.u[1], self.u[0]])
        self.o_c = np.array([[(o.footprint()[0] + o.footprint()[1]) / 2, (o.footprint()[2] + o.footprint()[3]) / 2]
                             for o in obs]).reshape(-1, 2)
        half, depth, top, base, strength = [], [], [], [], []
        for o in obs:
            x0, x1, y0, y1 = o.footprint()
            ex, ey = (x1 - x0) / 2, (y1 - y0) / 2
            half.append(abs(perp[0]) * ex + abs(perp[1]) * ey)   # semiancho visto por el viento
            depth.append(abs(self.u[0]) * ex + abs(self.u[1]) * ey)
            top.append(o.top)
            base.append(o.base)
            strength.append(0.8 if isinstance(o, Box) else 0.4)
        self.o_half, self.o_depth = np.array(half), np.array(depth)
        self.o_top, self.o_base, self.o_str = np.array(top), np.array(base), np.array(strength)
        self.boxes = [o for o in obs if isinstance(o, Box)]

    def factors(self, p) -> Tuple[float, float]:
        """(factor sobre el viento medio, multiplicador de turbulencia) en el punto p."""
        if self.world is None:
            return 1.0, 1.0
        f, turb = 1.0, 1.0
        if len(self.o_c):
            r = np.asarray(p[:2]) - self.o_c
            a = r @ self.u - self.o_depth                     # distancia aguas abajo desde la cara trasera
            c = np.abs(r @ np.array([-self.u[1], self.u[0]]))
            h = np.maximum(self.o_top - self.o_base, 0.5)
            width = self.o_half + 0.2 * np.maximum(a, 0) + 0.3
            inside = (a > -self.o_depth) & (a < 6 * h) & (c < width) & (p[2] < self.o_top + 0.3 * h)
            if inside.any():
                decay = np.exp(-np.maximum(a[inside], 0) / (1.5 * h[inside]))
                lateral = 1 - (c[inside] / width[inside]) ** 2
                s = self.o_str[inside] * decay * lateral
                shelter = 1 - float(np.prod(1 - s))
                f *= 1 - shelter
                turb += 2.5 * shelter
        for b in self.boxes:  # aceleración sobre azoteas
            if b.x0 <= p[0] <= b.x1 and b.y0 <= p[1] <= b.y1 and b.top <= p[2] <= b.top + 0.5 * b.h + 2:
                f *= 1.25
                break
        ter = self.world.terrain
        hground = ter.height(p[0], p[1])
        f *= 1 + 0.03 * hground                               # más viento en alto
        for k in (3, 6, 10, 15, 20, 26):                      # abrigo del relieve aguas arriba
            if ter.height(p[0] - self.u[0] * k, p[1] - self.u[1] * k) > p[2] + 0.5:
                f *= 0.4
                turb += 1.5
                break
        # la turbulencia de una estela es, como mucho, del orden del doble (ESTIMADO). Sin tope, varias estelas
        # sumadas daban ×5: ráfagas de 29 m/s con 6 m/s de viento medio sobre una azotea
        return f, min(turb, TURB_MAX)

    def mean(self, p) -> Tuple[float, float, float]:
        """Viento medio (sin ráfagas) en el punto p, con cizalladura y el efecto de obstáculos y relieve."""
        if self.speed <= 0:
            return (0.0, 0.0, 0.0)
        ground = self.world.terrain.height(p[0], p[1]) if self.world else 0.0
        h_ft = max((p[2] - ground) / FT, 1.0)
        z0 = 0.15
        w = self.speed * max(0.0, math.log(h_ft / z0) / math.log(20.0 / z0))
        f, _ = self.factors(p)
        return (w * f * self.u[0], w * f * self.u[1], 0.0)

    def step(self, p, airspeed: float, dt: float) -> Tuple[float, float, float]:
        """Avanza la turbulencia dt segundos y devuelve el viento total en el punto p (m/s)."""
        mx, my, mz = self.mean(p)
        level = GUST_LEVELS.get(self.gusts, 0.0)
        if level == 0 or self.speed <= 0:
            self.turb = [0.0, 0.0, 0.0]
            return (mx, my, mz)
        _, turb_mult = self.factors(p)
        ground = self.world.terrain.height(p[0], p[1]) if self.world else 0.0
        h = max((p[2] - ground) / FT, 3.0)
        w20 = max(self.speed, 3.0) / FT
        sig_w = 0.1 * w20 * level * turb_mult
        base = 0.177 + 0.000823 * h
        sig_uv = sig_w / base ** 0.4
        l_w, l_uv = h, h / base ** 1.2
        v = max(airspeed, 1.0) / FT
        for i, (sig, length) in enumerate(((sig_uv, l_uv), (sig_uv, l_uv), (sig_w, l_w))):
            a = min(1.0, v * dt / length)
            self.turb[i] = self.turb[i] * (1 - a) + sig * math.sqrt(2 * a) * self.rng.gauss(0, 1)
        u, vv, w = (x * FT for x in self.turb)
        return (mx + u, my + vv, mz + w)

    def field(self, step: float = 3.0, heights_agl=(2.0, 6.0, 12.0)):
        """Rejilla del factor de viento medio a varias alturas sobre el terreno (para dibujar las partículas)."""
        from .world import LENGTH, WIDTH
        xs, ys = np.arange(step / 2, LENGTH, step), np.arange(step / 2, WIDTH, step)
        out = []
        for hz in heights_agl:
            layer = [[round(self.factors((x, y, self.world.terrain.height(x, y) + hz))[0], 2) for y in ys] for x in xs]
            out.append(layer)
        return {"step": step, "heights": list(heights_agl), "f": out}
