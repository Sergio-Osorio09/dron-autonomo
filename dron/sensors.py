"""Sensores imperfectos. Nivel de ruido: "ideal" (sin ruido), "realista" o "alto" (el doble).

    sensor              frecuencia   modelo (nivel realista)
    IMU (acelerómetro)  200 Hz       aceleración + sesgo (paseo aleatorio) + ruido 0,1 m/s²
    GPS posición        10 Hz        ruido σ del perfil (Mini ±0,75 m, PX4 0,5 m, RTK 0,1 m) + deriva lenta
                                     (Gauss-Markov, τ = 30 s); en vertical, 1,5 veces más
    GPS velocidad       10 Hz        ruido 0,1 m/s
    barómetro           20 Hz        ruido 0,3 m + deriva lenta (τ = 60 s)
    brújula (rumbo)     20 Hz        ruido 2°
    telémetros          20 Hz        40 rayos (36 en anillo cada 10°, arriba, abajo, 2 al frente); ruido 2 % + 2 cm;
                                     con 30° entre rayos se colaban troncos finos entre dos lecturas;
                                     un 2 % de lecturas perdidas (devuelven el alcance máximo)
    cámara inferior     20 Hz        posición relativa de la plataforma (aterrizaje de precisión, como el
                                     IR-LOCK de PX4) si está a menos de 5 m en horizontal y 10 m por encima; σ 5 cm

Lluvia (`RAIN`): las gotas dispersan la luz de los telémetros (láser/infrarrojos) y de la cámara, y degradan algo el
GNSS. Moderada: alcance ×0,7, ruido ×1,5, 5 % de lecturas perdidas, GPS ×1,2. Fuerte: alcance ×0,5, ruido ×2,
12 % perdidas, GPS ×1,4, y la cámara solo ve la plataforma a menos de 3 m.
"""
import math
import random
from typing import Dict, List, Optional

import numpy as np

NOISE_LEVELS = {"ideal": 0.0, "realista": 1.0, "alto": 2.0}
RAIN = {"no": dict(range=1.0, noise=1.0, drop=0.0, gps=1.0, cam=5.0, drag=1.0),
        "moderada": dict(range=0.7, noise=1.5, drop=0.05, gps=1.2, cam=4.0, drag=1.08),
        "fuerte": dict(range=0.5, noise=2.0, drop=0.12, gps=1.4, cam=3.0, drag=1.15)}
RING = [a if a <= 180 else a - 360 for a in range(0, 360, 10)]  # 36 rayos cada 10° (PX4 usa 72 sectores de 5°)


def ray_dirs(yaw: float):
    out = []
    for a in RING:
        th = yaw + math.radians(a)
        out.append((str(a), (math.cos(th), math.sin(th), 0.0)))
    out.append(("up", (0.0, 0.0, 1.0)))
    out.append(("down", (0.0, 0.0, -1.0)))
    for name, el in (("front-up", 35), ("front-down", -35)):
        e = math.radians(el)
        out.append((name, (math.cos(yaw) * math.cos(e), math.sin(yaw) * math.cos(e), math.sin(e))))
    return out


class GaussMarkov:
    def __init__(self, sigma: float, tau: float, rng: random.Random, dim: int = 3):
        self.sigma, self.tau, self.rng = sigma, tau, rng
        self.x = np.array([rng.gauss(0, sigma) for _ in range(dim)]) if sigma > 0 else np.zeros(dim)

    def step(self, dt: float) -> np.ndarray:
        if self.sigma > 0:
            a = dt / self.tau
            self.x = self.x * (1 - a) + self.sigma * math.sqrt(2 * a) * np.array([self.rng.gauss(0, 1) for _ in self.x])
        return self.x


class Sensors:
    def __init__(self, profile, level: str = "realista", seed: int = 0, rain: str = "no"):
        self.prof = profile
        self.m = NOISE_LEVELS[level]
        self.rain = RAIN[rain]
        self.range = profile.sensor_range * self.rain["range"]
        self.rng = random.Random(seed)
        m = self.m
        self.acc_bias = GaussMarkov(0.05 * m, 300.0, self.rng)
        self.gps_drift = GaussMarkov(0.8 * profile.gps_sigma * m, 30.0, self.rng)
        self.baro_drift = GaussMarkov(0.3 * m, 60.0, self.rng, dim=1)
        self.t = 0.0
        self.next = {"gps": 0.0, "baro": 0.0, "rng": 0.0}
        self.last_gps: Optional[np.ndarray] = None
        self.last_rays: List[Dict] = []

    def _n(self, sigma: float) -> float:
        return self.rng.gauss(0, sigma * self.m) if self.m > 0 else 0.0

    def imu(self, true_acc: np.ndarray, dt: float) -> np.ndarray:
        b = self.acc_bias.step(dt)
        return true_acc + b + np.array([self._n(0.1) for _ in range(3)])

    def read(self, t: float, p: np.ndarray, v: np.ndarray, yaw: float, world, pad) -> Dict:
        """Lecturas que tocan en el instante t (cada sensor a su frecuencia)."""
        out = {}
        dt = t - self.t
        self.t = t
        drift = self.gps_drift.step(dt)
        bdrift = self.baro_drift.step(dt)[0]
        if t >= self.next["gps"]:
            self.next["gps"] = t + 0.1
            s = self.prof.gps_sigma * self.rain["gps"]
            gp = p + drift * np.array([1, 1, 1.5]) + np.array([self._n(s), self._n(s), self._n(1.5 * s)])
            out["gps_pos"] = gp
            out["gps_vel"] = v + np.array([self._n(0.1) for _ in range(3)])
            self.last_gps = gp
        if t >= self.next["baro"]:
            self.next["baro"] = t + 0.05
            out["baro"] = p[2] + bdrift + self._n(0.3)
            out["yaw"] = yaw + math.radians(self._n(2.0))
        if t >= self.next["rng"]:
            self.next["rng"] = t + 0.05
            rays = []
            rmax = self.range
            for label, d in ray_dirs(yaw):
                true = world.ray(tuple(p), d, rmax)
                meas = true + self._n((0.02 * true + 0.02) * self.rain["noise"])
                if self.rng.random() < 0.02 * self.m + self.rain["drop"]:
                    meas = rmax  # lectura perdida
                meas = float(np.clip(meas, 0.0, rmax))
                rays.append({"label": label, "dir": d, "dist": meas, "true": true})
            out["rays"] = rays
            self.last_rays = rays
            rel = np.array([pad[0] - p[0], pad[1] - p[1]])
            if np.linalg.norm(rel) < self.rain["cam"] and 0 < p[2] - pad[2] < 10.0:
                out["pad_rel"] = rel + np.array([self._n(0.05), self._n(0.05)])
        return out
