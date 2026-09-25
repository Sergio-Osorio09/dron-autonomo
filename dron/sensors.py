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
    cámara de           10 Hz        24 × 14 rayos en un cono de 87° × 58° hacia donde mira el dron (el campo de visión
    profundidad                      de la Intel RealSense D435, la cámara que usan EGO-Planner y los drones del FAST
                                     Lab); alcance el del perfil; ruido 2 % + 2 cm (D435: error < 2 % a 2 m); un 3 %
                                     de píxeles sin dato (se descartan, no se toman como "libre"). Es lo que usa el
                                     dron para CONSTRUIR SU MAPA en la fase 2 (junto con los 40 telémetros). En vuelo,
                                     también alimenta Collision Prevention en 72 sectores (como OBSTACLE_DISTANCE).
    visión inferior     10 Hz        otra cámara de profundidad igual, 16 × 12 rayos, mirando HACIA ABAJO (los DJI
                                     Mini 3 y Matrice 350 llevan visión inferior): mapa del suelo y de lo que hay
                                     debajo, y freno de la bajada sobre copas y arbustos.
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
DEPTH_HFOV, DEPTH_VFOV = 87.0, 58.0   # Intel RealSense D435, hoja de datos (campo de visión de profundidad)
DEPTH_COLS, DEPTH_ROWS, DEPTH_HZ = 24, 14, 10.0
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


def depth_dirs(yaw: float) -> np.ndarray:
    """Direcciones de los píxeles (submuestreados) de la cámara de profundidad, que mira al frente y nivelada
    (estabilizada: se ignora la inclinación del dron)."""
    az = np.radians(np.linspace(-DEPTH_HFOV / 2, DEPTH_HFOV / 2, DEPTH_COLS)) + yaw
    el = np.radians(np.linspace(-DEPTH_VFOV / 2, DEPTH_VFOV / 2, DEPTH_ROWS))
    A, E = np.meshgrid(az, el, indexing="ij")
    return np.stack([np.cos(A) * np.cos(E), np.sin(A) * np.cos(E), np.sin(E)], axis=-1).reshape(-1, 3)


DOWN_COLS, DOWN_ROWS = 16, 12


def depth_down_dirs(yaw: float) -> np.ndarray:
    """Píxeles de la cámara inferior: eje óptico hacia abajo, 58° a lo largo del rumbo y 87° a lo ancho."""
    u = np.tan(np.radians(np.linspace(-DEPTH_VFOV / 2, DEPTH_VFOV / 2, DOWN_ROWS)))   # adelante/atrás
    v = np.tan(np.radians(np.linspace(-DEPTH_HFOV / 2, DEPTH_HFOV / 2, DOWN_COLS)))   # izquierda/derecha
    U, W = np.meshgrid(u, v, indexing="ij")
    f, l = np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    D = U.reshape(-1, 1) * f + W.reshape(-1, 1) * l + np.array([0.0, 0.0, -1.0])
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def clearance_below(depth: Dict, radius: float) -> float:
    """Distancia vertical a lo más alto que hay bajo el dron (dentro de un círculo de `radius` m alrededor de su
    vertical), medida con la cámara inferior. Infinito si no ve nada."""
    d, D = depth["dist"], depth["dirs"]
    ok = np.isfinite(d) & (d < depth["max"] - 1e-3)
    rel = D[ok] * d[ok, None]
    rel = rel[np.hypot(rel[:, 0], rel[:, 1]) <= radius]
    return float(-rel[:, 2].max()) if len(rel) else math.inf


SECTORS = 72  # OBSTACLE_DISTANCE de MAVLink / Collision Prevention de PX4: 72 sectores de 5°


def depth_sectors(depth: Dict, band: float) -> List[Dict]:
    """Comprime la imagen de profundidad en sectores horizontales, como hace el ordenador de a bordo para la
    Collision Prevention de PX4 (mensaje OBSTACLE_DISTANCE): en cada sector de 5°, la distancia horizontal a lo más
    cercano dentro de una franja vertical de ±`band` m alrededor del dron (lo de más abajo es el suelo).
    Usa la medida RELATIVA de la cámara (dirección × distancia): no depende del GPS."""
    d, D = depth["dist"], depth["dirs"]
    ok = np.isfinite(d) & (d < depth["max"] - 1e-3)
    rel = D[ok] * d[ok, None]
    rel = rel[np.abs(rel[:, 2]) <= band]
    if not len(rel):
        return []
    hd = np.hypot(rel[:, 0], rel[:, 1])
    k = ((np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) + 360.0) % 360.0 / (360.0 / SECTORS)).astype(int) % SECTORS
    best = np.full(SECTORS, np.inf)
    np.minimum.at(best, k, hd)
    out = []
    for i in np.flatnonzero(np.isfinite(best)):
        a = math.radians((i + 0.5) * 360.0 / SECTORS)
        out.append({"label": "profundidad", "dir": (math.cos(a), math.sin(a), 0.0), "dist": float(best[i])})
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
        self.next = {"gps": 0.0, "baro": 0.0, "rng": 0.0, "depth": 0.0}
        self.np_rng = np.random.default_rng(seed)
        self.last_gps: Optional[np.ndarray] = None
        self.last_rays: List[Dict] = []
        self.depth_on = False   # la cámara de profundidad solo hace falta para construir el mapa (fase 2)

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
                if true >= rmax - 1e-6:
                    meas = rmax  # sin eco: el sensor dice "nada dentro del alcance", no una distancia con ruido
                rays.append({"label": label, "dir": d, "dist": meas, "true": true})
            out["rays"] = rays
            self.last_rays = rays
            rel = np.array([pad[0] - p[0], pad[1] - p[1]])
            if np.linalg.norm(rel) < self.rain["cam"] and 0 < p[2] - pad[2] < 10.0:
                out["pad_rel"] = rel + np.array([self._n(0.05), self._n(0.05)])
        if self.depth_on and t >= self.next["depth"]:
            self.next["depth"] = t + 1.0 / DEPTH_HZ
            out["depth"] = self.depth(p, yaw, world)
            out["depth_down"] = self.depth(p, yaw, world, down=True)
        return out

    def depth(self, p, yaw, world, down: bool = False) -> Dict:
        """Imagen de profundidad: direcciones y distancias medidas (NaN = píxel sin dato)."""
        D = depth_down_dirs(yaw) if down else depth_dirs(yaw)
        rmax = self.range
        true = world.rays(p, D, rmax)
        g = self.np_rng
        meas = true + g.normal(0, 1, len(D)) * (0.02 * true + 0.02) * self.rain["noise"] * self.m
        meas = np.clip(meas, 0.0, rmax)
        meas[true >= rmax - 1e-6] = rmax      # nada dentro del alcance
        meas[g.random(len(D)) < 0.03 * self.m + self.rain["drop"]] = np.nan
        return {"dirs": D, "dist": meas, "max": rmax}
