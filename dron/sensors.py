"""Sensores imperfectos. Nivel de ruido: "ideal" (sin ruido), "realista" o "alto" (el doble).

    sensor              frecuencia   modelo (nivel realista)
    IMU (acelerómetro)  200 Hz       aceleración + sesgo (paseo aleatorio) + ruido 0,1 m/s²
    GPS posición        10 Hz        ruido σ del perfil (Mini ±0,75 m, PX4 0,5 m, RTK 0,1 m) + deriva lenta
                                     (Gauss-Markov, τ = 30 s); en vertical, 1,5 veces más
    GPS velocidad       10 Hz        ruido 0,1 m/s
    VIO (interior)      10 Hz        en el almacén no hay GPS: odometría visual-inercial (como la Intel T265 o
                                     VINS-Mono): ruido de 5 cm y 0,05 m/s, pero con una DERIVA que crece con la
                                     distancia recorrida (1 % en una dirección al azar por vuelo + paseo aleatorio).
                                     ESTIMADO: la T265 anunciaba < 1 % de deriva en bucle cerrado.
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
    cámara de            2 Hz        (fase 3, búsqueda) detecta la plataforma dentro de su cono si nada la tapa, con
    detección                        probabilidad según su tamaño en la imagen; 1 % de falsas alarmas (ver search.py)
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

from .search import CAM_HALF_FOV, DET_HZ, FALSE_ALARM, cam_axis, in_cone, p_detect

NOISE_LEVELS = {"ideal": 0.0, "realista": 1.0, "alto": 2.0}
VIO_SIGMA, VIO_VEL, VIO_DRIFT = 0.05, 0.05, 0.01   # odometría visual-inercial (interior): ruido y deriva (ESTIMADO)
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


def clearance_below(depth: Dict, radius: float, ahead=(0.0, 0.0)) -> float:
    """Distancia vertical a lo más alto que hay bajo el dron, medida con la cámara inferior: dentro de un círculo de
    `radius` m alrededor de su vertical y a lo largo del tramo hasta donde estará en breve (`ahead`, en el plano).
    Así ve la copa de un árbol que tiene debajo y un poco por delante. Infinito si no ve nada."""
    d, D = depth["dist"], depth["dirs"]
    ok = np.isfinite(d) & (d < depth["max"] - 1e-3)
    rel = D[ok] * d[ok, None]
    a = np.asarray(ahead, float)
    L2 = float(a @ a)
    s = np.clip((rel[:, :2] @ a) / L2, 0.0, 1.0) if L2 > 1e-9 else np.zeros(len(rel))
    gap = np.linalg.norm(rel[:, :2] - s[:, None] * a, axis=1)   # distancia al tramo [0, ahead]
    rel = rel[gap <= radius]
    return float(-rel[:, 2].max()) if len(rel) else math.inf


SECTORS = 72  # OBSTACLE_DISTANCE de MAVLink / Collision Prevention de PX4: 72 sectores de 5°


def depth_sectors(depth: Dict, band: float, band_down: float = None) -> List[Dict]:
    """Comprime la imagen de profundidad en sectores horizontales, como hace el ordenador de a bordo para la
    Collision Prevention de PX4 (mensaje OBSTACLE_DISTANCE): en cada sector de 5°, la distancia horizontal a lo más
    cercano dentro de una franja vertical de ±`band` m alrededor del dron (lo de más abajo es el suelo; hacia abajo
    la franja llega a `band_down` si se da: al bajar, lo que tiene delante y debajo también cuenta).
    Usa la medida RELATIVA de la cámara (dirección × distancia): no depende del GPS."""
    d, D = depth["dist"], depth["dirs"]
    ok = np.isfinite(d) & (d < depth["max"] - 1e-3)
    rel = D[ok] * d[ok, None]
    lo = band if band_down is None else band_down
    rel = rel[(rel[:, 2] <= band) & (rel[:, 2] >= -lo)]
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
        self.next = {"gps": 0.0, "baro": 0.0, "rng": 0.0, "depth": 0.0, "det": 0.0}
        self.detect_on = False  # cámara de detección: solo en la misión de búsqueda (fase 3)
        self.vio = False        # interior: odometría visual-inercial en lugar de GPS
        self.vio_drift = np.zeros(3)
        self._vio_dir = None
        self._vio_last = None
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
            if self.vio:
                gp = self._vio(p)
                out["gps_pos"] = gp
                out["gps_vel"] = v + np.array([self._n(VIO_VEL) for _ in range(3)])
            else:
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
            dirs = ray_dirs(yaw)
            trues = world.rays(p, np.array([d for _, d in dirs]), rmax)   # los 40 a la vez (misma geometría)
            for (label, d), true in zip(dirs, trues):
                true = float(true)
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
        if self.detect_on and t >= self.next["det"]:
            self.next["det"] = t + 1.0 / DET_HZ
            out["detect"] = self.detect(p, yaw, world)
        return out

    def _vio(self, p):
        """Posición de la odometría visual-inercial: precisa a corto plazo, con deriva proporcional a lo recorrido."""
        if self._vio_dir is None:
            a = self.rng.uniform(-math.pi, math.pi)
            self._vio_dir = np.array([math.cos(a), math.sin(a), 0.2 * self.rng.uniform(-1, 1)])
            self._vio_last = p.copy()
        ds = float(np.linalg.norm(p - self._vio_last))
        self._vio_last = p.copy()
        k = VIO_DRIFT * self.m
        self.vio_drift = self.vio_drift + self._vio_dir * k * ds + np.array([self._n(0.3 * VIO_DRIFT * ds)
                                                                               for _ in range(3)])
        return p + self.vio_drift + np.array([self._n(VIO_SIGMA) for _ in range(3)])

    def detect_vehicle(self, p, yaw, world, target, aim=None):
        """(fase 4) Cámara de detección en un GIMBAL que sigue al vehículo, como el ActiveTrack de DJI: apunta a donde
        el dron predice que está (`aim`) o, si lo ha perdido, al frente y hacia abajo. Lo ve si cae dentro del cono de
        82°, nada lo tapa y según su tamaño en la imagen. Devuelve la posición RELATIVA medida o None."""
        g = self.np_rng
        rel = np.asarray(target, float) - p
        dist = float(np.linalg.norm(rel))
        if aim is not None:
            axis = np.asarray(aim, float) - p
            ok = rel @ axis >= dist * np.linalg.norm(axis) * math.cos(CAM_HALF_FOV)
        else:
            ok = in_cone(rel, yaw)[0]
        if ok and world.ray(tuple(p), tuple(rel / dist), dist) >= dist - 0.4 \
                and g.random() < float(p_detect(dist, self.rain["range"])):
            return rel + g.normal(0, (0.02 * dist + 0.1) * max(self.m, 0.1), 3)
        return None

    def detect(self, p, yaw, world) -> Dict:
        """Una imagen de la cámara de detección: posición RELATIVA medida de la plataforma, o None.
        `false`: si la detección es una falsa alarma (solo para la evaluación; el dron no lo sabe)."""
        g = self.np_rng
        rf = self.rain["range"]
        goal = np.array(world.goal, float)
        rel = goal - p
        dist = float(np.linalg.norm(rel))
        visible = in_cone(rel, yaw)[0] and world.ray(tuple(p), tuple(rel / dist), dist) >= dist - 0.3
        if visible and g.random() < float(p_detect(dist, rf)):
            s = (0.02 * dist + 0.1) * max(self.m, 0.1)
            return {"rel": rel + g.normal(0, s, 3), "false": False}
        if self.m > 0 and g.random() < FALSE_ALARM * self.m:  # sensores ideales: sin falsas alarmas
            # algo que parece la plataforma en un punto al azar de lo que ve la cámara
            a = cam_axis(yaw)
            for _ in range(10):
                d = a + g.normal(0, math.tan(CAM_HALF_FOV) / 2, 3)
                d /= np.linalg.norm(d)
                if in_cone(d, yaw)[0]:
                    t = world.ray(tuple(p), tuple(d), 40.0)
                    if t < 40.0:
                        return {"rel": d * t, "false": True}
        return {"rel": None, "false": False}

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
