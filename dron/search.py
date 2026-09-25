"""Misión de búsqueda (fase 3): el dron NO sabe dónde está la meta, solo que está dentro de una ZONA DE BÚSQUEDA.

Cámara de detección (la de la cámara principal, como la de un DJI Mini 3: campo de visión de 82,1°):
  * mira hacia delante y hacia abajo (60° bajo el horizonte, como en las misiones de búsqueda con dron);
  * detecta la plataforma (1,5 m de diámetro) si está dentro del cono, no hay nada en medio (oclusión) y según los
    píxeles que ocupa en la imagen del detector (entrada de 640 px, la estándar de YOLO):
        píxeles = 1,5 m · f / distancia,   f = 320 / tan(41°) ≈ 368 px
        P(detección) = 0,9 · clip((píxeles − 12) / (32 − 12), 0, 1)
    → 0,9 hasta 17 m, 0 a partir de 46 m (32 px: tamaño fiable para un detector; 12 px: nada). ESTIMADO.
  * 10 imágenes por segundo (una OAK-D Lite ejecuta YOLOv6n a ~60 por segundo según Luxonis; 10 es prudente y deja
    CPU): 2 → 10 acorta la búsqueda un 5 % (PX4, 40 semillas) y cuesta un 23 % más de cálculo. Un 1 % de las
    imágenes da una FALSA ALARMA en un punto al azar de lo que ve. La lluvia reduce el alcance como el de los demás
    sensores. (Las imágenes se tratan como independientes, como en la teoría de búsqueda de Koopman; con imágenes
    muy seguidas es algo optimista.)
  Es un sensor: usa el mundo real (sensors.py lo simula con `detect`).

Creencia (lo que sabe el dron), como en la búsqueda y rescate (teoría de búsqueda de Koopman; manual IAMSAR):
  * rejilla de 2 m sobre la zona con la probabilidad de que el objetivo esté en cada celda (a priori uniforme);
  * cada imagen SIN detección: P(c) ∝ P(c) · (1 − Pd(c)), con Pd calculada desde la posición ESTIMADA y comprobando la
    oclusión con el mapa que tiene el dron (el real si es conocido, el aprendido si no);
  * "cubierta" = celda con probabilidad acumulada de no haberla visto (∏(1 − Pd)) menor que 0,5.
  * una detección no se cree a la primera: el dron va a CONFIRMARLA (se acerca y la mira desde 5 m). Con 3
    detecciones que coinciden (a menos de 1,5 m), confirmada; si no, era una falsa alarma y sigue buscando.

Estrategias (qué punto visitar a continuación):
  * barrido     boustrophedon (cortacésped): pasadas paralelas al lado largo cada 12 m (la franja que ve bien la
                cámara a 8 m de altura mide ~16 m), en zigzag. La referencia clásica de cobertura.
  * fronteras   como FUEL (Zhou et al., 2021): fronteras = celdas sin cubrir junto a celdas cubiertas, agrupadas; va
                al grupo con más celdas por metro de viaje, a un punto 5 m antes para que la cámara lo vea.
  * bayesiana   búsqueda bayesiana voraz: va al punto donde puede eliminar más probabilidad por metro recorrido
                (masa de probabilidad a la vista / (distancia + 10 m)).
"""
import math
from typing import Dict, List, Optional

import numpy as np

from .world import LENGTH, WIDTH

STRATEGIES = ("barrido", "fronteras", "bayesiana")
SEARCH_MODES = ("no",) + STRATEGIES
CELL = 2.0
SEARCH_AGL = 8.0            # altura de búsqueda sobre el suelo (12 m no mejora: 21,4 frente a 21,8 s, 30 semillas)
CONFIRM_AGL = 5.0           # altura para confirmar una detección
CAM_HALF_FOV = math.radians(82.1 / 2)   # DJI Mini 3: campo de visión 82,1°
CAM_PITCH = math.radians(60.0)          # bajo el horizonte
DET_F = 320.0 / math.tan(math.radians(41.0))
PAD_SIZE = 1.5
DET_HZ = float(__import__("os").environ.get("DRON_DET_HZ", 10.0))
CONFIRM_LOOKS = max(8, int(round(1.5 * DET_HZ)))   # imágenes de cerca sin confirmarla → falsa alarma (≥ 1,5 s)
FALSE_ALARM = 0.01
LANE = 12.0
COVERED = 0.5


def cam_axis(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw) * math.cos(CAM_PITCH), math.sin(yaw) * math.cos(CAM_PITCH), -math.sin(CAM_PITCH)])


def p_detect(dist, range_factor: float = 1.0):
    """Probabilidad de detectar la plataforma a una distancia dada (vectorizada)."""
    px = PAD_SIZE * DET_F * range_factor / np.maximum(dist, 1e-3)
    return 0.9 * np.clip((px - 12.0) / 20.0, 0.0, 1.0)


def in_cone(rel: np.ndarray, yaw: float) -> np.ndarray:
    rel = np.atleast_2d(rel)
    n = np.linalg.norm(rel, axis=1)
    return (rel @ cam_axis(yaw)) >= n * math.cos(CAM_HALF_FOV)


def make_area(world, seed: int):
    """Zona de búsqueda de 30-40 m × 20-30 m que contiene la meta en un punto al azar (no en el centro)."""
    rng = np.random.default_rng(seed + 7)
    gx, gy = world.goal[:2]
    w, h = rng.uniform(30, 40), rng.uniform(20, 30)
    x0 = float(np.clip(gx - rng.uniform(0.15, 0.85) * w, 1.0, LENGTH - 1.0 - w))
    y0 = float(np.clip(gy - rng.uniform(0.15, 0.85) * h, 1.0, WIDTH - 1.0 - h))
    return (round(x0, 1), round(x0 + w, 1), round(y0, 1), round(y0 + h, 1))


class Searcher:
    """Creencia sobre dónde está el objetivo y elección del siguiente punto a visitar."""

    def __init__(self, area, strategy: str, surface, visible, range_factor: float = 1.0, prior=None,
                 target_speed: float = 0.0):
        if strategy not in STRATEGIES:
            raise ValueError("Estrategia de búsqueda desconocida %r" % strategy)
        self.area, self.strategy = area, strategy
        x0, x1, y0, y1 = area
        self.shape = (max(1, int(round((x1 - x0) / CELL))), max(1, int(round((y1 - y0) / CELL))))
        nx, ny = self.shape
        self.cx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        self.cy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        self.P = np.full(self.shape, 1.0 / (nx * ny))
        if prior is not None:   # (fase 4) a priori gaussiano alrededor de la última posición vista
            (px, py), sig = prior
            X, Y = np.meshgrid(self.cx, self.cy, indexing="ij")
            self.P = np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * sig * sig)) + 1e-6
            self.P /= self.P.sum()
        self.target_speed = target_speed   # > 0: el blanco se mueve (la creencia se difunde con el tiempo)
        self._pending_dt = 0.0
        self.Q = np.ones(self.shape)            # probabilidad acumulada de NO haberla visto
        self.surface = surface                  # altura del suelo en (x, y) según el dron
        self.visible = visible                  # oclusión según el mapa del dron: visible(origen, puntos) -> bool[]
        self.range_factor = range_factor
        self.detections: List[Dict] = []       # {"pos", "state": "pendiente" | "confirmada" | "falsa"}
        self.candidate: Optional[Dict] = None
        self.confirm_hits: List[np.ndarray] = []
        self.confirm_looks = 0
        self.counted_hits = 0
        self._hit_now = False
        self.recent = []                        # detecciones sueltas recientes (glimpse, posición): M de N
        self.goal: Optional[np.ndarray] = None
        self.lanes: Dict[int, List[np.ndarray]] = {}   # franjas pendientes de cada dron (en enjambre, las suyas)
        self.lane_pass: Dict[int, int] = {}
        self.glimpses = 0
        self._Z = None
        self.rejected: List[np.ndarray] = []   # puntos a los que no se pudo llegar (no se vuelven a proponer)
        self._center = None                     # zona que se quería mirar con el último punto propuesto

    # ------------------------------------------------------------------ creencia
    def cells_xyz(self):
        X, Y = np.meshgrid(self.cx, self.cy, indexing="ij")
        if self._Z is None or self.glimpses % 4 == 0:  # la superficie (sobre todo la aprendida) cambia despacio
            self._Z = np.array([[self.surface(x, y) for y in self.cy] for x in self.cx])
        return np.stack([X, Y, self._Z + 0.05], axis=-1)

    def glimpse(self, est_p, yaw, detection, visible=None, by=0):
        """Una imagen de la cámara. `detection`: posición medida de la plataforma o None. En un enjambre, cada dron
        mira con su propia oclusión (`visible`) y la creencia es la de todos (`by` = quién mira)."""
        self.glimpses += 1
        self._hit_now = False
        vis_fn = visible or self.visible
        if detection is not None:
            if self.candidate is not None and np.linalg.norm(detection[:2] - self.candidate["pos"][:2]) < 3.0:
                self.confirm_hits.append(np.asarray(detection))
                self._hit_now = True
            elif self.candidate is None:
                # M de N (como el seguimiento de blancos de un radar): una detección suelta no crea candidata; hace
                # falta otra a menos de 3 m en las últimas imágenes (~0,5 s). A 10 imágenes por segundo, un 1 % de
                # falsas alarmas por imagen es una cada 10 s: el dron las perseguía todas (30 en un vuelo) y no
                # llegaba a encontrar la plataforma. La plataforma, que sale en el 90 % de las imágenes, no lo nota
                d = np.asarray(detection, float)
                window = max(2, int(round(0.5 * DET_HZ)))
                self.recent = [(g, q) for g, q in self.recent if self.glimpses - g <= window]
                prev = [q for _, q in self.recent if np.linalg.norm(q[:2] - d[:2]) < 3.0]
                self.recent.append((self.glimpses, d))
                if prev:
                    self.candidate = {"pos": d, "state": "pendiente", "by": by}
                    self.confirm_hits, self.confirm_looks, self.counted_hits = prev[-1:] + [d], 0, 0
                    self.detections.append(self.candidate)
                    self.recent = []
            return
        C = self.cells_xyz().reshape(-1, 3)
        rel = C - est_p
        dist = np.linalg.norm(rel, axis=1)
        pd = p_detect(dist, self.range_factor) * in_cone(rel, yaw)
        idx = np.flatnonzero(pd > 0.01)
        if len(idx):
            vis = vis_fn(est_p, C[idx])
            pd[idx] *= vis
            pd[np.setdiff1d(np.arange(len(pd)), idx)] = 0.0
        pd = pd.reshape(self.shape)
        self.P *= (1.0 - pd)
        s = self.P.sum()
        self.P = self.P / s if s > 1e-12 else np.full(self.shape, 1.0 / self.P.size)
        self.Q *= (1.0 - pd)

    def predict(self, dt: float):
        """(fase 4) Modelo de movimiento del blanco (cadena de Markov de la teoría de búsqueda de blancos móviles):
        cada segundo, la probabilidad se reparte en un disco del radio que puede recorrer a su velocidad máxima."""
        if self.target_speed <= 0:
            return
        self._pending_dt += dt
        if self._pending_dt < 1.0:
            return
        from scipy.ndimage import convolve
        r = self.target_speed * self._pending_dt / CELL
        self._pending_dt = 0.0
        n = int(math.ceil(r))
        k = np.array([[1.0 if i * i + j * j <= r * r else 0.0 for j in range(-n, n + 1)] for i in range(-n, n + 1)])
        self.P = convolve(self.P, k / k.sum(), mode="constant") + 1e-9
        self.P /= self.P.sum()
        self.Q = np.minimum(1.0, self.Q + 0.2)   # lo ya mirado deja de estar "cubierto": se puede haber movido

    def covered(self) -> float:
        return float((self.Q < COVERED).mean())

    # ------------------------------------------------------------------ confirmación
    def confirm_step(self, counts: bool = True):
        """Cuenta una imagen durante la confirmación. Devuelve "confirmada", "falsa" o None (sigue mirando).
        `counts`=False: la imagen no cuenta como "mirada" (confirmando en vuelo, la detección aún está lejos o fuera
        del cono: no verla no dice nada)."""
        self.confirm_looks += 1 if counts else 0
        self.counted_hits += 1 if counts and self._hit_now else 0
        # 3 detecciones que coinciden (a menos de 1,5 m de su mediana). Con 2 no basta: al mirar fijamente la
        # zona, dos falsas alarmas pueden caer en el mismo sitio (pasaba en ~1 de cada 20 búsquedas). Con la
        # plataforma de verdad, de cerca, la probabilidad por imagen es 0,9: 3 de 8 es casi seguro
        H = np.array(self.confirm_hits)
        near = H[np.linalg.norm(H[:, :2] - np.median(H[:, :2], axis=0), axis=1) < 1.5]
        # y vistas DE CERCA (en imágenes que "cuentan") al menos 3 veces y en al menos el 40 % de esas imágenes (la
        # plataforma, de cerca, sale en el 90 %; una falsa alarma, en el 1 %). Sin esto, con 10 imágenes por
        # segundo: acercándose desde lejos, o con 3 drones recién despegados mirando la misma zona pequeña, 3 falsas
        # alarmas cayeron en el mismo sitio, se confirmaron y el dron aterrizó a 50 m de la plataforma
        if len(near) >= 3 and self.counted_hits >= 3 and self.counted_hits >= 0.4 * self.confirm_looks:
            self.candidate["pos"] = near.mean(axis=0)
            self.candidate["state"] = "confirmada"
            return "confirmada"
        if self.confirm_looks >= CONFIRM_LOOKS:   # mirándola de cerca sin confirmarla: falsa alarma
            self.candidate["state"] = "falsa"
            self.candidate = None
            return "falsa"
        return None

    # ------------------------------------------------------------------ siguiente punto
    def _z(self, x, y, agl=SEARCH_AGL):
        return self.surface(x, y) + agl

    def next_goal(self, est_p, team=None, agl=0.0) -> np.ndarray:
        """Siguiente punto. `team` (enjambre): {"me": i, "n": n, "others": [pos xy], "goals": [xy]} para repartirse:
        barrido → cada dron sus franjas; fronteras y bayesiana → solo su celda de VORONOI (lo que tiene más cerca que
        ningún otro, como en la cobertura de Cortés et al.) y lejos de los puntos que ya han elegido los demás
        (subasta voraz: el primero que lo pide se lo queda). `agl`: capa de altura propia (separación vertical)."""
        self._team = team
        x, y = {"barrido": self._lawnmower, "fronteras": self._frontier, "bayesiana": self._bayes}[self.strategy](est_p)
        self._team = None
        self.goal = np.array([x, y, self._z(x, y, SEARCH_AGL + agl)])
        return self.goal

    def _team_mask(self, est_p):
        """Celdas permitidas para este dron en el enjambre (Voronoi y sin los puntos de los demás)."""
        t = getattr(self, "_team", None)
        X, Y = np.meshgrid(self.cx, self.cy, indexing="ij")
        ok = np.ones(self.shape, bool)
        if not t:
            return ok
        d_me = np.hypot(X - est_p[0], Y - est_p[1])
        for q in t["others"]:
            ok &= d_me <= np.hypot(X - q[0], Y - q[1])
        for g in t["goals"]:
            ok &= np.hypot(X - g[0], Y - g[1]) > 8.0
        return ok if ok.any() else np.ones(self.shape, bool)

    def reject(self, goal):
        """El planificador no encuentra camino hasta `goal` (p. ej. un edificio que acaba de descubrir)."""
        self.rejected.append(np.asarray(goal[:2], float))
        if self._center is not None:           # y la zona que se quería mirar desde ahí
            self.rejected.append(self._center)

    def _ok(self, xy) -> bool:
        return all(np.linalg.norm(np.asarray(xy) - r) > 4.0 for r in self.rejected)

    def _lawnmower(self, est_p):
        self._center = None
        t = getattr(self, "_team", None)
        me = t["me"] if t else 0
        lanes = self.lanes.setdefault(me, [])
        if not lanes:
            x0, x1, y0, y1 = self.area
            long_x = (x1 - x0) >= (y1 - y0)
            a0, a1, b0, b1 = (x0, x1, y0, y1) if long_x else (y0, y1, x0, x1)
            off = LANE / 2 if self.lane_pass.get(me, 0) % 2 == 0 else LANE   # otra pasada desplazada media franja
            bs = list(np.arange(b0 + off, b1, LANE)) or [(b0 + b1) / 2]
            if t and t["n"] > 1:   # enjambre: franjas intercaladas (la k-ésima para el dron k mod n)
                mine = [b for k, b in enumerate(bs) if k % t["n"] == t["me"]]
                bs = mine or bs
            pts = []
            for k, b in enumerate(bs):
                ends = (a0 + 2, a1 - 2) if k % 2 == 0 else (a1 - 2, a0 + 2)
                pts += [np.array([e, b]) if long_x else np.array([b, e]) for e in ends]
            # empezar por la esquina más cercana
            d_start = np.linalg.norm(pts[0] - est_p[:2])
            d_end = np.linalg.norm(pts[-1] - est_p[:2])
            lanes[:] = pts if d_start <= d_end else pts[::-1]
            self.lane_pass[me] = self.lane_pass.get(me, 0) + 1
        p = lanes.pop(0)
        return float(p[0]), float(p[1])

    def _frontier(self, est_p):
        from scipy.ndimage import binary_dilation, label
        cov = self.Q < COVERED
        if not cov.any():  # nada cubierto aún: al punto de la zona más cercano
            return self._nearest_in_area(est_p)
        front = ~cov & binary_dilation(cov) & self._team_mask(est_p)
        if not front.any():
            return self._bayes(est_p)
        lab, n = label(front, structure=np.ones((3, 3)))
        best, best_score = None, -1.0
        for k in range(1, n + 1):
            ii, jj = np.nonzero(lab == k)
            c = np.array([self.cx[ii].mean(), self.cy[jj].mean()])
            d = float(np.linalg.norm(c - est_p[:2]))
            score = len(ii) / (d + 5.0)
            if score > best_score and self._ok(c):
                best, best_score = c, score
        if best is None:
            return self._bayes(est_p)
        self._center = best
        u = best - est_p[:2]
        n = np.linalg.norm(u)
        p = best - (u / n) * min(5.0, 0.5 * n) if n > 1e-6 else best  # un poco antes: la cámara mira adelante
        return float(p[0]), float(p[1])

    def _bayes(self, est_p):
        from scipy.ndimage import convolve
        r = int(round(8.0 / CELL))
        k = np.array([[1.0 if i * i + j * j <= r * r else 0.0 for j in range(-r, r + 1)] for i in range(-r, r + 1)])
        mass = convolve(self.P, k, mode="constant")
        X, Y = np.meshgrid(self.cx, self.cy, indexing="ij")
        d = np.hypot(X - est_p[0], Y - est_p[1])
        score = mass / (d + 10.0)
        score[~self._team_mask(est_p)] = -1.0
        for r in self.rejected:
            score[np.hypot(X - r[0], Y - r[1]) <= 4.0] = -1.0
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        c = np.array([self.cx[i], self.cy[j]])
        self._center = c
        u = c - est_p[:2]
        n = np.linalg.norm(u)
        p = c - (u / n) * min(5.0, 0.5 * n) if n > 1e-6 else c
        return float(p[0]), float(p[1])

    def _nearest_in_area(self, est_p):
        x0, x1, y0, y1 = self.area
        return float(np.clip(est_p[0], x0 + 3, x1 - 3)), float(np.clip(est_p[1], y0 + 3, y1 - 3))

    def to_dict(self):
        m = float(self.P.max())
        return {"area": list(self.area), "shape": list(self.shape), "strategy": self.strategy,
                "p": (np.round(999 * self.P / m).astype(int).ravel().tolist() if m > 0 else []),
                "covered": self.covered(),
                "goal": None if self.goal is None else self.goal.round(2).tolist(),
                "detections": [{"pos": d["pos"].round(2).tolist(), "state": d["state"]} for d in self.detections]}
