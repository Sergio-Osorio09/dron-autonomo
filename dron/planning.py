"""Planificación: camino global + trayectoria con perfil de velocidad.

1. Rejilla 3D de 1 m con el campo de distancias al obstáculo más cercano (ESDF, como Voxblox/FIESTA): cada celda
   sabe cuánta holgura tiene frente a obstáculos, bordes y TERRENO. Las celdas con menos de radio + margen de
   seguridad quedan bloqueadas. La rejilla solo llega hasta 8 m por encima de lo más alto del mapa.
2. A* ponderado (heurística × 1,3, "weighted A*": algo menos óptimo y mucho más rápido) en 26 vecinos. El coste de
   paso crece cerca de los obstáculos, así el camino prefiere pasillos anchos (como el término de "distancia a
   obstáculos" de EGO-Planner).
3. Estirado de cuerda (string pulling / line-of-sight, lo mismo que hace Theta*): se quitan los puntos
   intermedios mientras el segmento directo siga libre. Quedan tramos rectos largos.
4. Suavizado: B-spline cúbica optimizada como EGO-Planner (Zhou et al., 2021) si SMOOTHER = "bspline": puntos de
   control cada metro sobre el camino y descenso de gradiente con coste de suavidad (aceleración de la curva,
   diferencias segundas de los puntos de control) + coste de colisión (penalización cuadrática si la holgura del
   ESDF baja de radio + margen + 0,4 m, con su gradiente). Si la curva no queda libre, se usa Chaikin (3
   iteraciones comprobando colisión), que es el método por defecto. Después, remuestreo cada 0,25 m.
5. Perfil de velocidad óptimo en tiempo (el enfoque TOPP de los planificadores de trayectorias):
     * límite por curvatura: v ≤ √(a_lat / κ) (en curvas cerradas hay que frenar);
     * límite vertical: la componente vertical no supera la subida/bajada máxima del dron;
     * pasada hacia delante (acelerar como mucho a_max) y hacia atrás (frenar a tiempo): así va rápido en las
       rectas y frena justo antes de las curvas y del final (en modo carrera NO frena al final: cruza la meta);
     * suavizado del perfil para aproximar el límite de tirón (jerk);
     * al REPLANIFICAR en vuelo, la trayectoria nueva arranca a la velocidad actual (si no, frenaría cada vez);
     * y, si se le da un `prefix` (el tramo inmediato de la trayectoria actual), se COSE a él como en Fast-Planner
       y EGO-Planner: se conserva ese tramo, se planifica desde su final y el suavizado incluye el punto anterior,
       así la curva nueva sale en la dirección que ya lleva el dron, sin saltos.
   El resultado es una referencia p(t), v(t), a(t) que el control sigue con prealimentación.

Todo funciona igual con el mundo conocido (fase 1: `ClearanceGrid(world)` + `world.clearance`) que con el mapa
que construye el dron (fase 2: `mapping.LearnedGrid`, que tiene la misma interfaz). Por eso las funciones reciben
un `space`: cualquier objeto con `clearance(p, limit)`.
"""
import heapq
import os
import math
from typing import List, Optional, Tuple

import numpy as np

from .params import Profile
from .world import CEILING, LENGTH, WIDTH, Cylinder, World

CELL = 1.0
MARGIN = 0.6
DS = 0.25


class ClearanceGrid:
    def __init__(self, world: World):
        top = max([float(world.terrain.H.max())] + [o.top for o in world.obstacles] + [world.goal[2]]) + 8.0
        nx, ny, nz = int(LENGTH / CELL), int(WIDTH / CELL), int(min(CEILING, top) / CELL)
        self.shape = (nx, ny, nz)
        xs, ys, zs = ((np.arange(n) + 0.5) * CELL for n in (nx, ny, nz))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        ground = world.terrain.heights(X[:, :, 0], Y[:, :, 0])[:, :, None]
        clr = np.minimum.reduce([Z - ground, X, LENGTH - X, Y, WIDTH - Y])
        for ob in world.obstacles:
            x0, x1, y0, y1 = ob.footprint()
            i0, i1 = max(0, int(x0 / CELL) - 4), min(nx, int(x1 / CELL) + 5)   # solo la zona cercana
            j0, j1 = max(0, int(y0 / CELL) - 4), min(ny, int(y1 / CELL) + 5)
            Xs, Ys, Zs = X[i0:i1, j0:j1], Y[i0:i1, j0:j1], Z[i0:i1, j0:j1]
            dz = np.maximum.reduce([ob.base - Zs, np.zeros_like(Zs), Zs - ob.top])
            if isinstance(ob, Cylinder):
                dxy = np.hypot(Xs - ob.x, Ys - ob.y) - ob.r
                d = np.where(dz == 0, dxy, np.where(dxy > 0, np.hypot(np.maximum(dxy, 0), dz), dz))
            else:
                dx = np.maximum.reduce([ob.x0 - Xs, np.zeros_like(Xs), Xs - ob.x1])
                dy = np.maximum.reduce([ob.y0 - Ys, np.zeros_like(Ys), Ys - ob.y1])
                d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                d = np.where(d == 0, -1.0, d)
            clr[i0:i1, j0:j1] = np.minimum(clr[i0:i1, j0:j1], d)
        self.clr = clr

    def cell(self, p) -> Tuple[int, int, int]:
        return tuple(min(max(int(v / CELL), 0), n - 1) for v, n in zip(p, self.shape))

    @staticmethod
    def center(c) -> np.ndarray:
        return (np.array(c) + 0.5) * CELL


# VUELO HORIZONTAL. Antes, el coste de A* por pasar cerca de algo usaba la holgura 3D, en la que el SUELO cuenta:
# volar a 1,5 m "costaba" como rozar un obstáculo y el camino subía a 3 m y luego bajaba a la meta (el X650 pasaba el
# 21-26 % de la carrera subiendo o bajando a más de 1 m/s, y todo el empuje que va a subir no va hacia delante).
# Ahora la cercanía se mide en HORIZONTAL (distancia en su propio plano a lo sólido: un obstáculo o una ladera al
# lado sí cuentan, el suelo de debajo no) y subir o bajar cuesta VZ_COST veces más que avanzar.
# PROBADO Y DESCARTADO (26-09-2026, X650, 60 semillas por caso, clásico e híbrido): la mitad de tiempo subiendo o
# bajando (26 → 14 %) y altura media 3,1 → 2,0 m, pero ni más rápido (+1 a +6 %: volando bajo, los arbustos de
# 2-3 m quedan en el camino y antes los sobrevolaba) ni menos energía (590 W igual). Lo que limita la velocidad es
# frenar dentro del alcance de los sensores, no el empuje. Por defecto, como antes (DRON_NEAR=h, DRON_VZ_COST=2: la
# variante)
NEAR_MODE = os.environ.get("DRON_NEAR", "3d")
VZ_COST = float(os.environ.get("DRON_VZ_COST", 1.0))
# COSTE DE LO NO VISTO (planificación "optimista" corregida, como los planificadores de exploración): con el mapa
# desconocido, A* trata lo no observado como libre y traza la ruta por ahí; al descubrir algo, hay que desviarse y
# frenar (el clásico era un 13-16 % más lento sin mapa que con él). Cada metro por lo no visto cuesta UNKNOWN_COST
# veces más. PROBADO Y DESACTIVADO (clásico en el X650, 60 semillas por caso): con 1,3, igual o +4 % en el bosque; con
# 2, 4 choques de 60 en cada escenario (rutas pegadas a lo ya visto, junto a los obstáculos). DRON_UNKNOWN_COST=1,3: con él
UNKNOWN_COST = float(os.environ.get("DRON_UNKNOWN_COST", 1.0))
MAX_EXPANSIONS = 60_000  # presupuesto de nodos: una búsqueda nunca congela la simulación


def _components(grid: ClearanceGrid, need: float):
    """Componentes conexas (26 vecinos) del espacio libre para una holgura dada, en caché por holgura."""
    from scipy.ndimage import label
    cache = grid.__dict__.setdefault("_comp", {})
    key = round(need, 2)
    if key not in cache:
        cache[key] = label(grid.clr > need, structure=np.ones((3, 3, 3)))[0]
    return cache[key]


def _labels_near(labels, c):
    """Etiquetas de componente de la celda y sus vecinas (salida/meta pueden rozar el margen)."""
    sl = tuple(slice(max(0, v - 1), v + 2) for v in c)
    return set(np.unique(labels[sl]).tolist()) - {0}


def astar(grid: ClearanceGrid, start, goal, need: float,
          max_expansions: int = MAX_EXPANSIONS) -> Optional[List[Tuple[int, int, int]]]:
    """A* ponderado sobre la rejilla, con índices planos y arrays de numpy (varias veces más rápido que con
    diccionarios). Antes de buscar comprueba que salida y meta estén en la misma zona libre conectada: si no, no hay
    camino y lo dice al instante, en vez de explorar el mapa entero (llegaba a tardar 18 s)."""
    labels = _components(grid, need)
    s, g = grid.cell(start), grid.cell(goal)
    if not (_labels_near(labels, s) & _labels_near(labels, g)):
        return None
    nx, ny, nz = grid.shape
    px, py, pz = nx + 2, ny + 2, nz + 2           # borde de 1 celda bloqueada: sin comprobar límites
    # libre y coste de cada celda como LISTAS de Python (leer elementos sueltos de un array de numpy en el bucle
    # es varias veces más lento), guardadas por rejilla y holgura: las replanificaciones sobre el mismo mapa las
    # reutilizan
    cache = grid.__dict__.setdefault("_astar", {})
    key = round(need, 2)
    if key not in cache:
        free = np.zeros((px, py, pz), bool)
        free[1:-1, 1:-1, 1:-1] = grid.clr > need
        near = np.ones((px, py, pz))
        if NEAR_MODE == "h":
            from scipy.ndimage import distance_transform_edt
            solid = grid.clr <= 0.0
            # distancia en el plano de cada altura (en z, un paso enorme: no cuenta lo de arriba ni lo de abajo)
            hclr = distance_transform_edt(~solid, sampling=(CELL, CELL, 1e6)) - 0.5 * CELL if solid.any() \
                else np.full(grid.clr.shape, 1e6)
            near[1:-1, 1:-1, 1:-1] = 1.0 + 1.5 * np.clip(2.5 - hclr, 0, 2.5) / 2.5
        else:
            near[1:-1, 1:-1, 1:-1] = 1.0 + 1.5 * np.clip(2.5 - grid.clr, 0, 2.5) / 2.5  # más caro cerca de obstáculos
        unseen = getattr(grid, "unseen", None)
        if UNKNOWN_COST != 1.0 and unseen is not None and unseen.shape == grid.clr.shape:
            near[1:-1, 1:-1, 1:-1] *= np.where(unseen, UNKNOWN_COST, 1.0)
        cache[key] = (free.ravel().tolist(), near.ravel().tolist())
    free_f, near_f = cache[key]
    flat = lambda c: ((c[0] + 1) * py + (c[1] + 1)) * pz + (c[2] + 1)
    si, gi = flat(s), flat(g)
    fs, fg = free_f[si], free_f[gi]
    free_f[si] = free_f[gi] = True  # salida/meta: se permite aunque rocen el margen (se restauran al acabar)
    moves = [((dx * py + dy) * pz + dz, math.sqrt(dx * dx + dy * dy + (VZ_COST * dz) ** 2))
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    gx, gy, gz = g[0] + 1, g[1] + 1, g[2] + 1
    best = {si: 0.0}                              # solo las celdas visitadas
    parent = {si: -1}

    def h(i):
        z = i % pz
        y = (i // pz) % py
        x = i // (py * pz)
        return 1.3 * math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)  # A* ponderado

    heap = [(h(si), 0.0, si)]
    expanded = 0
    inf = math.inf
    try:
        while heap:
            _, cost, i = heapq.heappop(heap)
            if i == gi:
                path = []
                while i != -1:
                    x, rem = divmod(i, py * pz)
                    y, z = divmod(rem, pz)
                    path.append((x - 1, y - 1, z - 1))
                    i = parent[i]
                return path[::-1]
            if cost > best.get(i, inf):
                continue
            expanded += 1
            if expanded > max_expansions:
                return None
            for off, length in moves:
                n = i + off
                if not free_f[n]:
                    continue
                nc = cost + length * near_f[n]
                if nc < best.get(n, inf):
                    best[n] = nc
                    parent[n] = i
                    heapq.heappush(heap, (nc + h(n), nc, n))
        return None
    finally:
        free_f[si], free_f[gi] = fs, fg


def segment_clear(space, a, b, need: float, step: float = 0.3) -> bool:
    n = max(1, int(np.linalg.norm(b - a) / step))
    many = getattr(space, "clearance_many", None)
    if many is not None:  # mapa aprendido: todos los puntos de una vez
        P = a + (b - a) * (np.arange(n + 1)[:, None] / n)
        return bool((many(P) >= need).all())
    for k in range(n + 1):
        p = a + (b - a) * (k / n)
        if space.clearance(tuple(p), need + 0.5, ground=True) < need:
            return False
    return True


def trajectory_clear(space, traj, t_from: float, need: float, horizon: float = None) -> bool:
    """¿Sigue libre lo que queda de la trayectoria (desde t_from, hasta `horizon` s más si se da)?"""
    i0 = int(np.searchsorted(traj.t, t_from))
    i1 = len(traj.t) if horizon is None else int(np.searchsorted(traj.t, t_from + horizon)) + 1
    P = traj.P[i0:i1]
    if not len(P):
        return True
    return bool((space.clearance_many(P) >= need).all())


# APROXIMACIÓN HORIZONTAL a la meta que se cruza sin frenar (carrera). Las celdas de A* son de 1 m de alto: el camino
# llegaba a la altura de la celda de encima (2,5 m) y, justo antes de la puerta (1,5 m), caía 1 m de golpe. Ese quiebro
# obligaba al perfil de velocidad a frenar de 9 a ~1 m/s y el frenazo se propagaba hacia atrás: la trayectoria media
# del X650 no pasaba de 5,3 m/s de pico. Ahora, si hay sitio, se añade un punto a la altura de la puerta APPROACH
# metros antes (como la última recta de un piloto de carreras): baja pronto y suave, y cruza la puerta nivelado.
# PROBADO Y DESACTIVADO (X650, 60 semillas por caso): la trayectoria cruza la puerta a 7 m/s en vez de 3,3, pero la
# carrera no acelera (clásico +1 a +4 %; el híbrido no la usa): antes de llegar ahí el dron ha replanificado 20-25 veces
# y cerca de la puerta manda la guía terminal. DRON_APPROACH=6: con ella
APPROACH = float(os.environ.get("DRON_APPROACH", 0.0))


def level_approach(world: World, pts: List[np.ndarray], need: float) -> List[np.ndarray]:
    if len(pts) < 2:
        return pts
    goal = pts[-1]
    for L in (APPROACH, 0.6 * APPROACH, 0.35 * APPROACH):
        # los puntos del camino a menos de L m (en horizontal) de la puerta se sustituyen por el de aproximación
        k = len(pts) - 2
        while k > 0 and float(np.linalg.norm(pts[k][:2] - goal[:2])) < L:
            k -= 1
        prev = pts[k]
        d = goal[:2] - prev[:2]
        n = float(np.linalg.norm(d))
        if n < 1.0:
            continue
        Lk = min(L, 0.8 * n)
        # a la altura de la puerta o, si ahí no cabe con la holgura (la puerta está a ~1 m del suelo y la holgura del
        # planificador, radio + margen, es ~1,4 m: el suelo cuenta), a la más baja que quepa; el último tramo baja
        # suave hasta la puerta (≤ 0,75 m en Lk metros). El tramo nivelado se comprueba hasta la vertical de la
        # puerta; la bajada final, no (A* también exime la celda de la meta)
        for dz in (0.0, 0.25, 0.5, 0.75):
            q = np.array([goal[0] - d[0] / n * Lk, goal[1] - d[1] / n * Lk, goal[2] + dz])
            over = np.array([goal[0] - d[0] / n * min(1.0, 0.3 * Lk), goal[1] - d[1] / n * min(1.0, 0.3 * Lk),
                             goal[2] + dz])
            if segment_clear(world, prev, q, need) and segment_clear(world, q, over, need):
                return pts[:k + 1] + [q, goal]
    return pts


def string_pull(world: World, pts: List[np.ndarray], need: float) -> List[np.ndarray]:
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not segment_clear(world, pts[i], pts[j], need):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def chaikin(world: World, pts: List[np.ndarray], need: float, iters: int = 3) -> List[np.ndarray]:
    for _ in range(iters):
        new = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            q, r = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
            new += [q, r]
        new.append(pts[-1])
        if all(segment_clear(world, x, y, need, 0.5) for x, y in zip(new, new[1:])):
            pts = new
        else:
            break
    return pts


def resample(pts: List[np.ndarray], ds: float) -> np.ndarray:
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = np.linalg.norm(b - a)
        if seg < 1e-9:
            continue
        d = ds - carry
        while d <= seg:
            out.append(a + (b - a) * (d / seg))
            d += ds
        carry = seg - (d - ds)
    if np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1])
    return np.array(out)


class Trajectory:
    """Referencia muestreada en el tiempo: posiciones, velocidades y aceleraciones."""

    def __init__(self, P: np.ndarray, speed: np.ndarray):
        self.P = P
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        vm = 0.5 * (speed[:-1] + speed[1:])
        dt = seg / np.maximum(vm, 0.05)
        self.t = np.concatenate([[0.0], np.cumsum(dt)])
        self.speed = speed
        tang = np.zeros_like(P)
        tang[:-1] = np.diff(P, axis=0) / np.maximum(seg[:, None], 1e-9)
        tang[-1] = tang[-2]
        self.V = tang * speed[:, None]
        self.A = np.zeros_like(P)
        self.A[1:-1] = (self.V[2:] - self.V[:-2]) / np.maximum((self.t[2:] - self.t[:-2])[:, None], 1e-3)
        self.duration = float(self.t[-1])
        self.length = float(self.s[-1])

    def sample(self, t: float):
        if t >= self.duration:
            return self.P[-1].copy(), np.zeros(3), np.zeros(3)
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        i = min(max(i, 0), len(self.t) - 2)
        f = (t - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)
        lerp = lambda X: X[i] + (X[i + 1] - X[i]) * f
        return lerp(self.P), lerp(self.V), lerp(self.A)

    def to_dict(self, every: int = 2):
        return {"points": self.P[::every].round(2).tolist(), "speed": self.speed[::every].round(2).tolist(),
                "duration": self.duration, "length": self.length}


# PERFIL DE VELOCIDAD CON ARRASTRE (parametrización temporal óptima, TOPP, con la dinámica longitudinal del dron): la
# aceleración del planificador es la del empuje horizontal; el arrastre (k·v²) la reduce al acelerar y se suma al
# frenar. Sin él, el perfil suponía la misma aceleración a 10 m/s que parado (en el X650, a 10 m/s el arrastre se
# come 6,8 de 8,8 m/s²) y frenaba como si el aire no ayudara.
# PROBADO Y DESACTIVADO (clásico en el X650, 60 semillas por caso): más fiel (el dron va por detrás de la referencia
# un 36 % del tiempo en vez de un 40 %), pero un 1-4 % más lento salvo con el objetivo rápido (−2 %): pedir algo más
# de lo que da hace que vaya siempre al máximo. DRON_DRAG_PROFILE=1: con él
DRAG_PROFILE = float(os.environ.get("DRON_DRAG_PROFILE", 0.0))


def velocity_profile(P: np.ndarray, prof: Profile, v_cruise: float, end_speed: float = 0.0,
                     start_speed: float = 0.0) -> np.ndarray:
    # (vectorizado: con un bucle de numpy por punto, el perfil se llevaba ~18 ms de cada replanificación de ~55)
    n = len(P)
    vlim = np.full(n, float(v_cruise))
    a_lat = prof.acc_hor
    if n > 2:  # curvatura por el círculo que pasa por 3 puntos
        a, b, c = P[:-2], P[1:-1], P[2:]
        ab = np.linalg.norm(b - a, axis=1)
        bc = np.linalg.norm(c - b, axis=1)
        ca = np.linalg.norm(a - c, axis=1)
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        k = 2 * area2 / np.maximum(ab * bc * ca, 1e-9)
        with np.errstate(divide="ignore"):
            vlim[1:-1] = np.where(k > 1e-6, np.minimum(vlim[1:-1], np.sqrt(a_lat / np.maximum(k, 1e-12))), vlim[1:-1])
    seg = P[1:] - P[:-1]
    L = np.linalg.norm(seg, axis=1)
    sz = np.where(L > 1e-9, seg[:, 2] / np.maximum(L, 1e-12), 0.0)       # límites de subida y bajada
    with np.errstate(divide="ignore"):
        lim = np.where(sz > 1e-3, prof.v_up / np.maximum(sz, 1e-12),
                       np.where(sz < -1e-3, prof.v_down / np.maximum(-sz, 1e-12), np.inf))
    vlim[:-1] = np.minimum(vlim[:-1], lim)
    vlim[0] = min(vlim[0], start_speed)
    vlim[-1] = min(vlim[-1], end_speed)
    v = vlim.tolist()
    ds = L.tolist()
    two_a = 2 * prof.acc_hor
    # (TOPP con arrastre, ver DRAG_PROFILE: acelerando, el arrastre resta; frenando, ayuda)
    kd = prof.drag * DRAG_PROFILE

    def passes(v):
        for i in range(1, n):  # acelerar como mucho a_max
            m = math.sqrt(max(v[i - 1] ** 2 + two_a * ds[i - 1] - 2 * kd * v[i - 1] ** 2 * ds[i - 1], v[i - 1] ** 2))
            if v[i] > m:
                v[i] = m
        for i in range(n - 2, -1, -1):  # frenar a tiempo
            m = math.sqrt(v[i + 1] ** 2 + two_a * ds[i] + 2 * kd * v[i + 1] ** 2 * ds[i])
            if v[i] > m:
                v[i] = m
        return v
    v = np.array(passes(v))
    w = max(1, int(round(prof.acc_hor / prof.jerk / 0.05)))  # suavizado ≈ límite de tirón
    for _ in range(2):
        v = np.minimum(np.convolve(np.pad(v, w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), "valid"), v)
    v[0], v[-1] = min(v[0], start_speed), min(v[-1], end_speed)
    v = np.array(passes(v.tolist()))   # el suavizado puede dejar saltos junto a los extremos: se repasan los límites
    return np.maximum(v, 0.0)


SMOOTHER = os.environ.get("DRON_SMOOTHER", "chaikin")   # "bspline" (EGO-Planner) o "chaikin"


def straight(prof: Profile, start, goal, v_cruise: float, end_speed: float = 0.0, start_speed: float = 0.0):
    """Recta de start a goal con su perfil de velocidad (el piloto reactivo: no planifica, la sigue esquivando)."""
    a, b = np.asarray(start, float), np.asarray(goal, float)
    P = resample([a, (a + b) / 2, b], DS)
    if len(P) < 3:
        P = np.array([a, a + (b - a) * 0.5, b]) if np.linalg.norm(b - a) > 1e-6 else np.array([a, a, a + 1e-3])
    tr = Trajectory(P, velocity_profile(P, prof, v_cruise, end_speed, start_speed))
    tr.v_cruise, tr.end_speed = v_cruise, end_speed
    return tr


def free_goal_near(grid, goal, need: float, radius: float = 2.5):
    """Punto más cercano a `goal` (a menos de `radius` m, a la misma altura o hasta 1,5 m más arriba) con holgura
    ≥ need + 5 cm, o None. Solo si la meta NO tiene esa holgura (si la tiene, devuelve None: no hay nada que mover)."""
    g = np.asarray(goal, float)
    if grid_clearance(grid, g[None])[0] >= need:
        return None
    r = np.arange(-radius, radius + 1e-9, 0.25)
    X, Y, Z = np.meshgrid(r, r, (0.0, 0.5, 1.0, 1.5), indexing="ij")
    D = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    D = D[np.hypot(D[:, 0], D[:, 1]) <= radius]
    P = g + D
    ok = grid_clearance(grid, P) >= need + 0.05
    if not ok.any():
        return None
    return P[ok][np.argmin(np.linalg.norm(D[ok] * np.array([1.0, 1.0, 2.0]), axis=1))]   # mejor no subir


def grid_clearance(grid, P) -> np.ndarray:
    """Holgura interpolada (trilineal) en muchos puntos, para la rejilla del mundo o la aprendida (misma interfaz)."""
    from .mapping import LearnedGrid
    return LearnedGrid.clearance_many(grid, P)


def _bspline_eval(Q: np.ndarray, per_seg: int = 8) -> np.ndarray:
    """Puntos de una B-spline cúbica uniforme "sujeta" (pasa por el primer y el último punto de control)."""
    Qp = np.vstack([Q[:1], Q[:1], Q, Q[-1:], Q[-1:]])
    u = np.linspace(0, 1, per_seg, endpoint=False)
    B = np.stack([(1 - u) ** 3, 3 * u ** 3 - 6 * u ** 2 + 4, -3 * u ** 3 + 3 * u ** 2 + 3 * u + 1, u ** 3], axis=1) / 6
    out = [B @ Qp[i:i + 4] for i in range(len(Qp) - 3)]
    return np.vstack(out + [Q[-1:]])


def bspline_smooth(grid, pts, need: float, iters: int = 40):
    """Optimiza los puntos de control de una B-spline sobre el camino (suavidad + holgura con el ESDF)."""
    Q = resample(pts, 1.0).copy()
    n = len(Q)
    if n < 6:
        return None
    free = np.ones(n, bool)
    free[[0, 1, n - 2, n - 1]] = False       # extremos fijos (salida y meta)
    want = need + 0.4
    h = 0.25
    for _ in range(iters):
        A = Q[:-2] - 2 * Q[1:-1] + Q[2:]      # "aceleración" de los puntos de control
        g = np.zeros_like(Q)
        g[:-2] += 2 * A
        g[1:-1] -= 4 * A
        g[2:] += 2 * A
        c = grid_clearance(grid, Q)
        viol = np.maximum(want - c, 0.0)
        k = np.flatnonzero((viol > 0) & free)
        if len(k):
            grad = np.stack([(grid_clearance(grid, Q[k] + e) - grid_clearance(grid, Q[k] - e)) / (2 * h)
                             for e in np.eye(3) * h], axis=1)
            g[k] += 10.0 * (-2 * viol[k, None] * grad)
        step = -0.05 * g
        lim = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, 0.3 / np.maximum(lim, 1e-9))   # como mucho 0,3 m por iteración
        Q[free] += step[free]
    return list(_bspline_eval(Q))


def plan(world, grid, prof: Profile, start, goal, v_cruise: Optional[float] = None,
         end_speed: float = 0.0, margin: float = MARGIN, start_speed: float = 0.0, prefix=None,
         min_margin: float = MARGIN, start_vel=None):
    """Trayectoria de start a goal (puntos 3D) o None si no hay camino. end_speed > 0: cruza el final sin frenar.

    `world` es el espacio contra el que se comprueban los segmentos (el mundo real o el mapa aprendido) y `grid`
    la rejilla de holgura para A*.

    `margin` es la holgura de seguridad deseada; si con ella no hay camino, se prueba con márgenes menores
    (hasta `min_margin`, MARGIN de fábrica o el ajustado del dron). Así, con un GPS malo, el dron deja más espacio
    a los obstáculos cuando se puede.
    """
    # la meta nunca fuera de la arena (p. ej. un objetivo que huye extrapolado por el filtro más allá del borde:
    # el dron lo perseguía hasta rozar la geovalla)
    goal = np.array([min(max(goal[0], 2.0), LENGTH - 2.0), min(max(goal[1], 2.0), WIDTH - 2.0), goal[2]], float)
    if prefix is not None:
        start = prefix[-1]
    cells, need = None, prof.radius + min_margin
    for m in sorted({margin, (margin + min_margin) / 2, min_margin}, reverse=True):
        need = prof.radius + m
        cells = astar(grid, start, goal, need)
        if cells is not None:
            break
    if cells is None:
        # la meta ha quedado dentro o pegada a un obstáculo del mapa (en el almacén, la deriva de la odometría
        # visual desplaza el mapa ~0,5-1 m respecto a la plataforma, y en un pasillo estrecho eso la mete en la
        # estantería): se acerca al punto libre más cercano, como hacen Fast-Planner y EGO-Planner. La cámara del
        # aterrizaje de precisión corrige después. Sin esto, 3 de 40 vuelos se quedaban esperando junto a la meta
        need = prof.radius + min_margin
        alt = free_goal_near(grid, goal, need)
        if alt is None:
            return None
        goal = alt
        cells = astar(grid, start, goal, need)
        if cells is None:
            return None
    pts = [np.array(start, float)] + [grid.center(c) for c in cells[1:-1]] + [np.array(goal, float)]
    pts = string_pull(world, pts, need)
    if end_speed > 0 and APPROACH > 0:
        pts = level_approach(world, pts, need)
    if prefix is not None:   # cosido: el suavizado ve la dirección con la que llega el dron
        pts = [np.asarray(prefix[-2], float)] + pts
    smooth = bspline_smooth(grid, pts, need) if SMOOTHER == "bspline" else None
    if smooth is not None and all(segment_clear(world, a, b, need, 0.5) for a, b in zip(smooth, smooth[1:])):
        pts = smooth
    else:
        pts = chaikin(world, pts, need)
    if prefix is not None:
        pts = [np.asarray(q, float) for q in prefix[:-2]] + list(pts)
    P = resample(pts, DS)
    if len(P) < 3:
        P = resample([np.array(start, float), (np.array(start) + np.array(goal)) / 2, np.array(goal, float)], DS)
    if start_vel is not None and prefix is None:
        # la velocidad inicial es la componente de la actual EN LA DIRECCIÓN DEL CAMINO NUEVO: si hay que dar media
        # vuelta, empieza frenando (antes arrancaba a la misma rapidez pero en sentido contrario y se pasaba de largo)
        k = min(4, len(P) - 1)
        d0 = (P[k] - P[0]) / max(float(np.linalg.norm(P[k] - P[0])), 1e-9)
        start_speed = float(np.clip(np.dot(start_vel, d0), 0.0, start_speed))
    speed = velocity_profile(P, prof, v_cruise or prof.v_cruise, end_speed, start_speed)
    return Trajectory(P, speed)
