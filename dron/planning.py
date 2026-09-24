"""Planificación: camino global + trayectoria con perfil de velocidad.

1. Rejilla 3D de 1 m con el campo de distancias al obstáculo más cercano (ESDF, como Voxblox/FIESTA): cada celda
   sabe cuánta holgura tiene frente a obstáculos, bordes y TERRENO. Las celdas con menos de radio + margen de
   seguridad quedan bloqueadas. La rejilla solo llega hasta 8 m por encima de lo más alto del mapa.
2. A* ponderado (heurística × 1,3, "weighted A*": algo menos óptimo y mucho más rápido) en 26 vecinos. El coste de
   paso crece cerca de los obstáculos, así el camino prefiere pasillos anchos (como el término de "distancia a
   obstáculos" de EGO-Planner).
3. Estirado de cuerda (string pulling / line-of-sight, lo mismo que hace Theta*): se quitan los puntos
   intermedios mientras el segmento directo siga libre. Quedan tramos rectos largos.
4. Suavizado de esquinas (Chaikin, 3 iteraciones, comprobando colisión) y remuestreo cada 0,25 m.
5. Perfil de velocidad óptimo en tiempo (el enfoque TOPP de los planificadores de trayectorias):
     * límite por curvatura: v ≤ √(a_lat / κ) (en curvas cerradas hay que frenar);
     * límite vertical: la componente vertical no supera la subida/bajada máxima del dron;
     * pasada hacia delante (acelerar como mucho a_max) y hacia atrás (frenar a tiempo): así va rápido en las
       rectas y frena justo antes de las curvas y del final (en modo carrera NO frena al final: cruza la meta);
     * suavizado del perfil para aproximar el límite de tirón (jerk);
     * al REPLANIFICAR en vuelo, la trayectoria nueva arranca a la velocidad actual (si no, frenaría cada vez).
   El resultado es una referencia p(t), v(t), a(t) que el control sigue con prealimentación.
"""
import heapq
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


def astar(grid: ClearanceGrid, start, goal, need: float) -> Optional[List[Tuple[int, int, int]]]:
    free = grid.clr > need
    s, g = grid.cell(start), grid.cell(goal)
    for c in (s, g):
        if not free[c]:
            free[c] = True  # salida/meta: se permite aunque rocen el margen (el dron está posado o bajando)
    gx, gy, gz = g
    moves = [(dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    near = 1.0 + 1.5 * np.clip(2.5 - grid.clr, 0, 2.5) / 2.5  # más caro cerca de obstáculos
    best = {s: 0.0}
    parent = {s: None}
    h = lambda c: 1.3 * math.sqrt((c[0] - gx) ** 2 + (c[1] - gy) ** 2 + (c[2] - gz) ** 2)  # A* ponderado
    heap = [(h(s), 0.0, s)]
    nx, ny, nz = grid.shape
    while heap:
        _, cost, c = heapq.heappop(heap)
        if c == g:
            path = []
            while c is not None:
                path.append(c)
                c = parent[c]
            return path[::-1]
        if cost > best.get(c, math.inf):
            continue
        for dx, dy, dz, length in moves:
            n = (c[0] + dx, c[1] + dy, c[2] + dz)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz) or not free[n]:
                continue
            nc = cost + length * near[n]
            if nc < best.get(n, math.inf):
                best[n] = nc
                parent[n] = c
                heapq.heappush(heap, (nc + h(n), nc, n))
    return None


def segment_clear(world: World, a, b, need: float, step: float = 0.3) -> bool:
    n = max(1, int(np.linalg.norm(b - a) / step))
    for k in range(n + 1):
        p = a + (b - a) * (k / n)
        if world.clearance(tuple(p), need + 0.5, ground=True) < need:
            return False
    return True


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


def velocity_profile(P: np.ndarray, prof: Profile, v_cruise: float, end_speed: float = 0.0,
                     start_speed: float = 0.0) -> np.ndarray:
    n = len(P)
    vlim = np.full(n, v_cruise)
    a_lat = prof.acc_hor
    for i in range(1, n - 1):  # curvatura por el círculo que pasa por 3 puntos
        a, b, c = P[i - 1], P[i], P[i + 1]
        ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        k = 2 * area2 / max(ab * bc * ca, 1e-9)
        if k > 1e-6:
            vlim[i] = min(vlim[i], math.sqrt(a_lat / k))
    seg = P[1:] - P[:-1]
    for i, d in enumerate(seg):  # límites de subida y bajada
        L = np.linalg.norm(d)
        if L > 1e-9:
            sz = d[2] / L
            if sz > 1e-3:
                vlim[i] = min(vlim[i], prof.v_up / sz)
            elif sz < -1e-3:
                vlim[i] = min(vlim[i], prof.v_down / -sz)
    vlim[0] = min(vlim[0], start_speed)
    vlim[-1] = min(vlim[-1], end_speed)
    v = vlim.copy()
    ds = np.linalg.norm(seg, axis=1)
    for i in range(1, n):  # acelerar como mucho a_max
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * prof.acc_hor * ds[i - 1]))
    for i in range(n - 2, -1, -1):  # frenar a tiempo
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * prof.acc_hor * ds[i]))
    w = max(1, int(round(prof.acc_hor / prof.jerk / 0.05)))  # suavizado ≈ límite de tirón
    for _ in range(2):
        v = np.minimum(np.convolve(np.pad(v, w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), "valid"), v)
    v[0], v[-1] = min(v[0], start_speed), min(v[-1], end_speed)
    for i in range(1, n):  # el suavizado puede dejar saltos junto a los extremos: se repasan los límites
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * prof.acc_hor * ds[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * prof.acc_hor * ds[i]))
    return np.maximum(v, 0.0)


def plan(world: World, grid: ClearanceGrid, prof: Profile, start, goal, v_cruise: Optional[float] = None,
         end_speed: float = 0.0, margin: float = MARGIN, start_speed: float = 0.0):
    """Trayectoria de start a goal (puntos 3D) o None si no hay camino. end_speed > 0: cruza el final sin frenar.

    `margin` es la holgura de seguridad deseada; si con ella no hay camino, se prueba con márgenes menores
    (hasta el mínimo MARGIN). Así, con un GPS malo, el dron deja más espacio a los obstáculos cuando se puede.
    """
    cells, need = None, prof.radius + MARGIN
    for m in sorted({margin, (margin + MARGIN) / 2, MARGIN}, reverse=True):
        need = prof.radius + m
        cells = astar(grid, start, goal, need)
        if cells is not None:
            break
    if cells is None:
        return None
    pts = [np.array(start, float)] + [grid.center(c) for c in cells[1:-1]] + [np.array(goal, float)]
    pts = string_pull(world, pts, need)
    pts = chaikin(world, pts, need)
    P = resample(pts, DS)
    if len(P) < 3:
        P = resample([np.array(start, float), (np.array(start) + np.array(goal)) / 2, np.array(goal, float)], DS)
    speed = velocity_profile(P, prof, v_cruise or prof.v_cruise, end_speed, start_speed)
    return Trajectory(P, speed)
