"""Mapa que CONSTRUYE el dron con sus sensores (fase 2): ya no conoce el mundo.

1. Mapa de ocupación 3D en vóxeles de 1 m con log-odds, como OctoMap (Hornung et al., 2013). Cada rayo (40
   telémetros a 20 Hz + cámara de profundidad a 10 Hz) se inserta desde la posición ESTIMADA del dron (el error
   del GPS emborrona el mapa, como en un dron real):
     * las celdas que atraviesa el rayo reciben "libre" (log-odds −0,4) y la del final "ocupada" (+0,85);
     * en un mismo barrido, si una celda es a la vez libre y ocupada, gana ocupada (como OctoMap);
     * un rayo sin eco (distancia = alcance máximo) solo marca libre; un píxel sin dato no marca nada;
     * los valores se limitan a [−2, 3,5] para que el mapa pueda cambiar de opinión (lectura falsa o ruido).
   Parámetros por defecto de octomap_server: probabilidad de acierto 0,7 y de fallo 0,4, límites 0,12 y 0,97,
   umbral de ocupación 0,5 (log-odds 0).
   El recorrido del rayo se muestrea cada media celda (aproximación del recorrido exacto de Amanatides-Woo).
2. Suposición 2,5D (la de los mapas de elevación en exteriores): lo que queda DEBAJO de una celda ocupada se
   considera sólido. Terreno, edificios y árboles no tienen voladizos; sin esto, el planificador podría "colarse"
   por dentro de una montaña de la que solo ha visto la piel.
3. Campo de distancias (ESDF, como Voxblox/FIESTA) con la transformada de distancia euclídea exacta de scipy. Se
   recalcula entero solo cuando alguna celda cambia de estado: con 90 × 60 × 35 celdas tarda ~15 ms, así que la
   versión incremental de FIESTA (pensada para mapas mucho mayores) no hace falta aquí.
   Holgura de una celda = distancia a la celda ocupada más cercana − media celda (el obstáculo puede estar en el
   borde de su celda).
4. LO DESCONOCIDO ES LIBRE para el planificador (optimista, como Fast-Planner y EGO-Planner): así puede trazar
   un camino hacia una meta que todavía no ve. La seguridad viene de (a) no volar más rápido de lo que dejan
   frenar los sensores, (b) comprobar la trayectoria contra el mapa cada vez que cambia y replanificar si choca, y
   (c) Collision Prevention con los telémetros.
"""
import math
from typing import Tuple

import numpy as np

from .world import CEILING, LENGTH, WIDTH

CELL = 1.0
L_HIT, L_MISS = 0.85, -0.4          # log(0,7/0,3), log(0,4/0,6)
L_MIN, L_MAX = -2.0, 3.5            # log-odds de 0,12 y 0,97
FAR = 99.0                          # holgura "infinita" (no hay nada ocupado)


class OccupancyMap:
    def __init__(self):
        self.shape = (int(LENGTH / CELL), int(WIDTH / CELL), int(CEILING / CELL))
        self.L = np.zeros(self.shape, np.float32)          # log-odds
        self.seen = np.zeros(self.shape, bool)             # ¿observada alguna vez?
        self.hit_z = np.full(self.shape, -np.inf, np.float32)  # altura máxima de los ecos dentro de cada celda
        self.occ = np.zeros(self.shape, bool)
        self.version = 0                                   # cambia cuando alguna celda cambia de estado
        self._grid, self._grid_version = None, -1
        self.rays_in = 0

    # ------------------------------------------------------------------ inserción
    def _index(self, P):
        idx = np.floor(P / CELL).astype(np.int64)
        ok = (idx >= 0).all(axis=1) & (idx[:, 0] < self.shape[0]) & (idx[:, 1] < self.shape[1]) & \
             (idx[:, 2] < self.shape[2])
        return idx, ok

    def insert(self, origin, dirs, dists, rmax: float):
        """Inserta un barrido: rayos desde `origin` (posición estimada) con sus distancias medidas."""
        o = np.asarray(origin, float)
        D = np.asarray(dirs, float)
        d = np.asarray(dists, float)
        valid = np.isfinite(d)
        D, d = D[valid], d[valid]
        if not len(d):
            return
        self.rays_in += len(d)
        hit = d < rmax - 1e-3
        # libres: puntos a lo largo del rayo cada media celda, sin llegar al final
        step = 0.5 * CELL
        T = np.arange(0.0, rmax, step)[None, :]
        inside = T < (d[:, None] - 0.3 * CELL)
        P = o + (D[:, None, :] * T[..., None])[inside]
        fi, ok = self._index(P)
        nx, ny, nz = self.shape
        free = np.unique((fi[ok, 0] * ny + fi[ok, 1]) * nz + fi[ok, 2])
        # ocupadas: el final de los rayos con eco
        H = o + D[hit] * d[hit, None]
        H[:, 2] = np.maximum(H[:, 2], 0.0)  # el ruido puede dejar un eco del suelo llano un poco por debajo de 0
        hi, ok = self._index(H)
        Hk = H[ok]
        occ_flat = (hi[ok, 0] * ny + hi[ok, 1]) * nz + hi[ok, 2]
        occ = np.unique(occ_flat)
        free = np.setdiff1d(free, occ, assume_unique=True)
        L, seen, hz = self.L.reshape(-1), self.seen.reshape(-1), self.hit_z.reshape(-1)
        L[free] = np.maximum(L[free] + L_MISS, L_MIN)
        L[occ] = np.minimum(L[occ] + L_HIT, L_MAX)
        seen[free] = seen[occ] = True
        np.maximum.at(hz, occ_flat, Hk[:, 2].astype(np.float32))
        touched = np.concatenate([free, occ])
        now = L[touched] > 0.0
        occ_all = self.occ.reshape(-1)
        changed = now != occ_all[touched]
        if changed.any():
            occ_all[touched] = now
            self.version += 1

    # ------------------------------------------------------------------ consultas
    def solid(self) -> np.ndarray:
        """Ocupado + todo lo que hay debajo de algo ocupado (suposición 2,5D)."""
        return np.flip(np.logical_or.accumulate(np.flip(self.occ, axis=2), axis=2), axis=2)

    def grid(self) -> "LearnedGrid":
        """Rejilla de holgura (ESDF) del mapa actual, recalculada solo si el mapa ha cambiado."""
        if self._grid_version != self.version:
            self._grid = LearnedGrid(self.solid())
            self._grid_version = self.version
        return self._grid

    def surface(self, x: float, y: float, below: float) -> float:
        """Altura de lo más alto que ha visto el dron en la columna (x, y) por debajo de `below`, o -inf si no ha
        visto nada. Es la superficie estimada que sustituye a `world.surface` (usa la altura real de los ecos, no
        el borde de la celda)."""
        i, j = int(x / CELL), int(y / CELL)
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return -math.inf
        col = self.occ[i, j]
        k = min(int(below / CELL), self.shape[2] - 1)
        ks = np.nonzero(col[:k + 1])[0]
        if not len(ks):
            return -math.inf
        return float(min(self.hit_z[i, j, ks[-1]], below))

    def explored_fraction(self) -> float:
        return float(self.seen.any(axis=2).mean())

    def to_dict(self):
        """Celdas ocupadas (índices planos) y columnas exploradas, para dibujar el mapa en la interfaz."""
        nx, ny, nz = self.shape
        return {"cell": CELL, "shape": [nx, ny, nz], "occ": np.flatnonzero(self.occ).tolist(),
                "seen": "".join("1" if v else "0" for v in self.seen.any(axis=2).ravel())}


_BOUNDS = {}


def _bounds(shape):
    """Distancia a los bordes de la arena (geovalla conocida) y al nivel cero: no cambia, se calcula una vez."""
    if shape not in _BOUNDS:
        xs, ys, zs = ((np.arange(n) + 0.5) * CELL for n in shape)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        _BOUNDS[shape] = np.minimum.reduce([X, LENGTH - X, Y, WIDTH - Y, Z])
    return _BOUNDS[shape]


class LearnedGrid:
    """Misma interfaz que `planning.ClearanceGrid` (shape, clr, cell, center) más `clearance(p)` por interpolación
    trilineal del ESDF, que usa el planificador para comprobar segmentos (como Fast-Planner / EGO-Planner)."""

    def __init__(self, solid: np.ndarray):
        from scipy.ndimage import distance_transform_edt
        self.shape = solid.shape
        if solid.any():
            edt = distance_transform_edt(~solid) * CELL - 0.5 * CELL
            edt[solid] = -0.5 * CELL
        else:
            edt = np.full(solid.shape, FAR)
        self.clr = np.minimum(edt, _bounds(self.shape))

    def cell(self, p) -> Tuple[int, int, int]:
        return tuple(min(max(int(v / CELL), 0), n - 1) for v, n in zip(p, self.shape))

    @staticmethod
    def center(c) -> np.ndarray:
        return (np.array(c) + 0.5) * CELL

    def clearance_many(self, P: np.ndarray) -> np.ndarray:
        """Holgura interpolada (trilineal) en muchos puntos a la vez."""
        P = np.atleast_2d(np.asarray(P, float))
        g = P / CELL - 0.5
        hi = np.array(self.shape) - 1.001
        g = np.clip(g, 0.0, hi)
        i = g.astype(int)
        f = g - i
        c = self.clr
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    out = out + wx * wy * wz * c[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        # fuera de la arena o por debajo de cero: sin holgura
        outside = (P[:, 0] < 0) | (P[:, 0] > LENGTH) | (P[:, 1] < 0) | (P[:, 1] > WIDTH) | (P[:, 2] < 0)
        return np.where(outside, -1.0, out)

    def clearance(self, p, limit: float = FAR, ground: bool = True) -> float:
        """Misma firma que `World.clearance`, para que el planificador funcione igual con los dos."""
        return float(min(self.clearance_many(p)[0], limit))
