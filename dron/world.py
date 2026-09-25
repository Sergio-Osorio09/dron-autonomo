"""El mundo 3D: terreno con relieve, obstáculos apoyados en él, meta, rayos y colisiones.

Coordenadas: x, y en el plano y z hacia arriba, en metros. La arena mide LENGTH × WIDTH y el techo es CEILING (altura
absoluta). El suelo ya no es plano: `terrain.height(x, y)` da la altura del terreno.

Terrenos (`TERRAINS`):
  * plano       sin relieve;
  * colinas     suma de lomas suaves (2-5 m);
  * montañoso   varias montañas de 8-15 m y lomas;
  * precipicios mesetas de 5-10 m con paredes casi verticales (acantilados) y un par de lomas.
Obstáculos: árboles/arbustos (cilindros) y edificios (cajas; un tercio bajos), con su base en el terreno.
Densidad (`DENSITIES`): multiplica el número de obstáculos ("máxima": un bosque cerrado, casi sin huecos).

Almacén (nivel `almacen`, interior): suelo llano, techo a 10 m, paredes, filas de estanterías de 5-7 m con pasillos
de 3,6 m (densidad normal) a 2,4 m (máxima), pasillos transversales, pilares hasta el techo y palés en la zona de
carga. El dron despega en la zona de carga y la meta está al fondo del almacén, en un pasillo o sobre una
estantería. Sin viento ni lluvia (la simulación los anula). El techo es un obstáculo (`kind="techo"`), no una
superficie donde posarse.

Metas (`GOAL_KINDS`): la plataforma puede estar en el `suelo`, en la `azotea` de un edificio, en la `cima` del
terreno, o ser un punto `aire` (solo para el modo carrera). `azar` elige entre las posibles.
En modo carrera, la plataforma del suelo puede ir sobre un vehículo en movimiento (ver target.py).
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

LENGTH, WIDTH, CEILING = 90.0, 60.0, 35.0
DRONE_RADIUS = 0.35
RAY_RANGE = 12.0
LEVELS = ("bosque", "ciudad", "mixto")          # exteriores (los que usan las evaluaciones de siempre)
ALL_LEVELS = LEVELS + ("almacen",)
TERRAINS = ("plano", "colinas", "montañoso", "precipicios")
DENSITIES = {"baja": 0.5, "normal": 1.0, "alta": 1.8, "extrema": 2.8, "máxima": 4.0}
WAREHOUSE_CEILING = 10.0
AISLES = {"baja": 4.2, "normal": 3.6, "alta": 3.0, "extrema": 2.7, "máxima": 2.4}   # anchura de los pasillos (m)
GOAL_KINDS = ("suelo", "azotea", "cima", "aire", "azar")
PAD_HEIGHT = 0.05
ROVER_HEIGHT = 0.6   # la plataforma va sobre el vehículo
TERRAIN_RES = 0.5

Vec = Tuple[float, float, float]


# --------------------------------------------------------------------------- terreno

class Terrain:
    """Mapa de alturas en una rejilla de 0,5 m con interpolación bilineal (escalar y vectorizada)."""

    def __init__(self, kind: str, rng: random.Random):
        self.kind = kind
        nx, ny = int(LENGTH / TERRAIN_RES) + 1, int(WIDTH / TERRAIN_RES) + 1
        xs, ys = np.linspace(0, LENGTH, nx), np.linspace(0, WIDTH, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        H = np.zeros_like(X)

        def bump(amp, sx, sy=None):
            cx, cy = rng.uniform(0, LENGTH), rng.uniform(0, WIDTH)
            sy = sy or sx
            return amp * np.exp(-(((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2) / 2)

        if kind == "colinas":
            for _ in range(rng.randint(7, 11)):
                H += bump(rng.uniform(2, 5), rng.uniform(6, 12), rng.uniform(6, 12))
        elif kind == "montañoso":
            for _ in range(rng.randint(3, 5)):
                H += bump(rng.uniform(8, 15), rng.uniform(7, 13), rng.uniform(7, 13))
            for _ in range(6):
                H += bump(rng.uniform(1, 3), rng.uniform(4, 8))
        elif kind == "precipicios":
            for _ in range(rng.randint(2, 3)):  # mesetas con paredes casi verticales
                cx, cy = rng.uniform(15, LENGTH - 15), rng.uniform(12, WIDTH - 12)
                ax, ay, ang = rng.uniform(7, 14), rng.uniform(5, 10), rng.uniform(0, math.pi)
                u = ((X - cx) * math.cos(ang) + (Y - cy) * math.sin(ang)) / ax
                v = (-(X - cx) * math.sin(ang) + (Y - cy) * math.cos(ang)) / ay
                r = np.sqrt(u ** 2 + v ** 2)
                H = np.maximum(H, rng.uniform(5, 10) / (1 + np.exp((r - 1) * 25)))  # borde de ~0,5 m
            for _ in range(3):
                H += bump(rng.uniform(1, 2.5), rng.uniform(5, 9))
        elif kind != "plano":
            raise ValueError("Terreno desconocido %r (usa %s)" % (kind, ", ".join(TERRAINS)))
        self.H = H
        self.nx, self.ny = nx, ny

    def height(self, x: float, y: float) -> float:
        gx = min(max(x / TERRAIN_RES, 0.0), self.nx - 1.001)
        gy = min(max(y / TERRAIN_RES, 0.0), self.ny - 1.001)
        i, j = int(gx), int(gy)
        fx, fy = gx - i, gy - j
        H = self.H
        return float((H[i, j] * (1 - fx) + H[i + 1, j] * fx) * (1 - fy) + (H[i, j + 1] * (1 - fx) + H[i + 1, j + 1] * fx) * fy)

    def heights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = np.clip(x / TERRAIN_RES, 0.0, self.nx - 1.001)
        gy = np.clip(y / TERRAIN_RES, 0.0, self.ny - 1.001)
        i, j = gx.astype(int), gy.astype(int)
        fx, fy = gx - i, gy - j
        H = self.H
        return (H[i, j] * (1 - fx) + H[i + 1, j] * fx) * (1 - fy) + (H[i, j + 1] * (1 - fx) + H[i + 1, j + 1] * fx) * fy

    def slope(self, x: float, y: float) -> float:
        dx = self.height(x + 0.5, y) - self.height(x - 0.5, y)
        dy = self.height(x, y + 0.5) - self.height(x, y - 0.5)
        return math.hypot(dx, dy)  # desnivel por metro

    def to_dict(self, step: int = 2):
        return {"res": TERRAIN_RES * step, "h": np.round(self.H[::step, ::step], 2).tolist()}


# --------------------------------------------------------------------------- obstáculos

@dataclass
class Cylinder:
    x: float
    y: float
    r: float
    h: float
    kind: str = "tree"
    base: float = 0.0

    @property
    def top(self) -> float:
        return self.base + self.h

    def distance(self, p: Vec) -> float:
        dxy = math.hypot(p[0] - self.x, p[1] - self.y) - self.r
        dz = max(self.base - p[2], p[2] - self.top, 0.0)
        if dz == 0:
            return dxy
        return math.hypot(max(dxy, 0.0), dz) if dxy > 0 else dz

    def ray(self, o: Vec, d: Vec, tmax: float) -> float:
        best = tmax
        ox, oy = o[0] - self.x, o[1] - self.y
        a = d[0] ** 2 + d[1] ** 2
        if a > 1e-12:
            b = ox * d[0] + oy * d[1]
            c = ox * ox + oy * oy - self.r ** 2
            disc = b * b - a * c
            if disc >= 0:
                t = (-b - math.sqrt(disc)) / a
                if 0 <= t < best and self.base <= o[2] + t * d[2] <= self.top:
                    best = t
        if abs(d[2]) > 1e-12:
            t = (self.top - o[2]) / d[2]
            if 0 <= t < best:
                px, py = ox + t * d[0], oy + t * d[1]
                if px * px + py * py <= self.r ** 2:
                    best = t
        return best

    def footprint(self):
        return self.x - self.r, self.x + self.r, self.y - self.r, self.y + self.r

    def to_dict(self):
        return {"type": "cylinder", "x": self.x, "y": self.y, "r": self.r, "h": self.h, "base": self.base, "kind": self.kind}


@dataclass
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    h: float
    kind: str = "building"
    base: float = 0.0

    @property
    def top(self) -> float:
        return self.base + self.h

    def distance(self, p: Vec) -> float:
        dx = max(self.x0 - p[0], 0.0, p[0] - self.x1)
        dy = max(self.y0 - p[1], 0.0, p[1] - self.y1)
        dz = max(self.base - p[2], 0.0, p[2] - self.top)
        out = math.sqrt(dx * dx + dy * dy + dz * dz)
        if out > 0:
            return out
        return -min(p[0] - self.x0, self.x1 - p[0], p[1] - self.y0, self.y1 - p[1], self.top - p[2])

    def ray(self, o: Vec, d: Vec, tmax: float) -> float:
        tmin, tmx = 0.0, tmax
        for lo, hi, oi, di in ((self.x0, self.x1, o[0], d[0]), (self.y0, self.y1, o[1], d[1]), (self.base, self.top, o[2], d[2])):
            if abs(di) < 1e-12:
                if oi < lo or oi > hi:
                    return tmax
                continue
            t1, t2 = (lo - oi) / di, (hi - oi) / di
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmx = max(tmin, t1), min(tmx, t2)
            if tmin > tmx:
                return tmax
        return tmin

    def footprint(self):
        return self.x0, self.x1, self.y0, self.y1

    def to_dict(self):
        return {"type": "box", "x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1, "h": self.h,
                "base": self.base, "kind": self.kind}


# --------------------------------------------------------------------------- mundo

class World:
    def __init__(self, terrain: Terrain, obstacles, start: Vec, goal: Vec, goal_kind: str, level: str, seed: int,
                 density: str = "normal", goal_support: Optional[str] = None):
        self.terrain = terrain
        self.obstacles = obstacles
        self.start, self.goal = start, goal
        self.goal_kind = goal_kind          # suelo | azotea | cima | aire
        self.goal_support = goal_support    # dónde está apoyada la plataforma (None si es un punto en el aire)
        self.level, self.seed, self.density = level, seed, density
        self.motion = None                  # TargetMotion si la meta va sobre un vehículo

    @property
    def moving(self) -> bool:
        return self.motion is not None and self.motion.kind != "fija"

    def goal_at(self, t: float = 0.0) -> Vec:
        """Plataforma en el instante t (se mueve si va sobre un vehículo)."""
        if not self.moving:
            return self.goal
        x, y = self.motion.position(t)
        return (x, y, self.terrain.height(x, y) + ROVER_HEIGHT)

    # --- superficie sobre la que se puede posar el dron: terreno o azotea
    def surface(self, x: float, y: float) -> float:
        s = self.terrain.height(x, y)
        for ob in self.obstacles:
            if isinstance(ob, Box) and ob.kind != "techo" and ob.x0 <= x <= ob.x1 and ob.y0 <= y <= ob.y1:
                s = max(s, ob.top)
        return s

    def clearance(self, p: Vec, limit: float = RAY_RANGE, ground: bool = True) -> float:
        """Distancia libre a lo más cercano: obstáculos, bordes y (si ground) el terreno (en vertical)."""
        best = min(limit, p[0], LENGTH - p[0], p[1], WIDTH - p[1])
        if ground:
            best = min(best, p[2] - self.terrain.height(p[0], p[1]))
        for ob in self.obstacles:
            x0, x1, y0, y1 = ob.footprint()
            if p[0] < x0 - best or p[0] > x1 + best or p[1] < y0 - best or p[1] > y1 + best:
                continue
            best = min(best, ob.distance(p))
        return best

    def collides(self, p: Vec, radius: float = DRONE_RADIUS) -> bool:
        """¿Una esfera toca un obstáculo o el borde? (el terreno y las azoteas los gestiona la física)."""
        return self.clearance(p, radius + 0.01, ground=False) < radius

    def ray(self, o: Vec, d: Vec, tmax: float = RAY_RANGE) -> float:
        best = self.ray_terrain(o, d, tmax)
        for ob in self.obstacles:
            best = ob.ray(o, d, best)
        return best

    def rays(self, o, dirs: np.ndarray, tmax: float, step: float = 0.3) -> np.ndarray:
        """Muchos rayos a la vez desde el mismo origen (misma geometría que `ray`, vectorizada con numpy).
        Es lo que necesita una cámara de profundidad: cientos de rayos por imagen."""
        o = np.asarray(o, float)
        D = np.asarray(dirs, float)
        best = self._rays_terrain(o, D, tmax, step)
        cyl = [ob for ob in self.obstacles if isinstance(ob, Cylinder)]
        box = [ob for ob in self.obstacles if isinstance(ob, Box)]
        with np.errstate(divide="ignore", invalid="ignore"):
            if cyl:
                cx, cy, r, b, top = (np.array([getattr(c, k) for c in cyl])[None, :] for k in ("x", "y", "r", "base", "top"))
                dx, dy, dz = D[:, 0:1], D[:, 1:2], D[:, 2:3]
                ox, oy = o[0] - cx, o[1] - cy
                a = dx ** 2 + dy ** 2
                bb = ox * dx + oy * dy
                disc = bb * bb - a * (ox * ox + oy * oy - r ** 2)
                t = (-bb - np.sqrt(np.maximum(disc, 0))) / np.where(a > 1e-12, a, np.nan)
                z = o[2] + t * dz
                side = np.where((a > 1e-12) & (disc >= 0) & (t >= 0) & (z >= b) & (z <= top), t, np.inf)
                tc = (top - o[2]) / np.where(np.abs(dz) > 1e-12, dz, np.nan)
                px, py = ox + tc * dx, oy + tc * dy
                cap = np.where((tc >= 0) & (px * px + py * py <= r ** 2), tc, np.inf)
                best = np.minimum(best, np.nanmin(np.minimum(side, cap), axis=1))
            if box:
                lo = np.array([[bx.x0, bx.y0, bx.base] for bx in box])[None]   # 1 × M × 3
                hi = np.array([[bx.x1, bx.y1, bx.top] for bx in box])[None]
                Dn = D[:, None, :]
                par = np.abs(Dn) < 1e-12
                t1, t2 = (lo - o) / np.where(par, np.nan, Dn), (hi - o) / np.where(par, np.nan, Dn)
                tn = np.where(par, -np.inf, np.minimum(t1, t2))
                tf = np.where(par, np.inf, np.maximum(t1, t2))
                outside = (par & ((o < lo) | (o > hi))).any(axis=2)
                tmin, tmx = np.maximum(tn.max(axis=2), 0.0), tf.min(axis=2)
                hit = ~outside & (tmin <= tmx)
                best = np.minimum(best, np.where(hit, tmin, np.inf).min(axis=1))
        return np.minimum(best, tmax)

    def _rays_terrain(self, o, D, tmax, step):
        n = int(tmax / step) + 1
        t = np.linspace(0, tmax, n)[None, :]
        X, Y, Z = o[0] + D[:, 0:1] * t, o[1] + D[:, 1:2] * t, o[2] + D[:, 2:3] * t
        gap = Z - self.terrain.heights(X, Y)          # > 0 por encima del terreno
        below = gap < 0
        k = np.argmax(below, axis=1)
        anyb = below.any(axis=1)
        out = np.full(len(D), float(tmax))
        rows = np.nonzero(anyb & (k > 0))[0]
        kk = k[rows]
        z0, z1 = gap[rows, kk - 1], gap[rows, kk]
        out[rows] = t[0, kk - 1] + (t[0, kk] - t[0, kk - 1]) * z0 / (z0 - z1)
        out[anyb & (k == 0)] = 0.0
        return out

    def ray_terrain(self, o: Vec, d: Vec, tmax: float, step: float = 0.3) -> float:
        n = int(tmax / step) + 1
        t = np.linspace(0, tmax, n)
        x, y, z = o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t
        below = z < self.terrain.heights(x, y)
        if not below.any():
            return tmax
        k = int(np.argmax(below))
        if k == 0:
            return 0.0
        z0 = z[k - 1] - self.terrain.height(x[k - 1], y[k - 1])   # por encima (>0)
        z1 = z[k] - self.terrain.height(x[k], y[k])               # por debajo (<0)
        return float(t[k - 1] + (t[k] - t[k - 1]) * z0 / (z0 - z1))

    def to_dict(self):
        return {"size": [LENGTH, WIDTH, CEILING], "start": self.start, "goal": self.goal, "goal_kind": self.goal_kind,
                "goal_support": self.goal_support, "level": self.level, "seed": self.seed, "density": self.density,
                "terrain": self.terrain.to_dict(), "terrain_kind": self.terrain.kind,
                "motion": self.motion.kind if self.motion else "fija",
                "obstacles": [o.to_dict() for o in self.obstacles]}


# --------------------------------------------------------------------------- generación

def _gap(a, b) -> float:
    ax0, ax1, ay0, ay1 = a.footprint()
    bx0, bx1, by0, by1 = b.footprint()
    return math.hypot(max(bx0 - ax1, ax0 - bx1, 0.0), max(by0 - ay1, ay0 - by1, 0.0))


def _flat_spot(terrain, rng, avoid=(), min_dist=0.0, far_from=None, tries=400, margin=5.0):
    """Punto del terreno poco inclinado, lejos de `far_from` y de los puntos de `avoid`."""
    for _ in range(tries):
        x, y = rng.uniform(margin, LENGTH - margin), rng.uniform(margin, WIDTH - margin)
        if terrain.slope(x, y) > 0.25:
            continue
        if far_from is not None and math.dist((x, y), far_from[:2]) < min_dist:
            continue
        if any(math.dist((x, y), a[:2]) < 4 for a in avoid):
            continue
        return x, y
    return None


def _make_warehouse(rng, seed, density, goal_kind, race, motion) -> World:
    """Almacén: estanterías en filas (a lo largo de x) con pasillos, pasillos transversales, pilares y palés."""
    ter = Terrain("plano", rng)
    H = WAREHOUSE_CEILING
    aisle, depth = AISLES[density], 1.2
    obstacles: List = []
    t = 0.3   # paredes
    obstacles += [Box(0, LENGTH, 0, t, H, "pared"), Box(0, LENGTH, WIDTH - t, WIDTH, H, "pared"),
                  Box(0, t, 0, WIDTH, H, "pared"), Box(LENGTH - t, LENGTH, 0, WIDTH, H, "pared")]
    x_start, x_end = 16.0, LENGTH - 4.0                  # zona de carga libre en x < 16
    cross = sorted(rng.sample([x for x in np.arange(x_start + 14, x_end - 8, 2.0)], 2))   # pasillos transversales
    y = 2.5
    racks = []
    while y + depth < WIDTH - 2.5:
        xs = [x_start] + [c for c in cross] + [x_end]
        for a, b in zip(xs[:-1], xs[1:]):
            x0, x1 = a + (0 if a == x_start else 2.0), b - (0 if b == x_end else 2.0)
            if x1 - x0 > 3:
                h = round(rng.uniform(5.0, 7.0), 2)
                racks.append(Box(round(x0, 2), round(x1, 2), round(y, 2), round(y + depth, 2), h, "estantería"))
        y += depth + aisle
    obstacles += racks
    for px in np.arange(x_start + 6, x_end, 12.0):       # pilares en los pasillos transversales y la zona de carga
        for py in (WIDTH / 2,):
            c = Cylinder(round(float(px), 2), py, 0.3, H, "pilar", 0.0)
            if all(q.distance((float(px), py, 1.0)) > 0.8 for q in racks):
                obstacles.append(c)
    start = (rng.uniform(4, 9), rng.uniform(8, WIDTH - 8), 0.0)
    for _ in range(int(6 * DENSITIES[density])):       # palés en la zona de carga
        x0, y0 = rng.uniform(3, x_start - 3), rng.uniform(2, WIDTH - 4)
        p = Box(round(x0, 2), round(x0 + 1.2, 2), round(y0, 2), round(y0 + 1.0, 2), round(rng.uniform(0.8, 1.8), 2), "palé")
        if p.distance((start[0], start[1], 0.5)) > 3.0:
            obstacles.append(p)
    # meta: al fondo, en un pasillo (suelo), sobre una estantería (azotea) o en el aire de un pasillo (carrera)
    aisles_y = [r.y1 + aisle / 2 for r in racks if r.y1 + aisle < WIDTH - 1.5]
    gy = rng.choice(aisles_y) if aisles_y else WIDTH / 2
    gx = rng.uniform(x_end - 14, x_end - 3)
    support = "terreno"
    if goal_kind == "azotea":
        far = [r for r in racks if r.x1 > x_end - 16 and r.x1 - r.x0 >= 4]
        if far:
            r = rng.choice(far)
            gx, gy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            support = "azotea"
        else:
            goal_kind = "suelo"
    elif goal_kind in ("cima", "azar"):
        goal_kind = "suelo"
    elif goal_kind == "aire" and not race:
        goal_kind = "suelo"
    obstacles.append(Box(0, LENGTH, 0, WIDTH, 0.3, "techo", H))
    world = World(ter, obstacles, start, (0, 0, 0), goal_kind, "almacen", seed, density)
    if goal_kind == "aire":
        gz, support = rng.uniform(2.5, 5.0), None
    else:
        gz = world.surface(gx, gy) + PAD_HEIGHT
    world.goal, world.goal_support = (gx, gy, gz), support
    if race and motion != "fija" and goal_kind == "suelo":
        from .target import make_motion
        world.motion = make_motion(world, (gx, gy), motion, seed)
        world.goal = world.goal_at(0.0)
        world.goal_support = "vehículo"
    return world


def make_world(level: str = "mixto", seed: Optional[int] = None, goal_kind: str = "suelo", terrain: str = "plano",
               density: str = "normal", race: bool = False, motion: str = "fija") -> World:
    """Genera un mundo reproducible. La salida y la meta quedan libres (radio 4 m)."""
    if level not in ALL_LEVELS:
        raise ValueError("Nivel desconocido %r" % level)
    if density not in DENSITIES:
        raise ValueError("Densidad desconocida %r" % density)
    if goal_kind not in GOAL_KINDS:
        raise ValueError("Tipo de meta desconocido %r" % goal_kind)
    seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    if level == "almacen":
        return _make_warehouse(rng, seed, density, goal_kind, race, motion)
    ter = Terrain(terrain, rng)
    kinds = ["suelo", "azotea", "cima"] + (["aire"] if race else [])
    if goal_kind == "azar":
        goal_kind = rng.choice(kinds)
    if goal_kind == "aire" and not race:
        goal_kind = "suelo"  # en el aire no se puede aterrizar
    if goal_kind == "azotea" and level == "bosque":
        goal_kind = "suelo"  # en el bosque no hay edificios
    if goal_kind == "cima" and terrain == "plano":
        goal_kind = "suelo"
    moving = race and motion != "fija"
    if moving:
        goal_kind = "suelo"  # el objetivo en movimiento es un vehículo que va por el suelo

    sx, sy = _flat_spot(ter, rng) or (5.0, 5.0)
    start = (sx, sy, ter.height(sx, sy))
    # meta en el suelo o en la cima: se decide antes de poner obstáculos para dejarla libre
    goal_xy = None
    if goal_kind == "cima":
        H = ter.H.copy()
        m = int(6 / TERRAIN_RES)
        H[:m], H[-m:], H[:, :m], H[:, -m:] = -1, -1, -1, -1
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
        goal_xy = (float(i * TERRAIN_RES), float(j * TERRAIN_RES))
        if math.dist(goal_xy, start[:2]) < 25:
            goal_kind = "suelo"
            goal_xy = None
    if goal_kind in ("suelo", "aire"):
        goal_xy = _flat_spot(ter, rng, far_from=start, min_dist=40) or _flat_spot(ter, rng, far_from=start, min_dist=25)
    keep_clear = [start] + ([(goal_xy[0], goal_xy[1], 0.0)] if goal_xy else [])

    f = DENSITIES[density]
    obstacles: List = []

    def ok(ob, gap):
        return all(ob.distance((k[0], k[1], ter.height(k[0], k[1]) + 0.5)) > 4.0 for k in keep_clear) and \
            all(_gap(ob, o) > gap for o in obstacles)

    def base_of(x0, x1, y0, y1):
        hs = [ter.height(x, y) for x in (x0, x1, (x0 + x1) / 2) for y in (y0, y1, (y0 + y1) / 2)]
        return min(hs), max(hs)

    n_trees = int({"bosque": rng.randint(60, 90), "ciudad": 0, "mixto": rng.randint(25, 40)}[level] * f)
    n_boxes = int({"bosque": 0, "ciudad": rng.randint(14, 20), "mixto": rng.randint(6, 10)}[level] * f)
    gap_box, gap_tree = 2.2 / math.sqrt(f) + 0.6, 1.6 / math.sqrt(f) + 0.5
    tries = 0
    while n_boxes and tries < 1500:
        tries += 1
        w, d = rng.uniform(3, 9), rng.uniform(3, 10)
        x0, y0 = rng.uniform(2, LENGTH - 2 - w), rng.uniform(1, WIDTH - 1 - d)
        lo, hi = base_of(x0, x0 + w, y0, y0 + d)
        if hi - lo > 3:  # no construir sobre un acantilado
            continue
        h = rng.choice([rng.uniform(2.5, 4.5), rng.uniform(7, 16), rng.uniform(7, 16)]) + (hi - lo)
        b = Box(round(x0, 2), round(x0 + w, 2), round(y0, 2), round(y0 + d, 2), round(h, 2),
                "low" if h - (hi - lo) < 5 else "building", round(lo, 2))
        if ok(b, gap_box):
            obstacles.append(b)
            n_boxes -= 1
    tries = 0
    while n_trees and tries < 5000:
        tries += 1
        r = rng.uniform(0.3, 0.9)
        h = rng.uniform(2.0, 3.5) if rng.random() < 0.2 else rng.uniform(6, 13)
        x, y = rng.uniform(2, LENGTH - 2), rng.uniform(1, WIDTH - 1)
        c = Cylinder(round(x, 2), round(y, 2), round(r, 2), round(h, 2), "bush" if h < 4 else "tree",
                     round(ter.height(x, y) - 0.3, 2))
        if ok(c, gap_tree):
            obstacles.append(c)
            n_trees -= 1

    support = "terreno"
    if goal_kind == "azotea":
        roofs = [b for b in obstacles if b.kind == "building" and b.x1 - b.x0 >= 3.5 and b.y1 - b.y0 >= 3.5
                 and math.dist(((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2), start[:2]) > 25]
        if roofs:
            b = rng.choice(roofs)
            goal_xy = ((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)
            support = "azotea"
        else:
            goal_kind = "suelo"
            goal_xy = _flat_spot(ter, rng, avoid=[], far_from=start, min_dist=30)
            obstacles = [o for o in obstacles if o.distance((goal_xy[0], goal_xy[1], ter.height(*goal_xy) + 0.5)) > 4]
    if goal_kind == "cima":  # los árboles no ocupan la cima
        obstacles = [o for o in obstacles if math.dist(((o.footprint()[0] + o.footprint()[1]) / 2,
                                                         (o.footprint()[2] + o.footprint()[3]) / 2), goal_xy) > 5]
        support = "cima"
    world = World(ter, obstacles, start, (0, 0, 0), goal_kind, level, seed, density)
    gx, gy = goal_xy
    if goal_kind == "aire":
        gz = world.surface(gx, gy) + rng.uniform(3, 10)
        support = None
    else:
        gz = world.surface(gx, gy) + PAD_HEIGHT
    world.goal = (gx, gy, gz)
    world.goal_support = support
    if moving:
        from .target import make_motion
        world.motion = make_motion(world, (gx, gy), motion, seed)
        world.goal = world.goal_at(0.0)
        world.goal_support = "vehículo"
    return world
