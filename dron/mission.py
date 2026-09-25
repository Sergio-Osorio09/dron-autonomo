"""Máquina de estados de la misión (como los modos automáticos de PX4: Takeoff → Mission → Land).

Dos modos:

ATERRIZAR   en tierra → despegue → crucero → aproximación → aterrizaje → aterrizado
  * despegue: sube en vertical a MPC_TKO_SPEED hasta 2,5 m sobre el terreno.
  * crucero: sigue la trayectoria planificada hasta 2,5 m por encima de la plataforma (que puede estar en el suelo,
    en una azotea o en una cima). Si el error de seguimiento pasa de 3 m, replanifica desde donde está.
    La holgura con los obstáculos crece con la incertidumbre de posición: 0,6 m + 2σ del filtro (máx. 2,5 m).
  * aproximación: se estabiliza encima. Con aterrizaje de precisión corrige con la cámara inferior (como IR-LOCK).
  * aterrizaje: baja a MPC_LAND_SPEED; si el viento lo desplaza más de 35 cm, deja de bajar y se realinea.
    ÉXITO si queda posado sobre la plataforma.

CARRERA     en tierra → despegue → carrera → [persecución] → meta
  * despega hasta 1,5 m y vuela lo más rápido posible (90 % de la velocidad del fabricante) una trayectoria que NO
    frena al final: cruza la meta como una puerta. ÉXITO al pasar a menos de 1 m del punto meta (1 m sobre la plataforma,
    o un punto en el aire). Si la pasa de largo, vuelve a por ella.
  * OBJETIVO EN MOVIMIENTO: el vehículo emite su posición (rastreador GNSS, 5 Hz); un filtro de Kalman de velocidad
    constante estima dónde está y hacia dónde va (target.py). El dron vuela hacia el PUNTO DE INTERCEPCIÓN (donde
    estará el objetivo cuando llegue) y replanifica cada segundo si ese punto se mueve. A menos de 12 m, y SOLO si
    hay línea de visión libre (si no, la persecución en línea recta atravesaría el árbol que el vehículo rodea), pasa
    a PERSECUCIÓN PREDICTIVA: apunta a la posición predicha 0,5 s por delante y va a la velocidad del objetivo más
    una velocidad de cierre, hasta cruzar la puerta que lleva encima.

MAPA DESCONOCIDO (fase 2, `mapper` = mapping.OccupancyMap): el dron planifica sobre el mapa que va construyendo
con sus sensores, tratando lo que aún no ha visto como libre. Cada vez que el mapa cambia (hasta 5 veces por segundo)
comprueba si lo que queda de trayectoria sigue libre; si no, replanifica desde donde está sin frenar. Si no encuentra
camino, se queda quieto en el sitio y vuelve a intentarlo cada medio segundo (mientras mira alrededor, el mapa
mejora). La altura del objetivo en movimiento sale de su GNSS, no del mapa del terreno.

OBJETIVO QUE HUYE (fase 4, movimiento "huye"): ya no emite su posición. El dron lo sigue con la cámara de detección
en un gimbal (sabe dónde empieza, como en una misión real). Si deja de verlo 2,5 s (se ha escondido tras un
edificio), BÚSQUEDA DESDE LA ÚLTIMA POSICIÓN VISTA: una creencia gaussiana alrededor de donde lo predice el filtro
de Kalman, que se difunde a la velocidad máxima del vehículo, y estrategia bayesiana. En cuanto lo vuelve a ver,
vuelve a interceptarlo.

ENJAMBRE (fase 5, `team` = swarm.Swarm): varios drones comparten la creencia de la búsqueda. Cada uno pide su
siguiente punto sabiendo dónde están y adónde van los demás (Voronoi + subasta voraz, ver search.py), vuela en su
capa de altura y confirma solo lo que ha visto él. Cuando uno confirma la meta, va a por ella y los demás esperan.

BÚSQUEDA (fase 3, `searcher` = search.Searcher): el dron no sabe dónde está la meta, solo la zona donde buscarla.
  despegue → búsqueda (va a los puntos que elige la estrategia, a 8 m del suelo, mirando con la cámara de
  detección) → confirmación (se acerca a 5 m de una detección y la vuelve a mirar) → y, confirmada, lo mismo que
  sin búsqueda: crucero → aproximación → aterrizaje, o carrera hasta la puerta. `pad`, `pad_z` y `gate` son la
  verdad (solo para evaluar); el dron usa `pad_est`, `pad_z_est` y `gate_est`, que salen de lo que detecta.

En ambos modos la velocidad se limita a la que permite frenar dentro del alcance de los telémetros
(v ≤ √(2·a·(alcance − radio − 1 m))): la regla de oro de los drones autónomos. Con lluvia el alcance baja y el dron
vuela más despacio; en carrera, un dron con sensores de 12 m no puede ir a 14 m/s.
"""
import math

import numpy as np

from . import tuning
from .planning import plan, segment_clear
from .search import CONFIRM_AGL
from .target import TargetTracker, intercept_point
from .world import LENGTH, WIDTH

CRUISE_AGL = 2.5
PAD_RADIUS = 0.75
GATE_RADIUS = 1.0
STITCH = 0.5   # s de la trayectoria actual que se conservan al replanificar (0 = sin coser)
MODES = ("aterrizar", "carrera")


class Mission:
    def __init__(self, world, grid, prof, precision: bool = True, mode: str = "aterrizar", sensor_range: float = None,
                 mapper=None, searcher=None):
        if mode not in MODES:
            raise ValueError("Modo de misión desconocido %r" % mode)
        self.world, self.grid, self.prof, self.precision, self.mode = world, grid, prof, precision, mode
        self.phase = "en tierra"
        g = world.goal
        self.pad = np.array(g[:2])                       # centro de la plataforma (o del punto en el aire)
        self.pad_z = g[2]                                # altura de la plataforma (o del punto)
        self.gate = np.array([g[0], g[1], g[2] + (1.0 if world.goal_support else 0.0)])  # punto a cruzar en carrera
        self.moving = world.moving
        self.tracker = TargetTracker() if self.moving else None
        self.intercept = None
        self.last_plan_t = -1.0
        self.speed_now = 0.0
        self.vel_now = np.zeros(3)
        self.los_t, self.los_clear = -1.0, False
        self.pad_est = self.pad.copy()
        self.traj = None
        self.t0 = 0.0
        self.hold = None
        self.replans = 0
        self.yaw = None
        self.plan_error = None
        self.est = None       # filtro de Kalman (lo asigna la simulación): su covarianza fija el margen de seguridad
        self.margin = 0.6
        self.map = mapper     # None: mapa conocido (fase 1)
        self.map_checked = -1
        self.check_t = -1.0
        self.map_replans = 0
        self.blocked = None   # punto donde espera si no encuentra camino
        self.blocked_t = 0.0
        self.target_z = None  # altura del objetivo según su GNSS
        # lo que el dron CREE sobre la meta (sin búsqueda, la conoce; con búsqueda, lo que haya detectado)
        self.search = searcher
        self.pad_z_est, self.gate_est = self.pad_z, self.gate.copy()
        self.search_goal = None
        self.found_t = None
        self.heading = 0.0
        if searcher is not None:
            self.pad_est = self.pad_z_est = self.gate_est = None
        # fase 4: objetivo que huye (sin rastreador GNSS: solo lo que ve la cámara)
        self.evader = self.moving and world.motion.kind == "huye"
        self.last_seen_t, self.lost_count = 0.0, 0
        self.make_searcher = None            # la simulación da cómo crear la búsqueda (con el mapa del dron)
        self.id, self.team, self.search_agl, self.visible_fn = 0, None, 0.0, None   # enjambre (fase 5)
        self.last_meas = None
        if self.evader:                      # sabe dónde está al empezar (la última posición conocida)
            self.tracker.update(0.0, np.array(world.goal[:2]))
            self.target_z = world.goal[2]
        # no volar más rápido de lo que dejan ver los sensores: poder frenar dentro del alcance de los telémetros
        rng = sensor_range or prof.sensor_range
        self.P = tuning.params(prof.key)     # parámetros de seguridad y velocidad de este dron (tuning.py)
        # (Se probó a planificar con el límite más prudente de Collision Prevention, control.cp_speed_limit: bajan
        # algo las replanificaciones, 5,9 → 5,1 por vuelo, pero sube el tiempo 1-2 s y no mejora el éxito. Descartado.)
        self.v_sense = math.sqrt(2 * prof.acc_hor * max(rng - prof.radius - self.P["sense"], 1.0))

    def _track(self, t):
        """Posición y velocidad predichas del objetivo en t. Al que huye no se le extrapola más de 1,5 s desde la
        última vez que se le vio: si no, el filtro lo "movía" sin fin en línea recta y el dron perseguía un fantasma
        (sin llegar nunca lo bastante cerca como para darlo por perdido y buscarlo)."""
        if self.evader:
            t = min(t, self.last_seen_t + 1.5)
        return self.tracker.state_at(t)

    def gate_at(self, t):
        """Puerta a cruzar en el instante t: la real si el objetivo está quieto, la estimada si se mueve."""
        if not self.moving:
            return self.gate
        p, _ = self._track(t)
        if self.map is not None:
            return np.array([p[0], p[1], (self.target_z if self.target_z is not None else self.world.goal[2]) + 1.0])
        return np.array([p[0], p[1], self.world.terrain.height(p[0], p[1]) + self.world.goal[2]
                         - self.world.terrain.height(*self.world.goal[:2]) + 1.0])

    def _cruise(self):
        """Velocidad de crucero: la del perfil por el factor `cruise`, sin pasar del 90 % de la máxima."""
        return min(self.prof.v_cruise * self.P["cruise"], 0.9 * self.prof.v_max)

    def _space(self):
        """(espacio para comprobar segmentos, rejilla para A*): el mundo real o el mapa aprendido."""
        if self.map is None:
            return self.world, self.grid
        g = self.map.grid()
        return g, g

    def _ground_z(self, x, y):
        if self.map is None:
            return self.world.terrain.height(x, y)
        return (self.target_z if self.target_z is not None else self.world.goal[2]) - 0.6  # el vehículo mide 0,6 m

    def _prefix(self, t, p_est, space):
        """Tramo de la trayectoria actual que se conserva al replanificar (0,5 s por delante), si el dron va por
        ella y sigue libre; None si no se puede coser (entonces se planifica desde donde está)."""
        tr = self.traj
        if tr is None or not STITCH or self.speed_now < 1.0 or self.phase not in ("crucero", "carrera", "búsqueda"):
            return None, self.speed_now
        tr_t = t - self.t0
        i0 = int(np.searchsorted(tr.t, tr_t))
        i1 = int(np.searchsorted(tr.t, tr_t + STITCH))
        if i1 - i0 < 3 or i1 >= len(tr.P) - 1 or np.linalg.norm(tr.P[i0] - p_est) > 1.5:
            return None, self.speed_now
        pre = tr.P[i0:i1 + 1]
        need = self.prof.radius + 0.5 * self.P["margin"]
        if not all(segment_clear(space, a, b, need, 0.25) for a, b in zip(pre, pre[1:])):
            return None, self.speed_now
        return pre, float(tr.speed[i0])

    def _plan(self, t, p_est, stitch=False):
        space, grid = self._space()
        pre, v0 = self._prefix(t, p_est, space) if stitch else (None, self.speed_now)
        # margen que crece con la incertidumbre de posición del filtro (2 sigmas), como hacen los planificadores
        # que tienen en cuenta la covarianza
        sigma = math.sqrt(max(self.est.P[0, 0] + self.est.P[1, 1], 0.0)) if self.est is not None else 0.0
        self.margin = min(self.P["margin"] + self.P["sigma_k"] * sigma, 2.5)
        if self.phase in ("búsqueda", "confirmación"):  # hacia el punto que ha elegido la búsqueda
            # buscar a un blanco que HUYE es una persecución: a velocidad de carrera (a la de crucero, el PX4 va a
            # 5 m/s, lo mismo que el vehículo, y nunca le recortaba distancia)
            # (buscar la meta, en cambio, a la velocidad de crucero de fábrica, no la ajustada: el detector necesita
            # tiempo para ver bien, y con el crucero ajustado el Matrice llegaba a 17 m/s a los puntos de búsqueda)
            v = min(0.9 * self.prof.v_max if self.evader else self.prof.v_cruise, self.v_sense)
            new = plan(space, grid, self.prof, p_est, self.search_goal, v, 0.0,
                       self.margin, self.speed_now, min_margin=self.P["margin"], start_vel=self.vel_now)
            if new is None:
                return False
            self.traj, self.t0 = new, t
            return True
        if self.mode == "carrera":
            v = end = min(0.9 * self.prof.v_max, self.v_sense)
            goal = self.gate_est
            if self.moving:  # volar hacia donde ESTARÁ el objetivo
                p_t, v_t = self._track(t)
                xy, _ = intercept_point(p_est, p_t, v_t, 0.8 * v)
                xy = xy + self._flank(p_est, p_t, v_t)
                self.last_plan_t = t
                old, old_t0 = self.traj, self.t0
                for cand in (xy, p_t):  # si el punto de intercepción no es alcanzable, ir a por la posición actual
                    goal = np.array([cand[0], cand[1], self._ground_z(cand[0], cand[1]) + 2.0])
                    self.traj = None
                    if pre is not None:
                        self.traj = plan(space, grid, self.prof, p_est, goal, v, end, self.margin, v0, pre,
                                         min_margin=self.P["margin"])
                    if self.traj is None:
                        self.traj = plan(space, grid, self.prof, p_est, goal, v, end, self.margin,
                                         self.speed_now, min_margin=self.P["margin"], start_vel=self.vel_now)
                    if self.traj is not None:
                        self.intercept = goal
                        self.t0 = t
                        self.phase = "carrera"
                        return True
                self.traj, self.t0 = old, old_t0  # sin camino ahora: seguir con la trayectoria anterior
                if old is None:
                    self.plan_error = "sin camino"
                    return False
                return True
        else:
            goal = np.array([self.pad_est[0], self.pad_est[1], self.pad_z_est + CRUISE_AGL])
            v, end = min(self._cruise(), self.v_sense), 0.0
        mm = self.P["margin"]
        new = plan(space, grid, self.prof, p_est, goal, v, end, self.margin, v0, pre, min_margin=mm)             if pre is not None else None
        if new is None:
            new = plan(space, grid, self.prof, p_est, goal, v, end, self.margin, self.speed_now, min_margin=mm,
                       start_vel=self.vel_now)
        if new is None:
            if self.traj is None:  # ni siquiera hay un primer plan
                self.plan_error = "sin camino"
                return False
            return False  # replanificar falló (p. ej. desplazado junto a una pared): seguir con la trayectoria anterior
        self.traj = new
        self.t0 = t
        self.phase = "carrera" if self.mode == "carrera" else "crucero"
        return True

    # ------------------------------------------------------------------ búsqueda (fase 3)
    def _next_search(self, t, est_p):
        """Pide a la estrategia el siguiente punto y planifica hasta él (si no hay camino, prueba otro)."""
        for _ in range(4):
            # al vehículo que huye se le busca más alto (+7 m): los edificios y árboles tapan menos y se ve más terreno
            agl = self.search_agl + (7.0 if self.evader else 0.0)
            self.search_goal = self.search.next_goal(est_p, self._team_info(), agl)
            if self._plan(t, est_p):
                return True
            self.search.reject(self.search_goal)   # inalcanzable: la estrategia elegirá otro punto
        self.blocked, self.blocked_t = est_p.copy(), t
        return False

    def gimbal_aim(self):
        """(fase 4) Adónde apunta el gimbal de la cámara: al vehículo predicho mientras lo sigue; None si lo busca."""
        if not self.evader or self.phase not in ("carrera", "persecución", "reintento", "despegue", "en tierra"):
            return None
        p, _ = self.tracker.state_at(self.tracker.t)
        return np.array([p[0], p[1], (self.target_z or 0.0) - 0.3])

    def _lost(self, t, est_p):
        """(fase 4) Lo ha perdido: búsqueda desde la última posición vista."""
        self.lost_count += 1
        c, _ = self.tracker.state_at(min(t, self.last_seen_t + 1.5))   # dónde iba, sin fiarse de más
        # la creencia cubre TODA la arena: a 5 m/s, el vehículo sale en segundos de cualquier recuadro alrededor de
        # donde se perdió (con un recuadro de 44 m, a veces no se volvía a encontrar nunca)
        self.search = self.make_searcher((1.0, LENGTH - 1.0, 1.0, WIDTH - 1.0), (float(c[0]), float(c[1])))
        self.phase = "búsqueda"
        self._next_search(t, est_p)

    def share_sighting(self, t, xy, z):
        """(fase 5 + 4) Otro dron del equipo ha visto al vehículo: se actualiza el filtro como si lo viera él."""
        if t <= self.last_seen_t:
            return
        self.last_seen_t = t
        self.tracker.update(t, xy)
        if z is not None:
            self.target_z = z if self.target_z is None else self.target_z + 0.2 * (z - self.target_z)

    def _evader_search_step(self, t, est_p, readings):
        if "evader_look" not in readings:
            return
        if "target" in readings or t - self.last_seen_t < 0.3:   # ¡lo ha vuelto a ver (él u otro)! a interceptarlo
            self.phase = "carrera"
            self.blocked = None
            self._plan(t, est_p)
            return
        self.search.predict(0.2)
        self.search.glimpse(est_p, self.heading, None)

    def _flank(self, p_est, p_t, v_t):
        """(fases 4 + 5) Cerco: con varios perseguidores, el primero va al punto de intercepción y los demás se abren
        6 m a un lado y al otro de la dirección de huida, para cerrarle las salidas."""
        if self.team is None or not self.evader or self.id == 0:
            return np.zeros(2)
        d = np.asarray(v_t[:2], float)
        if np.linalg.norm(d) < 0.5:
            d = np.asarray(p_t[:2], float) - np.asarray(p_est[:2], float)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.zeros(2)
        side = 1.0 if self.id % 2 else -1.0
        return side * 6.0 * np.array([-d[1], d[0]]) / n

    def _team_info(self):
        """(fase 5) Lo que este dron sabe de los demás por radio: dónde están y adónde van."""
        if self.team is None:
            return None
        others = [m for m in self.team.missions if m is not self]
        return {"me": self.id, "n": len(self.team.missions),
                "others": [m.est.p[:2].copy() for m in others],
                "goals": [m.search_goal[:2].copy() for m in others if m.search_goal is not None]}

    def _search_step(self, t, est_p, readings):
        """Procesa una imagen de la cámara de detección y cambia de fase si hace falta."""
        if self.evader:
            self._evader_search_step(t, est_p, readings)
            return None
        found_by = getattr(self.search, "found_by", None)
        if found_by is not None and found_by != self.id:   # otro dron la ha encontrado: esperar en el aire
            self.phase = "espera"
            self.hold = est_p.copy()
            return None
        if "detect" not in readings:
            return None
        rel = readings["detect"]["rel"]
        s = self.search
        s.glimpse(est_p, self.heading, None if rel is None else est_p + rel, self.visible_fn, self.id)
        zero = np.zeros(3)
        if self.phase == "búsqueda" and s.candidate is not None and s.candidate.get("by", self.id) == self.id:
            # una detección: ir a confirmarla ya, a 5 m de altura y un poco antes (la cámara mira 60° hacia abajo).
            # (Se probó a comprobarla primero sin desviarse: fue peor, 31 → 40 s de media con la bayesiana, porque
            # mientras tanto se ignoraban las demás detecciones, incluida la de la plataforma de verdad.)
            c = s.candidate["pos"]
            u = c[:2] - est_p[:2]
            n = float(np.linalg.norm(u))
            back = CONFIRM_AGL / math.tan(math.radians(60.0))
            xy = c[:2] - (u / n) * back if n > back else est_p[:2]
            self.phase = "confirmación"
            self.search_goal = np.array([xy[0], xy[1], c[2] + CONFIRM_AGL])
            if not self._plan(t, est_p):
                self.blocked, self.blocked_t = est_p.copy(), t
            return None
        if self.phase == "confirmación" and self.traj is not None and t - self.t0 >= self.traj.duration:
            res = s.confirm_step()
            if res == "confirmada":
                s.found_by = self.id
                c = s.candidate["pos"]
                self.found_t = t
                self.pad_est = c[:2].copy()
                self.pad_z_est = float(c[2])
                self.gate_est = np.array([c[0], c[1], c[2] + 1.0])
                self.phase = "carrera" if self.mode == "carrera" else "crucero"
                if not self._plan(t, est_p):
                    self.blocked, self.blocked_t = est_p.copy(), t
                    return self.blocked, zero, zero, self.yaw
            elif res == "falsa":
                self.phase = "búsqueda"
                self._next_search(t, est_p)
        return None

    def _still_clear(self, t, est_p) -> bool:
        """¿Lo que queda de trayectoria sigue libre en el mapa actual? Se ignora el primer metro alrededor del dron:
        si acaba de ver un obstáculo muy cerca, replanificar desde ahí daría el mismo resultado una y otra vez
        (de eso ya se encarga Collision Prevention)."""
        tr = self.traj
        i0 = int(np.searchsorted(tr.t, t - self.t0))
        P = tr.P[i0:]
        P = P[np.linalg.norm(P - est_p, axis=1) > self.prof.radius + 1.0]
        if not len(P):
            return True
        return bool((self.map.grid().clearance_many(P) >= self.prof.radius + 0.5 * self.P["margin"]).all())

    def reference(self, t, est_p, est_v, readings):
        """(p_ref, v_ref, a_ref, yaw_ref) para el control."""
        prof = self.prof
        self.speed_now = min(float(np.linalg.norm(est_v)), 0.9 * prof.v_max)
        self.vel_now = np.asarray(est_v, float).copy()
        if self.moving and "target" in readings:
            self.last_seen_t = t
            self.last_meas = (t, np.asarray(readings["target"], float), readings.get("target_z"))
            self.tracker.update(t, readings["target"])
            if "target_z" in readings:
                z = readings["target_z"]
                self.target_z = z if self.target_z is None else self.target_z + 0.2 * (z - self.target_z)
        if "yaw" in readings:
            self.heading = readings["yaw"]
        # la cámara inferior ve la plataforma: mejor estimación relativa
        if "pad_rel" in readings and self.precision and self.pad_est is not None:
            self.pad_est += (est_p[:2] + readings["pad_rel"] - self.pad_est) * 0.2
        zero = np.zeros(3)
        if self.phase == "en tierra":
            self.phase = "despegue"
            self.hold = est_p.copy()
            if self.map is None:
                self.hold_ground = self.world.surface(est_p[0], est_p[1])
            else:  # sin mapa: el suelo está a lo que mide el telémetro inferior
                down = [x["dist"] for x in readings.get("rays", []) if x["label"] == "down"]
                self.hold_ground = est_p[2] - (down[0] if down else 0.0)
            self.t0 = t
        if self.phase == "despegue":
            alt = 1.5 if self.mode == "carrera" else CRUISE_AGL
            top = self.hold_ground + alt
            speed = prof.v_up if self.mode == "carrera" else prof.tko_speed
            z = min(self.hold[2] + speed * (t - self.t0), top)
            if est_p[2] > top - 0.3:
                if self.search is not None:
                    self.phase = "búsqueda"
                    self._next_search(t, est_p)
                else:
                    self._plan(t, est_p)
            return np.array([self.hold[0], self.hold[1], z]), np.array([0, 0, speed if z < top else 0.0]), zero, self.yaw
        if self.evader and self.phase in ("carrera", "reintento", "persecución") and t - self.last_seen_t > 2.5:
            # perdido = debería verlo (está cerca de donde lo predice) y no lo ve: se ha escondido. De lejos, la
            # cámara no alcanza: sigue volando hacia la predicción
            p_t, _ = self._track(t)
            if np.linalg.norm(p_t - est_p[:2]) < 25.0:
                self._lost(t, est_p)
        if self.moving and self.phase in ("carrera", "reintento", "persecución"):
            gate = self.gate_at(t)
            if t - self.los_t > 0.2:  # ¿línea de visión libre hasta el objetivo? (5 veces por segundo)
                self.los_t = t
                if self.map is None:
                    self.los_clear = segment_clear(self.world, est_p, gate, self.prof.radius + 0.8, 0.4)
                else:
                    # la puerta va a 1 m del vehículo y el dron vuela a su altura: con celdas de 1 m, el suelo
                    # aprendido les quita holgura a los dos extremos y la línea de visión saldría siempre
                    # "bloqueada". Se comprueba el tramo intermedio (1 m tras el dron, 1,5 m antes de la puerta)
                    # con radio + 0,3 m: basta para ver si hay un árbol o un edificio en medio.
                    d = gate - est_p
                    n = float(np.linalg.norm(d))
                    if n < 1.2:
                        self.los_clear = True
                    else:
                        u = d / n
                        a = est_p + u * min(1.0, 0.3 * n)
                        b = gate - u * min(1.5, 0.5 * n)
                        self.los_clear = segment_clear(self._space()[0], a, b, self.prof.radius + 0.3, 0.3)
            if np.linalg.norm(gate[:2] - est_p[:2]) < 12.0 and self.los_clear:  # cerca y a la vista: persecución
                self.phase = "persecución"
                p_t, v_t = self._track(t + 0.5)
                # en equipo, solo el primero baja a la puerta; los demás cierran por encima (+2 m por puesto), así
                # no convergen todos en el mismo punto (con 3 drones chocaban entre ellos)
                up = 2.0 * self.id if (self.team is not None and self.evader) else 0.0
                aim = np.array([min(max(p_t[0], 2.0), LENGTH - 2.0), min(max(p_t[1], 2.0), WIDTH - 2.0), gate[2] + up])
                d = aim - est_p
                dist = float(np.linalg.norm(d))
                close = min(self.v_sense, max(2.0, 0.8 * dist))  # velocidad de cierre
                v_ref = np.array([v_t[0], v_t[1], 0.0]) + d / max(dist, 1e-6) * close
                if math.hypot(v_ref[0], v_ref[1]) > 1.0:
                    self.yaw = math.atan2(v_ref[1], v_ref[0])
                return aim, v_ref, np.zeros(3), self.yaw
            if self.phase == "persecución":  # se ha alejado: volver a interceptar
                self._plan(t, est_p)
            elif t - self.last_plan_t > 1.0:  # el punto de intercepción cambia: replanificar cada segundo
                p_t, v_t = self._track(t)
                xy, _ = intercept_point(est_p, p_t, v_t, 0.8 * min(0.9 * prof.v_max, self.v_sense))
                if self.intercept is None or np.linalg.norm(xy - self.intercept[:2]) > 2.0 or self.phase == "reintento":
                    self._plan(t, est_p, stitch=True)
                else:
                    self.last_plan_t = t
        if self.search is not None and self.phase in ("búsqueda", "confirmación"):
            out = self._search_step(t, est_p, readings)
            if out is not None:
                return out
        # sin camino: esperar y reintentar
        if self.blocked is not None and self.phase in ("crucero", "carrera", "búsqueda", "confirmación"):
            if t - self.blocked_t > 0.5:
                self.blocked_t = t
                if self._plan(t, est_p):
                    self.blocked = None
                elif self.phase == "búsqueda":           # ese punto ya no es alcanzable: buscar por otro sitio
                    self.search.reject(self.search_goal)
                    if self._next_search(t, est_p):
                        self.blocked = None
            if self.blocked is not None:
                return self.blocked, zero, zero, self.yaw
        if self.phase in ("crucero", "carrera", "búsqueda", "confirmación") and self.map is not None \
                and t - self.check_t > 0.2 and self.map.version != self.map_checked:
            self.check_t, self.map_checked = t, self.map.version
            if not self._still_clear(t, est_p):
                self.replans += 1
                self.map_replans += 1
                if not self._plan(t, est_p, stitch=True):
                    self.blocked, self.blocked_t = est_p.copy(), t
                    return self.blocked, zero, zero, self.yaw
        if self.phase in ("crucero", "carrera", "búsqueda", "confirmación"):
            p, v, a = self.traj.sample(t - self.t0)
            if np.linalg.norm(p - est_p) > 3.0:  # nos hemos desviado mucho: replanificar desde aquí
                self.replans += 1
                if self._plan(t, est_p):
                    p, v, a = self.traj.sample(0.0)
            if math.hypot(v[0], v[1]) > 1.0:
                self.yaw = math.atan2(v[1], v[0])
            if t - self.t0 >= self.traj.duration:
                if self.phase == "búsqueda":           # punto alcanzado: el siguiente
                    self._next_search(t, est_p)
                elif self.phase == "confirmación":     # encima de la detección: quieto, mirándola
                    c = self.search.candidate
                    if c is not None:
                        self.yaw = math.atan2(c["pos"][1] - est_p[1], c["pos"][0] - est_p[0])
                    return self.traj.P[-1].copy(), zero, zero, self.yaw
                else:
                    self.phase = "reintento" if self.mode == "carrera" else "aproximación"
                    self.t0 = t
            return p, v, a, self.yaw
        if self.phase == "espera":  # enjambre: otro dron ha encontrado la meta
            return self.hold, zero, zero, self.yaw
        if self.phase == "reintento":  # carrera: la pasó de largo, vuelve a por ella
            g = self.gate_at(t) if self.moving else self.gate_est
            g = np.array([min(max(g[0], 2.0), LENGTH - 2.0), min(max(g[1], 2.0), WIDTH - 2.0), g[2]])  # dentro
            return g, zero, zero, self.yaw
        target = np.array([self.pad_est[0], self.pad_est[1], self.pad_z_est + CRUISE_AGL])
        if self.phase == "aproximación":
            err = math.hypot(*(target[:2] - est_p[:2]))
            if (err < 0.3 and np.linalg.norm(est_v) < 0.4) or t - self.t0 > 6.0:
                self.phase = "aterrizaje"
                self.t0 = t
                self.hold = est_p.copy()
            return target, zero, zero, self.yaw
        if self.phase == "aterrizaje":
            err = math.hypot(*(target[:2] - est_p[:2]))
            if err > 0.35 and est_p[2] > self.pad_z_est + 0.6:  # desplazado por el viento: dejar de bajar y realinear
                self.hold[2] = est_p[2]
                self.t0 = t
                return np.array([target[0], target[1], est_p[2]]), zero, zero, self.yaw
            # como el modo Land de PX4: baja a velocidad constante hasta que el detector de aterrizaje nota el suelo
            # (la referencia puede quedar por debajo de la plataforma si la altura estimada va algo desviada)
            z = max(self.hold[2] - prof.land_speed * (t - self.t0), self.pad_z_est - 1.5)
            return np.array([target[0], target[1], z]), np.array([0, 0, -prof.land_speed]), zero, self.yaw
        return est_p, zero, zero, self.yaw
