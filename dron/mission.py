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
  A POR TODAS (carrera, ALL_OUT): en carrera no se para a confirmar. Al detectar, sale a velocidad de carrera hacia
  la detección y la CONFIRMA EN VUELO con las imágenes que toma mientras se acerca (3 que coinciden; solo cuentan las
  imágenes en las que debería verla: dentro del cono y con probabilidad de detección ≥ 0,3). Si era una falsa alarma
  lo descubre por el camino y vuelve a buscar. Con el vehículo, igual: la persecución ya no frena al acercarse.
  SPRINT (carrera): en cuanto VE la meta (línea de visión libre hasta la puerta y, con mapa desconocido, todo el
  pasillo ya observado por los sensores), vuela a su VELOCIDAD MÁXIMA real (la del fabricante: 16 m/s el Mini 3 en
  modo Sport, 23 m/s el Matrice 350 en modo S, 12 m/s MPC_XY_VEL_MAX de PX4), no al 90 % ni al límite de frenar
  dentro del alcance de los sensores: ese límite existe para lo que el dron NO ve, y el pasillo hasta la meta lo ve
  libre. Collision Prevention sigue activo con cada rayo, pero sin su límite global de velocidad (la meta se cruza).
  Si pierde la línea de visión, vuelve a los límites normales.

En ambos modos la velocidad se limita a la que permite frenar dentro del alcance de los telémetros
(v ≤ √(2·a·(alcance − radio − 1 m))): la regla de oro de los drones autónomos. Con lluvia el alcance baja y el dron
vuela más despacio; en carrera, un dron con sensores de 12 m no puede ir a 14 m/s.
"""
import dataclasses
import math

import numpy as np

from . import pilots, tuning
from .control import collision_prevention
from .params import G
from .planning import plan, segment_clear, straight
from .search import CONFIRM_AGL, SEARCH_AGL, in_cone, p_detect
from .target import TargetTracker, intercept_point
from .world import LENGTH, WIDTH

CRUISE_AGL = 2.5
PAD_RADIUS = 0.75
GATE_RADIUS = 1.0
# (PROBADO Y DESACTIVADO, clásico en el X650, 60 semillas por caso: objetivo rápido −3,5 %, pero meta quieta +1 %;
# con otros umbrales, ±1 %. DRON_PROGRESS=1: con ella)
PROGRESS = __import__("os").environ.get("DRON_PROGRESS", "0") != "0"   # variable de progreso (ver reference)
PROGRESS_E0, PROGRESS_E1, PROGRESS_MIN = (float(x) for x in __import__("os").environ.get("DRON_PROGRESS_E", "0.8,2.0,0.2").split(","))   # m por delante sin frenar el reloj, m hasta el mínimo, mínimo
STITCH = 0.5   # s de la trayectoria actual que se conservan al replanificar (0 = sin coser)
MODES = ("aterrizar", "carrera")
# carrera "a por todas": confirmar la detección en vuelo y perseguir sin frenar al acercarse (DRON_ALL_OUT=0 vuelve a
# pararse a confirmar y a frenar en la persecución, para comparar)
ALL_OUT = __import__("os").environ.get("DRON_ALL_OUT", "1") != "0"
# sprint de carrera a la velocidad máxima del fabricante al ver la meta (DRON_SPRINT=0 lo desactiva, para comparar)
SPRINT = __import__("os").environ.get("DRON_SPRINT", "1") != "0"
# persecución: velocidad de cierre √(K·a·d). K = 1, 2 o 4 no cambian nada medible (40 semillas: objetivo rápido
# 15,6-15,7 s; el que huye 37-39/40): lo que manda es la geometría de la intercepción. Se deja el más agresivo de
# los que no empeoran (en carrera, llegar es lo que cuenta: impactar también vale)
PURSUIT_K = float(__import__("os").environ.get("DRON_PURSUIT_K", 2.0))
CARROT = float(__import__("os").environ.get("DRON_CARROT", 3.0))   # persecución: consigna de posición a ≤ CARROT m
RACE_DIVE = 2.0
JERK_RACE = float(__import__("os").environ.get("DRON_JERK_RACE", 8.0))   # MPC_JERK_MAX de PX4 (carrera)
# VENTANA DINÁMICA (Dynamic Window Approach, Fox, Burgard y Thrun 1997) en el planificador local del híbrido: en
# vez de pedir una velocidad y que Collision Prevention la recorte después (se "peleaban"), a cada lectura de los
# telémetros (20 Hz) prueba DWA_N direcciones a ±DWA_SPAN° de la que quiere, les aplica la MISMA Collision Prevention
# que el control y se queda con la que más le acerca a su punto de mira (menos una pequeña penalización por girar).
# PROBADO Y DESACTIVADO (X650, híbrido, 60 semillas por caso): la anticolisión deja de recortar (71 → 21 % del tiempo)
# pero no acelera (11,2 vs 11,3 s en el mixto; 14,8 = 14,8 en el bosque) y en la ciudad 1 choque en 120. DRON_DWA=1: con ella
DWA = __import__("os").environ.get("DRON_DWA", "0") != "0"
DWA_SPAN, DWA_N, DWA_TURN = 60.0, 9, 0.15
REACT_RAMP = float(__import__("os").environ.get("DRON_REACT_RAMP", 0.0))   # rampa del reactivo (× aceleración)
PN_N = float(__import__("os").environ.get("DRON_PN", 3.0))   # constante de la navegación proporcional (0 = sin ella)   # carrera: factor sobre la velocidad de bajada del fabricante al picar en diagonal hacia la meta
TERMINAL = __import__("os").environ.get("DRON_TERMINAL", "1") != "0"   # guía terminal hacia la puerta
# EMBESTIDA (carrera, meta a la vista y pasillo libre): el dron deja de volar "como un dron de fábrica" y usa todo el
# empuje de sus 4 motores para ir directo a la meta, muy inclinado, como un FPV de carreras o un interceptor. En un dron
# real hay que desbloquear la inclinación (PX4: MPC_TILT_MAX_AIR admite hasta 85°) y la velocidad (MPC_XY_VEL_MAX); los
# DJI no lo permiten con su firmware. DRON_RAM=0: sin embestida
RAM = __import__("os").environ.get("DRON_RAM", "1") != "0"
# Inclinación en la embestida: la mayor que aún sostiene el peso con un 10 % de reserva, acos(1,1/TWR) (≈ 55-60°).
# Se probó con 80° (el máximo de MPC_TILT_MAX_AIR es 85°): con las hélices casi apuntando a la meta el empuje
# vertical es cos 80° ≈ 0,17 del total y el dron CAE; a 2 m del suelo no le daba tiempo a enderezarse (Matrice:
# 5 choques en 30 carreras, 4 contra el suelo)
RAM_RESERVE = float(__import__("os").environ.get("DRON_RAM_RESERVE", 1.1))
# velocidad máxima en la embestida: la que da el empuje a esa inclinación contra el arrastre, sin pasar de 1,5 veces
# la del fabricante (ESTIMADO, de drones con modo manual desbloqueado: DJI Avata 2 16 → 27 m/s, DJI FPV 27 → 39 m/s;
# a más velocidad las hélices pierden empuje y el modelo de arrastre deja de valer)
RAM_V_K = 1.5
# y solo con el pasillo hasta la meta libre con tuning "ram_clear" m más de holgura por lado (por dron: a más velocidad,
# cualquier desvío lateral, la inercia de un giro, se come antes el margen; ver tuning.py). DRON_RAM_CLEAR lo fuerza
RAM_CLEAR = __import__("os").environ.get("DRON_RAM_CLEAR")
# aterrizando: confirmar la detección en vuelo, sin bajar a mirarla (DRON_CONFIRM_IN_FLIGHT=0: como antes)
CONFIRM_IN_FLIGHT = __import__("os").environ.get("DRON_CONFIRM_IN_FLIGHT", "1") != "0"


class Mission:
    def __init__(self, world, grid, prof, precision: bool = True, mode: str = "aterrizar", sensor_range: float = None,
                 mapper=None, searcher=None, pilot: str = "clasico", ram: bool = True):
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
        self.sprint, self.sprint_t, self.sprint_dist = False, -1.0, 0.0   # a velocidad máxima: ve la meta
        self.ram, self.ram_clear = False, False                           # embestida (ver RAM)
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
        self.dash = False                    # carrera a por una detección sin confirmar (confirmación en vuelo)
        self.refine = False                  # sigue afinando la puerta con lo que ve hasta cruzarla
        self.refine_t = -1.0
        self.dash_agl = 0.0                  # altura sobre la detección al salir hacia ella
        self.same_goal = 0                   # veces seguidas que la búsqueda propone el sitio donde ya está
        self.retry_t = -1.0                  # carrera: última vez que replanificó tras pasar la meta de largo
        self.pursuit_t, self.pursuit_v = -1.0, None   # persecución: última velocidad pedida (para suavizarla)
        self.dash_hits = []                  # (posición detectada, peso 1/σ²): las de cerca pesan más
        self.gate_hits, self.gate_fix_t = [], -1.0   # carrera: la puerta vista con la cámara (posición relativa)
        self.heading = 0.0
        if searcher is not None:
            self.pad_est = self.pad_z_est = self.gate_est = None
        # fase 4: objetivo que huye (sin rastreador GNSS: solo lo que ve la cámara)
        self.evader = self.moving and world.motion.kind == "huye"
        # sin rastreador: el que huye y, con búsqueda, cualquier objetivo en movimiento. No emite su posición: solo
        # se sabe dónde está cuando la cámara lo ve (antes, con búsqueda y objetivo en movimiento, la simulación
        # quitaba la búsqueda y el dron sabía dónde estaba desde el despegue)
        self.camera_only = self.evader or (self.moving and searcher is not None)
        self.last_seen_t, self.lost_count = 0.0, 0
        self.make_searcher = None            # la simulación da cómo crear la búsqueda (con el mapa del dron)
        self.id, self.team, self.search_agl, self.visible_fn = 0, None, 0.0, None   # enjambre (fase 5)
        self.last_meas = None
        if self.evader and searcher is None:   # sabe dónde está al empezar (la última posición conocida)
            self.tracker.update(0.0, np.array(world.goal[:2]))
            self.target_z = world.goal[2]
        # no volar más rápido de lo que dejan ver los sensores: poder frenar dentro del alcance de los telémetros
        rng = sensor_range or prof.sensor_range
        self.range = rng
        # el algoritmo que pilota (pilots.py) y sus parámetros de seguridad y velocidad (tuning.py)
        self.pilot = pilots.get(pilot)
        self.P = tuning.params(prof.key, tuned=self.pilot.tuned)
        self.all_out = ALL_OUT and self.pilot.all_out
        # (no contra el que huye: esquiva, y lanzarse sin frenar junto a obstáculos le costaba choques al Matrice:
        # 30 persecuciones, 2 → 5 choques, sin capturarlo antes)
        self.ram_on = RAM and ram and self.all_out and mode == "carrera" and not self.evader
        self.sprint_on = SPRINT and self.pilot.sprint
        self.terminal_on = TERMINAL and self.pilot.terminal
        self.confirm_in_flight = CONFIRM_IN_FLIGHT and self.pilot.confirm_in_flight
        self.reactive = self.pilot.planner in ("reactivo", "hibrido")
        self.guided = self.pilot.planner == "hibrido"          # reactivo con guía global (A* cada segundo)
        self.guide, self.guide_t = None, -1.0
        self.gains = dict(self.pilot.gains)
        self.P.update(pilots.RACE_DEFAULTS)
        if mode == "carrera":            # entrenados para la carrera (eval/evolve.py): márgenes, velocidad, ganancias
            for k, val in pilots.race_params(self.pilot, prof.key).items():
                (self.gains if k in self.gains else self.P)[k] = val
        self.react_best, self.react_t, self.stuck_until, self.stuck_side = math.inf, 0.0, -1.0, 1.0
        self.esc_start, self.react_arrived = -1.0, False
        # (Se probó a planificar con el límite más prudente de Collision Prevention, control.cp_speed_limit: bajan
        # algo las replanificaciones, 5,9 → 5,1 por vuelo, pero sube el tiempo 1-2 s y no mejora el éxito. Descartado.)
        self.v_sense = math.sqrt(2 * prof.acc_hor * max(rng - prof.radius - self.P["sense"], 1.0)) * self.P["vs_k"]

    def _track(self, t):
        """Posición y velocidad predichas del objetivo en t. Al que huye no se le extrapola más de 1,5 s desde la
        última vez que se le vio: si no, el filtro lo "movía" sin fin en línea recta y el dron perseguía un fantasma
        (sin llegar nunca lo bastante cerca como para darlo por perdido y buscarlo)."""
        if self.camera_only:
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

    def _race_v(self):
        """Velocidad de carrera: la máxima del fabricante si ve la meta (sprint), más en la embestida; si no, el 90 %
        sin pasar de la que permite frenar dentro del alcance de los sensores."""
        if self.ram:
            return self.ram_speed()
        return self.prof.v_max if self.sprint else min(self.P["race"] * self.prof.v_max, self.v_sense)

    def ram_tilt(self) -> float:
        """Inclinación máxima en la embestida (rad): la que sostiene el peso con reserva, acos(RAM_RESERVE/TWR)."""
        return max(math.acos(min(RAM_RESERVE / max(self.prof.twr, 1.01), 1.0)), self.prof.tilt_max)

    def ram_speed(self) -> float:
        """Velocidad máxima en la embestida. El arrastre del modelo (params.drag) sale de v_max a la inclinación
        máxima del fabricante; a la inclinación de la embestida (ram_tilt), la velocidad en la que el arrastre iguala
        el empuje horizontal es v_max·√(tan θ / tan θ_fábrica). Con el tope RAM_V_K."""
        p = self.prof
        th = self.ram_tilt()
        v = p.v_max * math.sqrt(max(math.tan(th), math.tan(p.tilt_max)) / math.tan(p.tilt_max))
        return min(v, RAM_V_K * p.v_max)

    def ram_acc(self) -> float:
        """Aceleración horizontal en la embestida: g·tan(ram_tilt), sosteniendo la altura."""
        return G * math.tan(self.ram_tilt())

    def _sprint_check(self, t, est_p) -> bool:
        """¿Ve la meta? Línea de visión libre hasta la puerta con la holgura del planificador y, con mapa
        desconocido, todo el pasillo ya observado (no basta con "desconocido = libre"). Al que huye, además, tiene que
        haberlo visto hace menos de 0,5 s; con búsqueda, solo tras detectar la meta."""
        if self.mode != "carrera" or not self.sprint_on or self.phase not in ("carrera", "persecución", "reintento"):
            return False
        gate = self.gate_at(t) if self.moving else self.gate_est
        if gate is None or (self.camera_only and t - self.last_seen_t > 0.5):
            return False
        d = gate - est_p
        n = float(np.linalg.norm(d))
        if n < 1.5:
            self.sprint_dist = n
            self.ram_clear = True
            return True
        u = d / n
        a = est_p + u * min(1.0, 0.3 * n)
        b = gate - u * min(1.5, 0.5 * n)       # el último tramo roza el suelo de la plataforma (celdas de 1 m)
        if not segment_clear(self._space()[0], a, b, self.prof.radius + self.P["margin"], 0.3):
            return False
        if self.map is not None:
            P = a + (b - a) * np.linspace(0.0, 1.0, max(2, int(n / 0.5)))[:, None]
            idx = np.clip(np.floor(P).astype(int), 0, np.array(self.map.seen.shape) - 1)
            if not self.map.seen[idx[:, 0], idx[:, 1], idx[:, 2]].all():
                return False
        self.sprint_dist = n
        self.ram_clear = self.ram_on and segment_clear(self._space()[0], a, b,
                                                       self.prof.radius + self.P["margin"]
                                                       + (float(RAM_CLEAR) if RAM_CLEAR else self.P["ram_clear"]), 0.3)
        return True

    def _planner(self, *args, **kw):
        """`planning.plan` o, con el piloto reactivo, una recta hasta la meta (sin coser: con `prefix`, None)."""
        if not self.reactive:
            return plan(*args, **kw)
        a = list(args) + [None] * 10
        prefix = a[9] if a[9] is not None else kw.get("prefix")
        if prefix is not None:
            return None
        start, goal, v, end = a[3], a[4], a[5], a[6] or 0.0
        start_speed = a[8] if a[8] is not None else kw.get("start_speed", 0.0)
        self.react_best, self.react_t, self.react_arrived = math.inf, -1.0, False
        return straight(a[2], start, goal, v or self.prof.v_cruise, end, start_speed)

    def _guide_point(self, t, est_p, goal, look):
        """(híbrido) Punto `look` metros por delante en un camino A* hasta `goal` sobre el mapa (el aprendido con mapa
        desconocido; lo desconocido es libre), recalculado cada segundo (~15-35 ms). None si no hay camino."""
        from .planning import astar
        if t - self.guide_t > 1.0:
            self.guide_t = t
            grid = self._space()[1]
            cells = astar(grid, est_p, goal, self.prof.radius + self.P["margin"])
            self.guide = None if cells is None else np.array([grid.center(c) for c in cells] + [goal])
        if self.guide is None or len(self.guide) < 2:
            return None
        dd = np.linalg.norm(self.guide - est_p, axis=1)
        i = int(np.argmin(dd))                                 # dónde está sobre el camino
        ahead = np.nonzero(dd[i:] >= look)[0]
        return self.guide[i + ahead[0]] if len(ahead) else self.guide[-1]

    def _reactive(self, t, est_p, est_v, readings):
        """PILOTO REACTIVO (campos de potencial, Khatib 1986): velocidad = atracción hacia el final de la recta +
        repulsión de cada obstáculo que ven los telémetros a menos de R (∝ 1/d − 1/R), más lento con algo delante.
        Atascado (no se acerca en `stuck_t` s: mínimo local), rodea por el lado más libre durante 3 s, como los
        algoritmos Bug. Llega cuando está a menos de 1 m (o lo cruza, en carrera): la misión sigue como al terminar
        una trayectoria."""
        G = self.gains
        # las ÚLTIMAS lecturas de los telémetros (miden a 20 Hz y el piloto decide a 200 Hz). Antes solo se usaban en
        # el paso en que llegaban: en 9 de cada 10 decisiones no había repulsión ni "algo delante" y el dron iba recto
        # hacia la meta; Collision Prevention le recortaba la velocidad pedida el 65 % del tiempo (X650)
        if "rays" in readings or not hasattr(self, "_rays"):
            self._rays = readings.get("rays", [])
        all_rays = self._rays
        goal = self.traj.P[-1]
        d = goal - est_p
        dist = float(np.linalg.norm(d))
        zero = np.zeros(3)
        if dist < 1.0 or (self.traj.end_speed > 0 and dist < 2.5 and float(d @ est_v) < 0):
            self.react_arrived = True                          # llegada: la misión pasa a lo siguiente
            return goal.copy(), zero, zero
        u = d / dist
        if self.guided:        # híbrido: hacia el punto de mira del camino A* (si lo hay), no en línea recta
            aim = self._guide_point(t, est_p, goal, G["look"])
            if aim is not None:
                da = aim - est_p
                na = float(np.linalg.norm(da))
                if na > 0.3:
                    u = da / na
        acc = self.prof.acc_hor
        spd = min(self.traj.v_cruise, math.sqrt(self.traj.end_speed ** 2 + 2 * 0.7 * acc * dist))
        rays = [r for r in all_rays if abs(r["dir"][2]) < 0.5 and r["dist"] < G["radius"]]
        rep = np.zeros(3)
        front = math.inf
        for r in rays:
            q = np.asarray(r["dir"], float)
            dd = max(float(r["dist"]) - self.prof.radius, 0.2)
            ahead = float(q @ u)
            w = 1.0 if ahead > 0 else G["k_side"]
            rep -= q * w * G["k_rep"] * (1.0 / dd - 1.0 / G["radius"])
            if ahead > 0.7:
                front = min(front, dd)
        # los bordes de la arena (la geovalla) también repelen, con el margen del planificador (que crece con la
        # incertidumbre del GPS): los telémetros no los ven y, empujado por los edificios, el dron se pegaba a ellos
        # y con el error del GPS acababa fuera (2 choques de 40 en la ciudad)
        for dist_b, away in ((est_p[0], (1, 0)), (LENGTH - est_p[0], (-1, 0)), (est_p[1], (0, 1)),
                             (WIDTH - est_p[1], (0, -1))):
            dd = max(dist_b - self.prof.radius - self.margin, 0.2)
            if dd < G["radius"]:
                rep += np.array([away[0], away[1], 0.0]) * G["k_rep"] * (1.0 / dd - 1.0 / G["radius"])
        n = float(np.linalg.norm(rep))
        if n > 1.5 * spd:
            rep *= 1.5 * spd / n
        if front < G["radius"]:
            spd *= max(0.2, 1.0 - G["slow"] * (1.0 - front / G["radius"]) * 2.0)
        # atasco (mínimo local): no mejora su mejor distancia en stuck_t segundos
        if dist < self.react_best - 0.5:
            self.react_best, self.react_t = dist, t
        elif self.react_t < 0:
            self.react_t = t
        elif t - self.react_t > G["stuck_t"] and t > self.stuck_until:
            # atascado: rodea el obstáculo (por el lado más libre, siempre el mismo) hasta ver libre la dirección de
            # la meta o hasta esc_t segundos, como el algoritmo Bug2 (seguir el borde del obstáculo)
            self.stuck_until, self.react_t, self.esc_start = t + G["esc_t"], t, t
            side = [float(np.cross(u, r["dir"])[2]) for r in all_rays]
            left = sum(r["dist"] for r, c in zip(all_rays, side) if c > 0.3)
            right = sum(r["dist"] for r, c in zip(all_rays, side) if c < -0.3)
            self.stuck_side = 1.0 if left >= right else -1.0
        if t < self.stuck_until and t - self.esc_start > 1.0 and front >= G["radius"]:
            self.stuck_until = t                               # ya ve libre la dirección de la meta
        v = u * spd + rep
        if t < self.stuck_until:                               # rodear: perpendicular a la meta, hacia lo más libre
            v = u * spd * 0.3 + rep + self.stuck_side * np.array([-u[1], u[0], 0.0]) * G["k_tan"] * self.traj.v_cruise
        elif DWA and self.guided:
            v = self._dwa(v, u, est_v, all_rays, "rays" in readings)
        if REACT_RAMP:
            # velocidad pedida en RAMPA (como la persecución y el generador de consignas de PX4): cambia como mucho a
            # la aceleración del planificador; el campo de potencial pedía saltos que el dron no puede dar
            dt = t - getattr(self, "_react_t", -1.0)
            base = getattr(self, "_react_v", None)
            if base is None or not 0.0 < dt < 0.5:
                base = np.asarray(est_v, float)
                dt = 0.05
            step = REACT_RAMP * self.prof.acc_hor * self.P["acc_k"] * dt
            dv = v - base
            ndv = float(np.linalg.norm(dv))
            if ndv > step:
                v = base + dv * (step / ndv)
            self._react_t, self._react_v = t, v.copy()
        return est_p.copy(), v, zero

    def _dwa(self, v, u, est_v, rays, fresh):
        """Ventana dinámica (ver DWA): de las direcciones cercanas a la de `v`, la que más avanza hacia `u` después de
        Collision Prevention. Se recalcula con cada lectura nueva de los telémetros; entre medias, el mismo giro."""
        h = math.hypot(v[0], v[1])
        if h < 0.5 or not rays:
            return v
        if fresh or not hasattr(self, "_dwa_turn"):
            best, self._dwa_turn, self._dwa_k = -math.inf, 0.0, 1.0
            for k in range(DWA_N):
                a = math.radians(-DWA_SPAN + 2 * DWA_SPAN * k / (DWA_N - 1))
                c, s = math.cos(a), math.sin(a)
                cand = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])
                lim, _ = collision_prevention(cand.copy(), np.zeros(3), rays, self.prof, est_v, 0.0, self.P)
                score = float(lim[:2] @ u[:2]) - DWA_TURN * abs(a) * h
                if score > best:
                    best, self._dwa_turn = score, a
                    n = math.hypot(lim[0], lim[1])
                    self._dwa_k = min(1.0, n / h) if h > 1e-6 else 1.0
        c, s = math.cos(self._dwa_turn), math.sin(self._dwa_turn)
        return np.array([(v[0] * c - v[1] * s) * self._dwa_k, (v[0] * s + v[1] * c) * self._dwa_k, v[2]])

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
            # (buscar a 1,5-2 veces el crucero no acorta nada: 40 semillas, los tres drones)
            v = min(0.9 * self.prof.v_max if self.camera_only else self.prof.v_cruise, self.v_sense)
            new = self._planner(space, grid, self.prof, p_est, self.search_goal, v, 0.0,
                       self.margin, self.speed_now, min_margin=self.P["margin"], start_vel=self.vel_now)
            if new is None:
                return False
            self.traj, self.t0 = new, t
            return True
        prof = self.prof
        if self.mode == "carrera":
            v = end = self._race_v()
            # aceleración de la carrera: la del planificador por acc_k (entrenado), sin pasar del 90 % de la física
            # (g·tan de la inclinación máxima); en el sprint, al menos acc_max (MPC_ACC_HOR_MAX en PX4)
            a_plan = min(self.prof.acc_hor * self.P["acc_k"], 0.9 * G * math.tan(self.prof.tilt_max))
            if self.sprint:
                a_plan = max(a_plan, self.prof.acc_max)
            # y el picado hacia la meta (a 1 m del suelo) sin el límite de bajada de los modos automáticos
            # (MPC_Z_V_AUTO_DN: pensado para bajar en vertical, donde está el riesgo del anillo de vórtice; bajando en
            # diagonal con velocidad horizontal no lo hay): RACE_DIVE veces la bajada del fabricante (ESTIMADO)
            # y el tirón de los modos manuales (MPC_JERK_MAX, 8 m/s³ por defecto en PX4) en vez del de misión
            # (MPC_JERK_AUTO, 4): el perfil de velocidad se suaviza menos y no recorta los picos de las trayectorias
            # cortas (clásico en el X650, 60 semillas: mixto 13,1 → 12,8 s, bosque 18,4 → 17,3 s, sin choques)
            prof = dataclasses.replace(self.prof, acc_max=0.0, acc_hor=max(a_plan, self.prof.acc_hor),
                                       v_down=self.prof.v_down * RACE_DIVE, jerk=max(self.prof.jerk, JERK_RACE))
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
                        self.traj = self._planner(space, grid, prof, p_est, goal, v, end, self.margin, v0, pre,
                                         min_margin=self.P["margin"])
                    if self.traj is None:
                        self.traj = self._planner(space, grid, prof, p_est, goal, v, end, self.margin,
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
            # confirmando en vuelo, a la altura que lleva (la de búsqueda): baja en vertical sobre la plataforma en la
            # aproximación. En diagonal desde 8-13 m, bajando entre copas, con el error del GPS del Mini rozó un árbol
            agl = max(CRUISE_AGL, self.dash_agl) if self.refine else CRUISE_AGL
            goal = np.array([self.pad_est[0], self.pad_est[1], self.pad_z_est + agl])
            v, end = min(self._cruise(), self.v_sense), 0.0
        mm = self.P["margin"]
        new = self._planner(space, grid, prof, p_est, goal, v, end, self.margin, v0, pre, min_margin=mm) \
            if pre is not None else None
        if new is None:
            new = self._planner(space, grid, prof, p_est, goal, v, end, self.margin, self.speed_now, min_margin=mm,
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
            agl = self.search_agl + (7.0 if self.camera_only else 0.0)
            self.search_goal = self.search.next_goal(est_p, self._team_info(), agl)
            # si propone 3 veces seguidas el sitio donde ya está, desde ahí no ve esa zona (una azotea la tapa, por
            # ejemplo): se descarta. Antes el dron se quedaba ahí parado hasta agotar el tiempo
            near = float(np.linalg.norm(self.search_goal[:2] - est_p[:2])) < 2.0
            self.same_goal = self.same_goal + 1 if near else 0
            if self.same_goal >= 3:
                self.same_goal = 0
                self.search.reject(self.search_goal)
                continue
            if self._plan(t, est_p):
                return True
            self.search.reject(self.search_goal)   # inalcanzable: la estrategia elegirá otro punto
        self.blocked, self.blocked_t = est_p.copy(), t
        return False

    def gimbal_aim(self):
        """(fase 4) Adónde apunta el gimbal de la cámara: al vehículo predicho mientras lo sigue; None si lo busca."""
        if not self.camera_only or self.tracker.x is None \
                or self.phase not in ("carrera", "persecución", "reintento", "despegue", "en tierra"):
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
            if self.found_t is None:
                self.found_t = t
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
        if self.camera_only:
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
        if self.dash or self.refine:
            return self._dash_step(t, est_p, None if rel is None else est_p + rel)
        if self.phase == "búsqueda" and s.candidate is not None and s.candidate.get("by", self.id) == self.id \
                and (self.mode == "carrera" and self.all_out or self.mode == "aterrizar" and self.confirm_in_flight):
            # sin pararse: hacia la detección (en carrera a toda velocidad; aterrizando, a crucero hasta 2,5 m sobre
            # ella), confirmándola por el camino con las imágenes que toma. Aterrizando, antes bajaba a 5 m, se
            # quedaba hasta 4 s mirándola y luego volvía a subir para ir a la plataforma: ~17 s de 43
            self.dash = self.refine = True
            self.dash_hits = []
            self._goal_from(s.candidate["pos"])
            # (como mucho la altura de búsqueda normal: en enjambre, cada dron busca en su capa, 2-4 m más arriba, y la
            # bajada en vertical desde ahí hacía que 2-3 drones no fueran más rápidos que uno)
            self.dash_agl = min(est_p[2] - self.pad_z_est, SEARCH_AGL)
            self.phase = "carrera" if self.mode == "carrera" else "crucero"
            # cosida a la trayectoria de búsqueda: sin coser, el camino nuevo salía en otra dirección y además bajando,
            # y el Matrice (6,5 kg, 30° de inclinación máxima) iba saturado, se desviaba 1,5 m y rozó una copa
            if not self._plan(t, est_p, stitch=True):
                self.blocked, self.blocked_t = est_p.copy(), t
            return None
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

    def _goal_from(self, c):
        """Lo que el dron cree de la meta a partir de una detección (posición de la plataforma)."""
        self.pad_est = np.array(c[:2], float)
        self.pad_z_est = float(c[2])
        self.gate_est = np.array([c[0], c[1], c[2] + 1.0])

    def _dash_step(self, t, est_p, det):
        """Confirmación EN VUELO (carrera a por todas): cada imagen que debería ver la detección cuenta; con 3 que
        coinciden, confirmada; 8 sin confirmarla, falsa alarma. Hasta cruzar la puerta, la afina con la media de lo
        visto pesada por 1/σ² (el error de la detección crece con la distancia: vista desde 20 m y confirmada de
        lejos, la puerta quedaba a 1,3 m y el dron esperaba en el sitio equivocado) y, si se mueve más de 0,5 m,
        replanifica cosiendo la trayectoria."""
        s = self.search
        c = s.candidate
        if c is None:
            self.dash = self.refine = False
            return None
        if det is not None and np.linalg.norm(det[:2] - c["pos"][:2]) < 3.0:
            self.dash_hits.append((det, 1.0 / (0.02 * float(np.linalg.norm(det - est_p)) + 0.1) ** 2))
        if self.dash and self.phase == "aproximación":
            # encima de la detección: la cámara mira adelante y abajo, así que se gira hacia ella (encima, lo que
            # queda detrás no lo ve y el dron se quedaba quieto para siempre). Si en 5 s no la confirma, no era
            self.yaw = math.atan2(c["pos"][1] - est_p[1], c["pos"][0] - est_p[0])
            if t - self.t0 > 5.0:
                c["state"] = "falsa"
                s.candidate = None
                self.dash = self.refine = False
                self.phase = "búsqueda"
                self._next_search(t, est_p)
                return None
        if self.dash:
            rel = c["pos"] - est_p
            counts = bool(in_cone(rel, self.heading)[0]) and float(p_detect(np.linalg.norm(rel), s.range_factor)) >= 0.3
            res = s.confirm_step(counts)
            if res == "falsa":
                self.dash = self.refine = False
                self.phase = "búsqueda"
                self._next_search(t, est_p)
                return None
            if res == "confirmada":
                self.dash = False
                s.found_by = self.id
                self.found_t = t
        if self.dash_hits:
            W = np.array([w for _, w in self.dash_hits])
            pos = (np.array([h for h, _ in self.dash_hits]) * W[:, None]).sum(axis=0) / W.sum()
            moved = float(np.linalg.norm(pos - self.gate_est + np.array([0, 0, 1.0])))
            if self.phase == "carrera" and moved > 0.5 or self.phase == "crucero" and moved > 1.0 \
                    and t - self.refine_t > 1.0:
                # (en la aproximación manda el aterrizaje de precisión; en crucero, replanificar a cada medio metro,
                # bajando entre árboles, dejaba al Matrice 1,1 m fuera de su trayectoria y rozó una copa)
                self.refine_t = t
                self._goal_from(pos)
                self._plan(t, est_p, stitch=True)
        return None

    def _gate_seen(self, t, est_p, q):
        """(carrera a una meta quieta) La cámara ha visto la plataforma en `q` (posición estimada + medida RELATIVA).
        La puerta pasa a la media de lo visto pesada por 1/σ²: queda en el mismo marco que la posición estimada del
        dron, así que el error del GPS deja de importar. Con el GPS del Mini (~1 m de error) casi la mitad de las
        carreras rozaban la puerta de 1 m sin cruzarla y tenían que volver."""
        up = 1.0 if self.world.goal_support else 0.0
        g = np.asarray(q, float) + np.array([0.0, 0.0, up])
        if np.linalg.norm(g[:2] - self.gate_est[:2]) > 3.0:     # no es la plataforma
            return
        self.gate_hits.append((g, 1.0 / (0.02 * float(np.linalg.norm(q - est_p)) + 0.1) ** 2))
        W = np.array([w for _, w in self.gate_hits])
        new = (np.array([h for h, _ in self.gate_hits]) * W[:, None]).sum(axis=0) / W.sum()
        moved = float(np.linalg.norm(new - self.gate_est))
        self.gate_est = new
        self.pad_est, self.pad_z_est = new[:2].copy(), float(new[2] - up)
        if moved > 0.3 and self.phase == "carrera" and t - self.gate_fix_t > 0.4:
            self.gate_fix_t = t
            self._plan(t, est_p, stitch=True)

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
        self.speed_now = min(float(np.linalg.norm(est_v)), prof.v_max)
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
                # (posado, el suelo está por debajo de la distancia mínima del telémetro, que entonces no da eco y
                # mide su alcance máximo: el dron creía tener el suelo 15 m más abajo, "despegaba" al instante y
                # buscaba a 1 m del suelo sin ver nada. Sin eco, el suelo es el de debajo)
                down = [x["dist"] for x in readings.get("rays", []) if x["label"] == "down" and x["dist"] < self.range - 0.1]
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
        if self.camera_only and self.phase in ("carrera", "reintento", "persecución") and t - self.last_seen_t > 2.5:
            # perdido = debería verlo (está cerca de donde lo predice) y no lo ve: se ha escondido. De lejos, la
            # cámara no alcanza: sigue volando hacia la predicción
            p_t, _ = self._track(t)
            if np.linalg.norm(p_t - est_p[:2]) < 25.0:
                self._lost(t, est_p)
        if "gate_seen" in readings and self.gate_est is not None and not self.moving:
            self._gate_seen(t, est_p, readings["gate_seen"])
        if self.mode == "carrera" and t - self.sprint_t > 0.2:   # ¿ve la meta? (5 veces por segundo)
            self.sprint_t = t
            s = self._sprint_check(t, est_p)
            self.ram = self.ram_on and s and self.ram_clear
            if s != self.sprint:
                self.sprint = s
                if self.phase == "carrera" and self.traj is not None and self.blocked is None:
                    self._plan(t, est_p, stitch=True)   # a la nueva velocidad
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
                # velocidad de cierre. A por todas: la mayor desde la que aún puede frenar hasta la del vehículo justo
                # en la puerta, √(2·a·d) con la mitad de la aceleración (perfil de tiempo mínimo). Sin frenar nada se
                # pasaba de largo, lo perdía de vista y lo capturaba menos (82 % frente a 95 %, 40 semillas).
                # Si no, la de antes: proporcional a la distancia.
                if self.ram:
                    # embestida: no frena para igualar la velocidad del vehículo en la puerta; la navegación
                    # proporcional lo lleva a chocar con él (impactar también es llegar)
                    close = self._race_v()
                elif self.all_out:
                    close = min(self._race_v(), max(2.0, math.sqrt(PURSUIT_K * prof.acc_hor * dist)))
                else:
                    close = min(self.v_sense, max(2.0, 0.8 * dist))
                v_ref = np.array([v_t[0], v_t[1], 0.0]) + d / max(dist, 1e-6) * close
                # velocidad pedida SUAVIZADA (como el generador de consignas de PX4, MPC_ACC_HOR_MAX): cambia como
                # mucho a la aceleración que el dron puede dar. Pidiendo de golpe 14 m/s a un Matrice que iba a 2, el
                # control se saturaba a la inclinación máxima y no le quedaba fuerza lateral: se deslizaba contra un
                # arbusto o el borde
                # (solo en horizontal: en vertical manda el lazo de posición; con la rampa también en z seguía
                # bajando en picado tras la carrera y se estrellaba contra el terreno)
                dt = t - self.pursuit_t
                recent = self.pursuit_v is not None and 0.0 < dt < 0.5
                base = self.pursuit_v[:2] if recent else np.asarray(est_v[:2], float)   # al entrar: la que lleva
                step = (self.ram_acc() if self.ram else max(prof.acc_max, prof.acc_hor)) * (dt if recent else 0.05)
                dv = v_ref[:2] - base
                ndv = float(np.linalg.norm(dv))
                if ndv > step:
                    v_ref[:2] = base + dv * (step / ndv)
                # y nunca hacia el borde más rápido de lo que puede frenar antes de él: el vehículo, pegado al
                # borde, "huía" hacia fuera y la velocidad pedida lo seguía (choque contra el borde, semilla 3017).
                # Con 2 sigmas del GPS de margen: el dron solo sabe dónde está con ese error (el Mini, ±0,75 m, creía
                # estar a 1,6 m de donde estaba y rozaba el borde, Mini "rápido" semillas 3008 y 3012)
                lim = 1.0 + prof.radius + 2.0 * prof.gps_sigma
                for i, top in ((0, LENGTH), (1, WIDTH)):
                    v_hi = math.sqrt(2.0 * prof.acc_hor * max(top - lim - est_p[i], 0.0))
                    v_lo = math.sqrt(2.0 * prof.acc_hor * max(est_p[i] - lim, 0.0))
                    v_ref[i] = min(max(v_ref[i], -v_lo), v_hi)
                # ni hacia la altura de la puerta más rápido de lo que puede frenar (bajando a 5 m/s desde la carrera
                # se pasaba de la altura y tocaba el suelo, semilla 3002). Con 0,2 g: con 0,3 g el Matrice bajaba a 9 m/s
                # y no frenaba antes de los arbustos; con 0,15 g la persecución en equipo tardaba 141 s en vez de <120
                v_ref[2] = math.copysign(min(abs(v_ref[2]), math.sqrt(2.0 * 0.2 * G * abs(d[2]))), d[2])
                self.pursuit_t, self.pursuit_v = t, v_ref.copy()
                if math.hypot(v_ref[0], v_ref[1]) > 1.0:
                    self.yaw = math.atan2(v_ref[1], v_ref[0])
                # la consigna de posición, a como mucho CARROT (3 m) en horizontal (como el "carrot" de PX4): con el vehículo
                # lejos, kp × (distancia) se sumaba a la velocidad pedida, la anulaba (rampa y límite del borde
                # incluidos) y el control se saturaba hacia el vehículo: chocaba contra el borde (semillas 3016, 3036)
                p_ref = aim.copy()
                off = aim[:2] - est_p[:2]
                n = float(np.linalg.norm(off))
                if n > CARROT:
                    p_ref[:2] = est_p[:2] + off * (CARROT / n)
                # y en vertical a ≤ 1 m: con el vehículo 9 m más abajo, kp × 9 m pedía bajar a 9 m/s (el doble de
                # v_down en carrera) y la rampa de frenada de arriba no servía: tocaba el suelo (Matrice, semilla 5006)
                p_ref[2] = est_p[2] + min(max(aim[2] - est_p[2], -1.0), 1.0)
                return p_ref, v_ref, np.zeros(3), self.yaw
            if self.phase == "persecución":  # se ha alejado: volver a interceptar
                self._plan(t, est_p)
            elif t - self.last_plan_t > 1.0:  # el punto de intercepción cambia: replanificar cada segundo
                p_t, v_t = self._track(t)
                xy, _ = intercept_point(est_p, p_t, v_t, 0.8 * self._race_v())
                if self.intercept is None or np.linalg.norm(xy - self.intercept[:2]) > 2.0 or self.phase == "reintento":
                    self._plan(t, est_p, stitch=True)
                else:
                    self.last_plan_t = t
        if self.dash and self.phase == "reintento":
            # llegó a la detección sin confirmarla: no era la plataforma (encima ya no la ve la cámara)
            self.search.candidate["state"] = "falsa"
            self.search.candidate = None
            self.dash = self.refine = False
            self.phase = "búsqueda"
            self._next_search(t, est_p)
        if self.search is not None and (self.phase in ("búsqueda", "confirmación") or self.dash or self.refine):
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
                and not self.reactive and t - self.check_t > 0.2 and self.map.version != self.map_checked:
            self.check_t, self.map_checked = t, self.map.version
            if not self._still_clear(t, est_p):
                self.replans += 1
                self.map_replans += 1
                need = self.prof.radius + 0.5 * self.P["margin"]
                if self.phase == "búsqueda" and self.search_goal is not None and not self.camera_only \
                        and float(self.map.grid().clearance_many(self.search_goal[None])[0]) < need:
                    # el punto de búsqueda ha quedado pegado a algo recién visto (una copa): es solo un sitio desde el
                    # que mirar, así que se elige otro. Replanificando hacia él, el dron se quedaba dando vueltas
                    # alrededor (500 replanificaciones en un vuelo) sin llegar nunca
                    self.search.reject(self.search_goal)
                    if not self._next_search(t, est_p):
                        return self.blocked, zero, zero, self.yaw
                elif not self._plan(t, est_p, stitch=True):
                    self.blocked, self.blocked_t = est_p.copy(), t
                    return self.blocked, zero, zero, self.yaw
        if self.phase == "carrera" and not self.moving and self.sprint and self.terminal_on \
                and self.gate_est is not None:
            # GUÍA TERMINAL (carrera a una meta quieta que ve): en los últimos metros deja la trayectoria y apunta
            # directamente a la puerta a la velocidad que lleva (persecución pura, como los drones de carreras
            # autónomos). A 12 m/s el dron se quedaba 1-2 m fuera de la trayectoria al final y rozaba la puerta de
            # 1 m sin cruzarla. Si ya la ha dejado atrás, vuelve a por ella
            d = self.gate_est - est_p
            dist = float(np.linalg.norm(d))
            # a la velocidad de CARRERA, no a la que lleva: con la que llevaba, si bajaba un poco (el picado, la
            # inclinación máxima) la referencia bajaba con ella y el dron cruzaba la meta a 2-3 m/s tras ir a 10
            spd = max(self.speed_now, self._race_v(), 3.0)
            if dist < max(5.0, self.P["term"] * spd) or self.ram:
                if dist > 0.5 and float(d @ est_v) < 0:
                    # la ha pasado de largo: trayectoria de vuelta (esquivando obstáculos), no "pararse en la puerta":
                    # a 10 m/s, con la referencia quieta en la puerta, un Matrice siguió 4 m por inercia y chocó
                    if t - self.retry_t > 1.0:
                        self.retry_t = t
                        self._plan(t, est_p)
                else:
                    u = d / max(dist, 1e-6)
                    # la curva que queda hasta la puerta tiene que caber en lo que el dron puede girar: aceleración
                    # lateral necesaria ≈ 2·v²·sen θ / d (el arco que llega a la puerta); si pasa de la que da su
                    # inclinación máxima, velocidad justa para tomarla (un Matrice de 30° a 9 m/s, con ~14 m de radio
                    # de giro mínimo, pasaba a 1,2 m de la puerta y chocaba después con lo que hubiera detrás)
                    sp_now = float(np.linalg.norm(est_v))
                    if sp_now > 1.0:
                        sin_t = min(1.0, float(np.linalg.norm(np.cross(est_v / sp_now, u))))
                        a_lat = 0.9 * (self.ram_acc() if self.ram else G * math.tan(prof.tilt_max))
                        spd = min(spd, max(3.0, math.sqrt(a_lat * dist / (2 * max(sin_t, 0.05)))))
                    v_ref = u * spd
                    # NAVEGACIÓN PROPORCIONAL (la ley de guiado de los interceptores): aceleración = N · velocidad de
                    # cierre · giro de la línea de visión, perpendicular a ella. Con la persecución pura (apuntar a
                    # donde está la puerta) un Matrice a 9 m/s la pasaba a 1,5 m, sin tiempo para corregir
                    vc = float(est_v @ u)
                    los_rate = np.cross(d, -est_v) / max(dist * dist, 1e-6)
                    a_pn = PN_N * max(vc, 0.0) * np.cross(los_rate, u)
                    if math.hypot(v_ref[0], v_ref[1]) > 1.0:
                        self.yaw = math.atan2(v_ref[1], v_ref[0])
                    return est_p.copy(), v_ref, a_pn, self.yaw
        if self.phase in ("crucero", "carrera", "búsqueda", "confirmación") and self.reactive:
            p, v, a = self._reactive(t, est_p, est_v, readings)
        elif self.phase in ("crucero", "carrera", "búsqueda", "confirmación"):
            p, v, a = self.traj.sample(t - self.t0)
            if PROGRESS:
                # VARIABLE DE PROGRESO (como el control de contorno, MPCC: Lam et al. 2010, Romero et al. 2022): si el
                # dron va por detrás de la referencia (la anticolisión lo ha frenado, la inercia de un giro), el reloj
                # de la trayectoria se ralentiza en vez de dejar que la referencia se escape. Antes, a 3 m de
                # distancia se replanificaba desde el estado del dron, casi parado (caída media de la velocidad
                # pedida de ~2 m/s en esas replanificaciones)
                dt = t - getattr(self, "_ref_t", t)
                self._ref_t = t
                sp = float(np.linalg.norm(v))
                if 0.0 < dt < 0.1 and sp > 0.5:
                    ahead = float((p - est_p) @ v) / sp          # cuánto va la referencia por delante, en su dirección
                    rate = min(1.0, max(PROGRESS_MIN, 1.0 - (ahead - PROGRESS_E0) / PROGRESS_E1))
                    if rate < 1.0:
                        self.t0 += dt * (1.0 - rate)
                        p, v, a = self.traj.sample(t - self.t0)
        if self.phase in ("crucero", "carrera", "búsqueda", "confirmación"):
            if not self.reactive and np.linalg.norm(p - est_p) > 3.0:  # desviado: replanificar desde aquí
                self.replans += 1
                if self._plan(t, est_p):
                    p, v, a = self.traj.sample(0.0)
            if math.hypot(v[0], v[1]) > 1.0:
                self.yaw = math.atan2(v[1], v[0])
            elif self.phase == "búsqueda" and getattr(self.search, "_center", None) is not None:
                # casi parado buscando: la cámara (hacia delante y abajo) mira a la zona que quería ver, no a donde
                # venía volando
                c = self.search._center
                if math.hypot(c[0] - est_p[0], c[1] - est_p[1]) > 1.0:
                    self.yaw = math.atan2(c[1] - est_p[1], c[0] - est_p[0])
            # (el reactivo no sigue el perfil de velocidad de la recta: termina el tramo solo al LLEGAR. Antes, al
            # agotarse la duración de la recta a 23 m de la meta, pasaba a "reintento", iba en línea recta contra
            # el obstáculo de delante, Collision Prevention lo paraba y se quedaba ahí hasta agotar el tiempo)
            if (self.react_arrived if self.reactive else t - self.t0 >= self.traj.duration):
                if self.phase == "búsqueda":           # punto alcanzado: el siguiente
                    self._next_search(t, est_p)
                elif self.phase == "confirmación":     # encima de la detección: quieto, mirándola
                    c = self.search.candidate
                    if c is not None:
                        self.yaw = math.atan2(c["pos"][1] - est_p[1], c["pos"][0] - est_p[0])
                    return self.traj.P[-1].copy(), zero, zero, self.yaw
                else:
                    if self.mode == "carrera" and not self.moving and not self.reactive and t - self.retry_t > 1.0:
                        # carrera a una meta quieta que no ha cruzado: trayectoria de vuelta (esquivando), no una
                        # referencia quieta en la puerta (con inercia, el dron seguía de largo contra lo que hubiera)
                        self.retry_t = t
                        if self._plan(t, est_p):
                            return p, v, a, self.yaw
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
            if ((err < 0.3 and np.linalg.norm(est_v) < 0.4) or t - self.t0 > 6.0) and not self.dash:
                # (confirmando en vuelo, no baja hasta confirmarla: encima, la cámara aún la ve)
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
