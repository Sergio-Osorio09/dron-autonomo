"""Algoritmos de pilotaje que se pueden elegir (qué decide adónde y a qué velocidad va el dron).

Todos comparten la física, los sensores, el filtro de Kalman, el mapa, la misión (despegar, buscar, aterrizar...) y
el control en cascada de PX4 con Collision Prevention: lo que cambia es CÓMO se genera la referencia que sigue el
dron y con qué parámetros.

  clasico   Clásico ajustado: A* sobre el campo de distancias + estirado de cuerda + Chaikin + perfil de velocidad
            óptimo en tiempo, replanificación cosida, sprint de carrera, confirmación en vuelo y guía terminal; con
            los parámetros ajustados por dron (tuning.py). El de siempre.
  fabrica   PX4 de fábrica: la misma planificación con los parámetros prudentes de fábrica y sin lo "agresivo":
            nunca pasa del límite de frenar dentro del alcance de los sensores (sin sprint), se para a confirmar las
            detecciones y no usa guía terminal ni la cámara de la puerta. Como un dron recién sacado de la caja.
  hibrido   Híbrido: el reactivo, pero en vez de ir hacia la meta en línea recta va hacia un "punto de mira" `look`
            metros por delante en un camino A* sobre el mapa (recalculado cada segundo), como la navegación de ROS
            (move_base: planificador global + planificador local). El reactivo puro se perdía en los pasillos del
            almacén (8 de 12 fallos del Mini en validación); con la guía global no hay mínimos locales grandes.
  (reactivo) Reactivo puro (campos de potencial, Khatib 1986: la meta atrae, los obstáculos repelen, rodea si se
            atasca). RETIRADO como piloto elegible (26-09-2026): en validación no pasaba del 81-85 % de éxito (se
            atasca en pasillos en U y callejones, un límite de diseño que no arregla ningún ajuste) y el genético no
            lo mejoraba. Su código sigue: es el planificador LOCAL del híbrido.

Para añadir otro piloto (p. ej. uno cuyos parámetros salgan de un algoritmo genético), basta con registrarlo en
PILOTS; `Mission` y `Simulation` leen de aquí lo que necesitan.
"""
import json
import os
from dataclasses import dataclass, field, replace
from typing import Dict


@dataclass(frozen=True)
class Pilot:
    key: str
    name: str
    description: str
    tuned: bool = True            # parámetros ajustados por dron (tuning.py) o los de fábrica
    planner: str = "astar"        # "astar" (planificación global) o "reactivo" (campos de potencial)
    sprint: bool = True           # carrera: a la velocidad máxima del fabricante al ver la meta
    confirm_in_flight: bool = True
    all_out: bool = True          # carrera: confirmación en vuelo y persecución a tope
    terminal: bool = True         # carrera: guía terminal hacia la puerta
    gate_camera: bool = True      # carrera: cámara en gimbal que afina la puerta
    gains: Dict[str, float] = field(default_factory=dict)
    # parámetros ENTRENADOS para la misión de carrera, por dron (eval/evolve.py, algoritmo genético): pueden cambiar
    # los de tuning.py (cp_acc, cp_margin, cp_react, margin, sigma_k, sense), los de carrera (race: fracción de la
    # velocidad máxima sin ver la meta; corridor: medio ancho extra del pasillo del sprint; term: segundos de guía
    # terminal) y las ganancias del reactivo
    carrera: Dict[str, Dict[str, float]] = field(default_factory=dict)


# Ganancias del piloto reactivo (elegidas a mano; son las que ajustaría un algoritmo genético):
# (26-09-2026: k_rep y slow a 1/10. Antes las lecturas de los telémetros solo se usaban en 1 de cada 10 decisiones,
# así que la repulsión y la frenada actuaban a pulsos, un 10 % del tiempo; ahora actúan siempre, con 1/10 de fuerza:
# el mismo efecto medio, sin pulsos. Híbrido del X650, 60 semillas: mixto 11,4 → 11,3 s, bosque 15,1 → 14,7 s)
REACTIVE_GAINS = {
    "k_rep": 0.6,      # fuerza de repulsión (m/s por (1/d − 1/R), acotada)
    "radius": 6.0,     # m: más lejos, un obstáculo no repele (R)
    "k_side": 0.6,     # peso de los obstáculos laterales (los de delante pesan 1)
    "slow": 0.035,      # cuánto frena ante un obstáculo delante (fracción de la velocidad por m dentro de R)
    "stuck_t": 2.5,    # s sin acercarse a la meta = atascado: rodea el obstáculo
    "k_tan": 0.8,      # fracción de la velocidad que rodea el obstáculo cuando está atascado
    "esc_t": 6.0,      # s como mucho rodeando el obstáculo (acaba antes si ve libre la dirección de la meta)
}

# lo que pone eval/evolve.py mientras prueba individuos: {(piloto, dron): {parámetro: valor}}
OVERRIDE: Dict[tuple, Dict[str, float]] = {}
# valores de carrera sin entrenar (los de antes del algoritmo genético)
# vs_k: factor sobre la velocidad que permite frenar dentro del alcance de los sensores (1 = respetarla);
# acc_k: factor sobre la aceleración del planificador (hasta el 90 % de la física, g·tan(inclinación máxima))
# att_k, rate_pk/ik/dk: factores sobre las ganancias de actitud y de velocidad angular del autopiloto interno;
# xy_p_k, vel_p_k: sobre las de posición y velocidad horizontal (MPC_XY_P, MPC_XY_VEL_P_ACC)
RACE_DEFAULTS = {"race": 0.9, "corridor": 0.6, "term": 0.7, "vs_k": 1.0, "acc_k": 1.0,
                 "att_k": 1.0, "rate_pk": 1.0, "rate_ik": 1.0, "rate_dk": 1.0, "xy_p_k": 1.0, "vel_p_k": 1.0}

PILOTS: Dict[str, Pilot] = {
    "clasico": Pilot("clasico", "Clásico ajustado (A* + PX4)",
                     "Planificación global sobre el mapa, perfil de velocidad óptimo y parámetros ajustados por dron"),
    "fabrica": Pilot("fabrica", "PX4 de fábrica (prudente)",
                     "La misma planificación con los parámetros de fábrica: nunca más rápido de lo que ven sus sensores",
                     tuned=False, sprint=False, confirm_in_flight=False, all_out=False, terminal=False,
                     gate_camera=False),
    "hibrido": Pilot("hibrido", "Híbrido (A* global + reactivo local)",
                     "A* sobre el mapa marca un punto de mira unos metros por delante; los campos de potencial "
                     "esquivan lo que haya (como la navegación de ROS: planificador global + local)",
                     planner="hibrido", gains=dict(REACTIVE_GAINS, look=4.0)),
}


def _load_trained():
    """Parámetros entrenados y VALIDADOS por el algoritmo genético (eval/apply_evolved.py → dron/entrenados.json)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entrenados.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key, per_drone in data.items():
        if key in PILOTS:
            PILOTS[key] = replace(PILOTS[key], carrera={d: dict(v["params"]) for d, v in per_drone.items()})


_load_trained()


def race_params(pilot: Pilot, profile_key: str) -> Dict[str, float]:
    """Parámetros de carrera de un piloto para un dron: los entrenados (si los hay) y, encima, los que se prueban."""
    p = dict(pilot.carrera.get(profile_key, {}))
    p.update(OVERRIDE.get((pilot.key, profile_key), {}))
    return p


def get(key: str) -> Pilot:
    if key not in PILOTS:
        raise ValueError("Piloto desconocido %r (hay: %s)" % (key, ", ".join(PILOTS)))
    return PILOTS[key]
