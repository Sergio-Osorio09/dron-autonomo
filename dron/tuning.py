"""Parámetros de seguridad y velocidad, POR DRON (como en PX4, donde cada aparato se ajusta por separado).

  cp_acc      fracción de la aceleración con la que frena Collision Prevention
  cp_margin   metros de seguridad de Collision Prevention, además del radio
  cp_react    segundos de reacción (el margen crece con la velocidad HACIA el obstáculo)
  margin      holgura mínima del planificador
  sigma_k     cuántas σ de incertidumbre de posición se suman a esa holgura
  sense       metros que se descuentan al alcance de los sensores para la velocidad máxima
  cruise      factor sobre la velocidad de crucero del perfil

FACTORY son los valores prudentes elegidos a mano en las fases 1-6. TUNED, los que encontró `eval/tune.py` para cada
dron en mapas densos y con viento (y validados en semillas nuevas); ver la lección correspondiente en CONTEXTO.md.
DRON_TUNED=0 en el entorno vuelve a los de fábrica (para comparar).
"""
import os

FACTORY = {"cp_acc": 0.5, "cp_margin": 0.8, "cp_react": 0.3, "margin": 0.6, "sigma_k": 2.0, "sense": 1.0,
           "cruise": 1.0}
# eval/tune.py (25-09-2026): 32 configuraciones × 24 vuelos por dron en bosque de densidad máxima, almacén normal y
# estrecho, ciudad densa, mixto con mapa desconocido y viento de 8 m/s con ráfagas; validado en 60 vuelos con
# semillas nuevas: Mini 28,9 → 24,7 s, PX4 36,7 → 33,7 s, Matrice 27,4 → 22,7 s, sin perder éxito ni añadir choques.
# Lo que aprende: frenar más fuerte (90 % en vez de 50 %), menos margen de reacción y de seguridad, más crucero.
TUNED = {
    "mini":    {"cp_acc": 0.9, "cp_margin": 0.41, "cp_react": 0.10, "margin": 0.45, "sigma_k": 1.75, "sense": 1.15,
                "cruise": 1.6},
    "px4":     {"cp_acc": 0.89, "cp_margin": 0.47, "cp_react": 0.20, "margin": 0.47, "sigma_k": 1.34, "sense": 1.62,
                "cruise": 1.43},
    "matrice": {"cp_acc": 0.9, "cp_margin": 0.40, "cp_react": 0.10, "margin": 0.51, "sigma_k": 2.0, "sense": 1.47,
                "cruise": 1.8},
}
OVERRIDE = {}   # eval/tune.py pone aquí la configuración que está probando ({"*": {...}} o por perfil)


def params(profile_key: str) -> dict:
    p = dict(FACTORY)
    if os.environ.get("DRON_TUNED", "1") != "0":
        p.update(TUNED.get(profile_key, {}))
    p.update(OVERRIDE.get("*", {}))
    p.update(OVERRIDE.get(profile_key, {}))
    return p
