"""Ajuste automático de parámetros (sin redes neuronales): búsqueda de los parámetros de seguridad y velocidad que
dan más éxito, cero choques y menos tiempo en mapas DENSOS (bosque de densidad máxima, almacén, ciudad densa) y
con VIENTO fuerte. Se ajusta CADA DRON POR SEPARADO (como en PX4): unos parámetros ajustados con el Mini y el PX4
hicieron chocar al Matrice, que es más pesado y más rápido. El resultado va a dron/tuning.py (TUNED).

Qué se ajusta (hoy son valores prudentes elegidos a mano):
  los de dron/tuning.py: cp_acc, cp_margin, cp_react (Collision Prevention), margin y sigma_k (holgura del
  planificador), sense (velocidad máxima según el alcance de los sensores) y cruise (factor de crucero).

Puntuación de un vuelo: éxito → 1 − tiempo/150 (más rápido, más puntos); choque → −3 (un choque pesa más que tres
éxitos); fuera de la plataforma o sin tiempo → 0. Método: búsqueda aleatoria + refinamiento local alrededor del
mejor (una versión sencilla de lo que hacen los ajustadores de hiperparámetros). Para no sobreajustar, al final se
compara con los valores de fábrica en semillas que el ajuste NO ha visto.

    python eval/tune.py --profile matrice --rounds 20 --refine 12 --seeds 4 --jobs 18
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dron.tuning import FACTORY as DEFAULTS  # noqa: E402
RANGES = {"cp_acc": (0.35, 0.9), "cp_margin": (0.4, 1.1), "cp_react": (0.1, 0.5), "margin": (0.35, 0.8),
          "sigma_k": (1.0, 3.0), "sense": (0.3, 2.0), "cruise": (0.8, 1.8)}
SCENARIOS = [dict(level="bosque", density="máxima", wind_speed=4, gusts=1),
             dict(level="almacen"),
             dict(level="almacen", density="máxima"),
             dict(level="ciudad", density="alta", terrain="colinas", wind_speed=6, gusts=1),
             dict(level="mixto", density="alta", wind_speed=6, gusts=1, map_mode="desconocido"),
             dict(level="mixto", wind_speed=8, gusts=2, noise="realista")]
PROFILES_TUNE = ("px4",)


def apply(params):
    """Pone los parámetros que se prueban (en el proceso que va a volar)."""
    from dron import tuning
    tuning.OVERRIDE = {"*": dict(params)}


def fly(job):
    params, profile, scen, seed = job
    apply(params)
    from dron.sim import Simulation
    s = Simulation(profile=profile, seed=seed, **scen)
    while not s.done:
        s.step(1.0)
    score = (1 - s.t / 150.0) if s.status == "success" else (-3.0 if s.status == "crash" else 0.0)
    return score, s.status, s.t


def evaluate(pool, params, seeds):
    jobs = [(params, p, sc, sd) for p in PROFILES_TUNE for sc in SCENARIOS for sd in seeds]
    res = list(pool.map(fly, jobs))
    ok = [t for _, st, t in res if st == "success"]
    return {"score": statistics.mean(r[0] for r in res), "success": len(ok) / len(res),
            "crash": sum(st == "crash" for _, st, _ in res), "time": statistics.mean(ok) if ok else None, "n": len(res)}


def sample(rng, around=None, scale=0.25):
    out = {}
    for k, (lo, hi) in RANGES.items():
        if around is None:
            v = rng.uniform(lo, hi)
        else:
            v = around[k] + rng.gauss(0, scale * (hi - lo))
        out[k] = round(min(max(v, lo), hi), 3)
    return out


def fmt(r):
    return "puntos %.3f · éxito %.0f %% · choques %d · tiempo %s (%d vuelos)" % (
        r["score"], 100 * r["success"], r["crash"], "%.1f s" % r["time"] if r["time"] else "—", r["n"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24, help="configuraciones al azar")
    ap.add_argument("--refine", type=int, default=12, help="configuraciones alrededor de la mejor")
    ap.add_argument("--seeds", type=int, default=4, help="semillas por escenario y perfil durante el ajuste")
    ap.add_argument("--valid", type=int, default=10, help="semillas NUEVAS para la validación final")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=None, help="guardar la mejor configuración (JSON)")
    ap.add_argument("--profile", default="px4", help="dron que se ajusta (cada uno por separado)")
    a = ap.parse_args()
    global PROFILES_TUNE
    PROFILES_TUNE = (a.profile,)
    rng = random.Random(7)
    seeds = list(range(9000, 9000 + a.seeds))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        base = evaluate(pool, DEFAULTS, seeds)
        print("valores de fábrica:", fmt(base), flush=True)
        best, best_r = dict(DEFAULTS), base
        for i in range(a.rounds + a.refine):
            cand = sample(rng) if i < a.rounds else sample(rng, best, 0.12)
            r = evaluate(pool, cand, seeds)
            tag = ""
            if r["crash"] <= best_r["crash"] and r["score"] > best_r["score"]:
                best, best_r, tag = cand, r, "  ← mejor"
            print("%2d %s %s%s" % (i + 1, json.dumps(cand), fmt(r), tag), flush=True)
        print("\nmejor en el ajuste:", json.dumps(best), fmt(best_r))
        vseeds = list(range(9500, 9500 + a.valid))           # validación: semillas que el ajuste no ha visto
        vb, vt = evaluate(pool, DEFAULTS, vseeds), evaluate(pool, best, vseeds)
        print("validación (semillas nuevas):\n  fábrica  %s\n  ajustado %s" % (fmt(vb), fmt(vt)))
    print("(%.0f s)" % (time.time() - t0))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"best": best, "tuning": best_r, "valid_default": vb, "valid_tuned": vt}, f, indent=2)


if __name__ == "__main__":
    main()
