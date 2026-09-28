"""AUTOTUNE del autopiloto de cada dron con un algoritmo genético, sobre maniobras en campo abierto (eval/maniobras.py),
como el autotune de PX4 o el ajuste de PID que hace un piloto en un vuelo de prueba.

Genoma: factores sobre las ganancias de fábrica de actitud (att_k) y de velocidad angular P/I/D (rate_pk, rate_ik,
rate_dk) del autopiloto interno, y sobre las de posición y velocidad horizontal (xy_p_k, vel_p_k).

Coste (menos es mejor), sobre 3 drones "reales" con la dinámica aleatorizada (empuje ±20 %, retardos de los motores
±30 %, inercias ±20 %: el ajuste tiene que valer aunque el dron no sea exactamente el del modelo):
    tiempo en estabilizarse tras un escalón de 10 m + 2 × lo que se pasa + 3 × error RMS en el círculo
    + 0,3 × tiempo del giro de 180° + 50 si choca o no se estabiliza
Mismo GA que eval/evolve.py (torneo, BLX-α, mutación gaussiana decreciente, élite). Al final, validación en otros 3
drones aleatorizados: se adopta (dron/autotune.json) solo si mejora.

    python eval/autotune.py --profile px4 --jobs 4
"""
import argparse
import dataclasses
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import maniobras as M  # noqa: E402
from dron.params import PROFILES  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "dron", "autotune.json")
RANGES = {"att_k": (0.6, 1.8), "rate_pk": (0.5, 2.5), "rate_ik": (0.3, 2.0), "rate_dk": (0.0, 3.0),
          "xy_p_k": (0.6, 2.5), "vel_p_k": (0.6, 2.0)}


def variant(key, seed):
    """El dron "real": el del perfil con empuje, retardos e inercias aleatorios (reproducibles por semilla)."""
    r = np.random.default_rng(seed)
    p = PROFILES[key]
    k = float(r.uniform(0.7, 1.3))
    return dataclasses.replace(p, twr=p.twr * float(r.uniform(0.8, 1.2)), tau_up=p.tau_up * k, tau_down=p.tau_down * k,
                               inertia=tuple(i * float(r.uniform(0.8, 1.2)) for i in p.inertia))


def cost(job):
    key, genes, seed = job
    prof = variant(key, seed)
    old = M.PROFILES[key]
    M.PROFILES[key] = prof                     # las maniobras usan PROFILES[key]
    try:
        s = M.m_step(key, "6dof", genes)
        c = M.m_circle(key, "6dof", genes)
        y = M.m_yaw(key, "6dof", genes)
    except (StopIteration, ValueError):
        return 100.0
    finally:
        M.PROFILES[key] = old
    if s["estable"] is None or y["giro_180"] is None or not np.isfinite(c["error_rms"]):
        return 50.0 + (s["se_pasa"] if s["se_pasa"] is not None else 0)
    return s["estable"] + 2 * s["se_pasa"] + 3 * c["error_rms"] + 0.3 * y["giro_180"]


def evaluate(pool, key, pop, seeds):
    jobs = [(key, g, sd) for g in pop for sd in seeds]
    res = list(pool.map(cost, jobs))
    n = len(seeds)
    return [statistics.mean(res[i * n:(i + 1) * n]) for i in range(len(pop))]


def clip(g):
    return {k: round(min(max(v, RANGES[k][0]), RANGES[k][1]), 3) for k, v in g.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="x650")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(5)
    base = {k: 1.0 for k in RANGES}
    prev = None   # las ganancias ya adoptadas: también en la población inicial y en la validación
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f).get(a.profile, {}).get("genes")
    pop = [base] + ([clip(prev)] if prev else [])
    pop += [clip({k: v * rng.uniform(0.7, 1.4) for k, v in base.items()}) for _ in range(a.pop // 3)]
    pop += [clip({k: rng.uniform(*RANGES[k]) for k in RANGES}) for _ in range(a.pop - len(pop))]
    seeds = [101, 102, 103]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as pool:
        for g in range(a.gens):
            fit = evaluate(pool, a.profile, pop, seeds)
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            best, bf = pop[order[0]], fit[order[0]]
            print("generación %2d: mejor coste %.3f · media %.3f · %s · %.0f s" % (
                g + 1, bf, statistics.mean(fit), json.dumps(best), time.time() - t0), flush=True)
            elite = [pop[i] for i in order[:2]]
            sigma = 0.12 * (1 - 0.6 * g / max(a.gens - 1, 1))

            def tour():
                idx = rng.sample(range(len(pop)), 3)
                return pop[min(idx, key=lambda i: fit[i])]
            kids = []
            while len(kids) < a.pop - 2:
                p1, p2 = tour(), tour()
                c = {}
                for k in RANGES:
                    lo, hi = min(p1[k], p2[k]), max(p1[k], p2[k])
                    c[k] = rng.uniform(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
                    if rng.random() < 0.3:
                        c[k] += rng.gauss(0, sigma * (RANGES[k][1] - RANGES[k][0]))
                kids.append(clip(c))
            pop = elite + kids
        vb, vt = evaluate(pool, a.profile, [base, best], [201, 202, 203])
        vp = evaluate(pool, a.profile, [clip(prev)], [201, 202, 203])[0] if prev else float("inf")
    print("validación (otros 3 drones aleatorizados): fábrica %.3f · anterior %.3f · autotune %.3f" % (vb, vp, vt))
    tuned = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            tuned = json.load(f)
    if vt < min(vb, vp):
        tuned[a.profile] = {"genes": best, "coste_fabrica": vb, "coste_autotune": vt}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(tuned, f, indent=2, ensure_ascii=False)
        print("ADOPTADO → dron/autotune.json")
    else:
        print("no mejora: se queda con las %s" % ("anteriores" if prev and vp <= vb else "de fábrica"))
    print("(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
