"""Entrenamiento con un ALGORITMO GENÉTICO de los pilotos para la misión de CARRERA (sin redes neuronales).

Inspirado en cómo se entrenan los drones de carreras autónomos (ver docs/ENTRENAMIENTO.md, con las fuentes):

* Genoma: los parámetros de un piloto (dron/pilots.py) para un dron: frenada y márgenes de Collision Prevention, a qué
  fracción de la velocidad máxima corre sin ver la meta (race), cuánto puede superar el límite de "frenar dentro de
  lo que ven los sensores" (vs_k), con cuánta aceleración planifica (acc_k, hasta el 90 % de la física), el ancho del
  pasillo del sprint, la guía terminal y, en el reactivo, sus 7 ganancias.
* Aptitud por vuelo, como la recompensa de Swift (Kaufmann et al., Nature 2023): el PROGRESO hacia la meta y, al
  llegar, la rapidez; un choque resta 5 (la penalización de Swift) y acaba el intento:
      llega → 1 + (1 − tiempo/60) · no llega en 45 s → fracción del camino recorrida · choca → progreso − 5
  Con "llega o no llega" a secas, un individuo que se queda a 1 m valía lo mismo que uno que no despega, y el
  algoritmo no tenía por dónde mejorar (se estancó en la 2.ª generación).
* Aleatorización de la dinámica en CADA vuelo (lo que permite pasar de la simulación a un dron real; SimpleFlight
  2024: aleatorizar lo incierto, no lo que se mide): empuje ±20 % (la relación empuje/peso es estimada), arrastre
  ±20 %, retardos de motores y actitud ±30 % y viento de 0 a 4 m/s con ráfagas; el piloto no lo sabe.
* Escenarios de carrera con obstáculos, todos SIN conocer el mapa: bosque extremo, ciudad densa, precipicios,
  almacén, montaña con meta en el aire y carrera con BÚSQUEDA (tiene que identificar la meta).
* GA de codificación real: población inicial con los parámetros actuales (para no empeorar) y el resto al azar;
  torneo de 3, cruce BLX-α (α = 0,3), mutación gaussiana (prob. 0,3 por gen; σ del 12 % al 5 % del rango según
  avanza) y élite de 2. Todos vuelan LOS MISMOS mundos (con la misma aleatorización) en todas las generaciones: la
  simulación es determinista, así que la comparación es justa y un individuo repetido no se vuelve a volar. (Una
  primera versión cambiaba los mundos en cada generación con 1 por escenario: la aptitud dependía más del mundo que
  del individuo, seleccionaba suerte y el "entrenado" validó PEOR, 55 % frente a 62 %.)
* Validación en semillas que el entrenamiento NO ha visto: el mejor frente a los parámetros actuales. Solo se adopta
  si mejora sin chocar más (eval/apply_evolved.py).
* Optimizador (--opt): por defecto CMA-ES (Hansen, "The CMA Evolution Strategy: A Tutorial", 2016), que con
  parámetros continuos suele necesitar muchas menos evaluaciones que un genético: en vez de cruzar y mutar al azar,
  aprende la forma de la región buena (una gaussiana que se estira hacia donde mejora la aptitud y se estrecha al
  converger). Trabaja con los genes normalizados a [0, 1] según sus rangos realistas (se recortan a ellos) y parte
  de los parámetros actuales. --opt ga: el genético de antes.

    python eval/evolve.py --pilot hibrido --profile x650 --out eval/evo_hibrido_x650.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# límites REALISTAS: la reacción no baja de 0,1 s (latencia típica de percepción + cálculo + motores de un dron real)
# y no se supera en más de un 50 % el límite de frenar dentro de lo que ven los sensores. Con 0,05 s y 2×, el
# clásico del Matrice aprendió 0,06 s y 1,72×: validó sin choques pero chocaba un 5 % con ráfagas y precipicios
SAFETY = {"cp_acc": (0.4, 0.95), "cp_margin": (0.3, 1.0), "cp_react": (0.1, 0.4), "sense": (0.3, 2.0),
          "race": (0.8, 1.0), "vs_k": (1.0, 1.5), "acc_k": (1.0, 3.0), "corridor": (0.3, 1.2), "term": (0.4, 1.2),
          # "autotune" del autopiloto (física de 4 motores): factores sobre las ganancias de fábrica de actitud y
          # de velocidad angular (P, I, D) y sobre las de posición y velocidad (dominar la inercia)
          "att_k": (0.6, 1.6), "rate_pk": (0.5, 2.0), "rate_ik": (0.3, 2.0), "rate_dk": (0.0, 3.0),
          "xy_p_k": (0.6, 2.0), "vel_p_k": (0.6, 1.8)}
RANGES = {
    "clasico": dict(SAFETY, margin=(0.35, 0.8), sigma_k=(1.0, 3.0)),
    # el híbrido: los de seguridad + las 7 ganancias de su planificador local (campos de potencial) + el punto de mira
    "hibrido": dict(SAFETY, k_rep=(0.1, 1.5), radius=(3.0, 10.0), k_side=(0.1, 1.0), slow=(0.0, 0.08),
                    stuck_t=(0.8, 4.0), k_tan=(0.2, 1.5), esc_t=(2.0, 10.0), look=(2.0, 8.0)),
}
SCENARIOS = [dict(level="bosque", density="extrema"),
             dict(level="ciudad", density="alta", terrain="colinas"),
             dict(level="mixto", density="alta", terrain="precipicios"),
             dict(level="almacen"),
             dict(terrain="montañoso", goal_kind="aire"),
             dict(level="mixto", search="bayesiana")]
MAX_T = 45.0      # s: en carrera, más es no llegar (acorta el entrenamiento: sin esto, un atasco son 150 s de vuelo)
CRASH = 5.0       # penalización por choque (la de Swift)


def randomization(seed):
    """Dinámica y viento aleatorios pero reproducibles para cada mundo (misma semilla, misma aleatorización)."""
    r = np.random.default_rng(seed * 7 + 3)
    dyn = {"twr": float(r.uniform(0.8, 1.2)), "drag": float(r.uniform(0.8, 1.2)), "tau": float(r.uniform(0.7, 1.3))}
    wind = {"wind_speed": float(r.uniform(0, 4)), "wind_dir": float(r.uniform(0, 360)), "gusts": int(r.integers(0, 2))}
    return dyn, wind


def current(pilot, profile):
    """Los parámetros que usa hoy el piloto en carrera (el punto de partida)."""
    from dron import pilots, tuning
    p = pilots.get(pilot)
    from dron.sim import AUTOTUNE
    out = dict(tuning.params(profile, tuned=p.tuned))
    out.update(pilots.RACE_DEFAULTS)
    out.update(AUTOTUNE.get(profile, {}))       # las ganancias del autopiloto parten del autotune del dron
    out.update(p.gains)
    out.update(p.carrera.get(profile, {}))
    return {k: round(float(out[k]), 3) for k in RANGES[pilot]}


def fly(job):
    pilot, profile, params, scen, seed = job
    from dron import pilots
    pilots.OVERRIDE = {(pilot, profile): dict(params)}
    from dron.sim import Simulation
    dyn, wind = randomization(seed)
    kw = dict(wind, **scen)
    s = Simulation(profile=profile, seed=seed, mode="carrera", map_mode="desconocido", pilot=pilot, dyn_rand=dyn,
                   **kw)
    s.max_time = MAX_T
    goal = np.array(s.world.goal, float)
    d0 = max(float(np.linalg.norm(s.drone.p - goal)), 1.0)
    dmin = d0
    while not s.done:
        s.step(0.5)
        dmin = min(dmin, float(np.linalg.norm(s.drone.p - goal)))
    progress = max(0.0, (d0 - dmin) / d0)
    if s.status == "success":
        score = 1.0 + max(0.0, 1.0 - s.t / 60.0)
    else:
        score = progress - (CRASH if s.status == "crash" else 0.0)
    return score, s.status, s.t


_CACHE = {}


def evaluate(pool, pilot, profile, population, seeds):
    """Aptitud de cada individuo en los mismos mundos. Lo ya volado (mismo individuo y mismos mundos) no se repite."""
    key = lambda ind: (json.dumps(ind, sort_keys=True), tuple(seeds))
    todo = [ind for ind in {key(i): i for i in population}.values() if key(ind) not in _CACHE]
    jobs = [(pilot, profile, ind, sc, sd) for ind in todo for sc in SCENARIOS for sd in seeds]
    res = list(pool.map(fly, jobs, chunksize=1))
    n = len(SCENARIOS) * len(seeds)
    for i, ind in enumerate(todo):
        _CACHE[key(ind)] = res[i * n:(i + 1) * n]
    out = []
    for ind in population:
        r = _CACHE[key(ind)]
        ok = [t for _, st, t in r if st == "success"]
        out.append({"fit": statistics.mean(x[0] for x in r), "success": len(ok) / n,
                    "crash": sum(st == "crash" for _, st, _ in r), "time": statistics.mean(ok) if ok else None,
                    "n": n})
    return out


def clip(pilot, ind):
    return {k: round(min(max(v, RANGES[pilot][k][0]), RANGES[pilot][k][1]), 3) for k, v in ind.items()}


def random_ind(rng, pilot):
    return {k: round(rng.uniform(lo, hi), 3) for k, (lo, hi) in RANGES[pilot].items()}


def crossover(rng, pilot, a, b, alpha=0.3):
    child = {}
    for k in a:
        lo, hi = min(a[k], b[k]), max(a[k], b[k])
        d = hi - lo
        child[k] = rng.uniform(lo - alpha * d, hi + alpha * d)
    return clip(pilot, child)


def mutate(rng, pilot, ind, p=0.3, sigma=0.1):
    out = dict(ind)
    for k, (lo, hi) in RANGES[pilot].items():
        if rng.random() < p:
            out[k] = out[k] + rng.gauss(0, sigma * (hi - lo))
    return clip(pilot, out)


def tournament(rng, pop, fits, k=3):
    idx = rng.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fits[i]["fit"])]


def cmaes(pool, a, base, seeds, history, save):
    """CMA-ES (μ/μ_w, λ) con los ajustes estándar del tutorial de Hansen, maximizando la aptitud. λ = a.pop,
    a.gens generaciones. Devuelve el mejor individuo visto (en parámetros reales)."""
    keys = list(RANGES[a.pilot])
    lo = np.array([RANGES[a.pilot][k][0] for k in keys])
    hi = np.array([RANGES[a.pilot][k][1] for k in keys])
    to_ind = lambda x: clip(a.pilot, {k: float(v) for k, v in zip(keys, lo + np.clip(x, 0, 1) * (hi - lo))})
    n = len(keys)
    lam = max(4, a.pop)
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / float((w ** 2).sum())
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m = np.clip((np.array([base[k] for k in keys]) - lo) / (hi - lo), 0, 1)   # parte de los parámetros actuales
    sigma = 0.2
    C, pc, ps = np.eye(n), np.zeros(n), np.zeros(n)
    rs = np.random.default_rng(a.seed)
    t0 = time.time()
    best, best_r = base, evaluate(pool, a.pilot, a.profile, [base], seeds)[0]
    for g in range(a.gens):
        vals, vecs = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(vals, 1e-20))
        Z = rs.standard_normal((lam, n))
        Y = Z @ np.diag(D) @ vecs.T                       # y ~ N(0, C)
        X = np.clip(m + sigma * Y, 0, 1)                  # (reparación: dentro de los rangos)
        Y = (X - m) / sigma
        inds = [to_ind(x) for x in X]
        fits = evaluate(pool, a.pilot, a.profile, inds, seeds)
        order = sorted(range(lam), key=lambda i: -fits[i]["fit"])
        if fits[order[0]]["fit"] > best_r["fit"]:
            best, best_r = inds[order[0]], fits[order[0]]
        yw = (w[:, None] * Y[order[:mu]]).sum(axis=0)
        m = np.clip(m + sigma * yw, 0, 1)
        Cinvsqrt = vecs @ np.diag(1 / D) @ vecs.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chin < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        Ym = Y[order[:mu]]
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (Ym.T @ np.diag(w) @ Ym))
        sigma *= float(np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1)))
        sigma = min(sigma, 0.5)
        mean = statistics.mean(f["fit"] for f in fits)
        history.append({"gen": g + 1, "best": best_r, "mean": mean, "params": best, "sigma": round(sigma, 4)})
        print("generación %2d: mejor %s · media %.3f · σ %.3f · %.0f s" % (g + 1, fmt(best_r), mean, sigma,
                                                                          time.time() - t0), flush=True)
        save({"best": best})
    return best


def fmt(r):
    return "aptitud %.3f · éxito %.0f %% · choques %d · tiempo %s (%d vuelos)" % (
        r["fit"], 100 * r["success"], r["crash"], "%.1f s" % r["time"] if r["time"] else "—", r["n"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="hibrido", choices=list(RANGES))
    ap.add_argument("--profile", default="x650")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=2, help="semillas (mundos) por escenario para entrenar")
    ap.add_argument("--valid", type=int, default=8, help="semillas NUEVAS por escenario para la validación")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=11, help="semilla del algoritmo genético")
    ap.add_argument("--opt", default="cmaes", choices=["cmaes", "ga"], help="optimizador (CMA-ES o genético)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    base = current(a.pilot, a.profile)
    best = base
    pop = [base] + [mutate(rng, a.pilot, base, p=0.6, sigma=0.15) for _ in range(a.pop // 4)]
    pop += [random_ind(rng, a.pilot) for _ in range(a.pop - len(pop))]
    history = []
    seeds = [7000 + i for i in range(a.seeds)]                        # los mismos mundos en todas las generaciones

    def save(extra=None):
        if a.out:     # se guarda en cada generación: si se para, no se pierde lo aprendido
            d = {"pilot": a.pilot, "profile": a.profile, "base": base, "best": best, "history": history}
            d.update(extra or {})
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)

    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        if a.opt == "cmaes":
            best = cmaes(pool, a, base, seeds, history, save)
        for g in range(a.gens if a.opt == "ga" else 0):
            fits = evaluate(pool, a.pilot, a.profile, pop, seeds)
            order = sorted(range(len(pop)), key=lambda i: -fits[i]["fit"])
            best, best_r = pop[order[0]], fits[order[0]]
            mean = statistics.mean(f["fit"] for f in fits)
            history.append({"gen": g + 1, "best": best_r, "mean": mean, "params": best})
            print("generación %2d: mejor %s · media %.3f · %.0f s" % (g + 1, fmt(best_r), mean, time.time() - t0),
                  flush=True)
            save()
            sigma = 0.12 * (1 - 0.6 * g / max(a.gens - 1, 1))            # mutación que se reduce al converger
            elite = [pop[i] for i in order[:2]]
            children = []
            while len(children) < a.pop - len(elite):
                c = crossover(rng, a.pilot, tournament(rng, pop, fits), tournament(rng, pop, fits))
                children.append(mutate(rng, a.pilot, c, sigma=sigma))
            pop = elite + children
        print("\nmejor:", json.dumps(best))
        vseeds = [9600 + i for i in range(a.valid)]                  # validación: semillas nunca vistas
        vb, vt = evaluate(pool, a.pilot, a.profile, [base, best], vseeds)
        print("validación (semillas nuevas, %d vuelos):\n  antes     %s\n  entrenado %s" % (vb["n"], fmt(vb), fmt(vt)),
              flush=True)
    print("(%.0f s)" % (time.time() - t0))
    save({"valid_base": vb, "valid_best": vt})


if __name__ == "__main__":
    main()
