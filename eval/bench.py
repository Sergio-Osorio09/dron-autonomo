"""Banco de pruebas (fase 6): muchas semillas de UN escenario, en paralelo, con intervalos de confianza.

Las tablas de eval_headless.py y eval_search.py usan pocos vuelos por casilla (3), así que un 67 % puede ser mala
suerte. Aquí se repite un escenario con N semillas y se da:
  * la tasa de éxito con su intervalo de confianza del 95 % de Wilson (el adecuado para proporciones con pocas
    muestras o cerca del 0 % / 100 %);
  * el tiempo: media, mediana y percentil 90; y las causas de fallo;
  * opcionalmente, un CSV con cada vuelo (semilla, resultado, tiempo, causa...) para analizarlo aparte.

    python eval/bench.py --n 60 --jobs 12 profile=mini level=bosque density=extrema wind_speed=6 gusts=1
    python eval/bench.py --n 40 --drones 3 profile=px4 search=bayesiana --csv enjambre.csv
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dron.sim import Simulation  # noqa: E402
from dron.swarm import Swarm  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    """Intervalo de confianza de Wilson para una proporción k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fly(args):
    seed, drones, kw = args
    s = Swarm(drones, seed=seed, **kw) if drones > 1 else Simulation(seed=seed, **kw)
    t0 = time.time()
    while not s.done:
        s.step(0.5)
    m = s.mission
    return {"semilla": seed, "resultado": s.status, "causa": s.cause or "", "tiempo": round(s.t, 2),
            "hallada": None if getattr(m, "found_t", None) is None else round(m.found_t, 2),
            "perdido": getattr(m, "lost_count", 0), "distancia": round(s.distance, 1),
            "replanificaciones": m.replans, "cpu": round(time.time() - t0, 1)}


def parse_kw(items):
    kw = {}
    for it in items:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        kw[k] = v
    return kw


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("kw", nargs="*", help="parámetros de Simulation: clave=valor (profile=mini level=bosque ...)")
    ap.add_argument("--n", type=int, default=30, help="número de semillas")
    ap.add_argument("--seed", type=int, default=1000, help="primera semilla")
    ap.add_argument("--drones", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    kw = parse_kw(a.kw)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(fly, [(a.seed + i, a.drones, kw) for i in range(a.n)]))
    n = len(rows)
    ok = [r for r in rows if r["resultado"] == "success"]
    lo, hi = wilson(len(ok), n)
    times = sorted(r["tiempo"] for r in ok)
    print("escenario: %s%s" % (kw, "" if a.drones == 1 else ", %d drones" % a.drones))
    print("éxito: %d/%d = %.0f %%  (IC 95 %% de Wilson: %.0f-%.0f %%)" % (
        len(ok), n, 100 * len(ok) / n, 100 * lo, 100 * hi))
    if times:
        p90 = times[min(len(times) - 1, int(math.ceil(0.9 * len(times))) - 1)]
        print("tiempo (éxitos): media %.1f s · mediana %.1f s · p90 %.1f s" % (
            statistics.mean(times), statistics.median(times), p90))
    fails = Counter((r["resultado"], r["causa"]) for r in rows if r["resultado"] != "success")
    for (res, cause), c in fails.most_common():
        print("  %s%s: %d (semillas %s)" % (res, " (" + cause + ")" if cause else "", c,
                                           ", ".join(str(r["semilla"]) for r in rows
                                                     if (r["resultado"], r["causa"]) == (res, cause))[:80]))
    print("(%.0f s con %d procesos)" % (time.time() - t0, a.jobs))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
