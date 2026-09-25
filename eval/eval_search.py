"""Evaluación de la misión de búsqueda (fase 3): las tres estrategias con los mismos mundos y zonas.

    python eval/eval_search.py                                     # 3 drones × 3 estrategias × 2 mapas
    python eval/eval_search.py --flights 2 --md eval/resultados_busqueda.md
    python eval/eval_search.py --jobs 12                   # combinaciones en paralelo (mismos resultados)
    python eval/eval_search.py --drones 1,2,3 --maps conocido --md eval/resultados_enjambre.md   # fase 5
"""
import argparse
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dron.search import STRATEGIES  # noqa: E402
from dron.sim import Simulation  # noqa: E402
from dron.swarm import Swarm  # noqa: E402
from dron.world import LEVELS  # noqa: E402

CONDITIONS = {
    "calma": dict(wind_speed=0, gusts=0),
    "viento 6 m/s": dict(wind_speed=6, gusts=1),
    "colinas + lluvia moderada": dict(terrain="colinas", rain="moderada", wind_speed=4, gusts=1),
}


def run(profile, strategy, map_mode, cond, flights, seed, drones=1):
    res, found, total, covered, falses, dist = Counter(), [], [], [], [], []
    for level in LEVELS:
        for i in range(flights):
            kw = dict(profile=profile, level=level, seed=seed + i, wind_dir=(i * 97) % 360, search=strategy,
                      map_mode=map_mode, **CONDITIONS[cond])
            s = Swarm(drones, **kw) if drones > 1 else Simulation(**kw)
            while not s.done:
                s.step(0.5)
            res[s.status] += 1
            if s.mission.found_t is not None:
                found.append(s.mission.found_t)
            if s.status == "success":
                total.append(s.t)
                dist.append(s.distance)
            covered.append(s.searcher.covered())
            falses.append(sum(d["state"] == "falsa" for d in s.searcher.detections))
    n = len(LEVELS) * flights
    mean = lambda xs: statistics.mean(xs) if xs else None
    return {"success": res["success"] / n, "crash": res["crash"] / n, "found": len(found) / n,
            "t_found": mean(found), "t_total": mean(total), "covered": mean(covered), "falses": mean(falses),
            "dist": mean(dist)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="mini,px4,matrice")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--maps", default="conocido,desconocido")
    ap.add_argument("--conditions", default=",".join(str(i) for i in range(len(CONDITIONS))))
    ap.add_argument("--flights", type=int, default=1, help="vuelos por nivel")
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--md", default=None)
    ap.add_argument("--drones", default="1", help="drones del enjambre, p. ej. 1,2,3 (fase 5)")
    ap.add_argument("--jobs", type=int, default=1, help="procesos en paralelo (cada combinación es independiente)")
    a = ap.parse_args()
    t0 = time.time()
    rows = ["| perfil | condición | estrategia | mapa | drones | encontrada | éxito | choca | tiempo hasta encontrarla "
            "| tiempo total | zona cubierta | falsas alarmas | distancia |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "—" if v is None else f % v
    combos = [(prof, cond, st, mm, int(n)) for prof in a.profiles.split(",")
              for cond in [list(CONDITIONS)[int(i)] for i in a.conditions.split(",")]
              for st in a.strategies.split(",") for mm in a.maps.split(",") for n in a.drones.split(",")]
    args = [(p, s, m, c, a.flights, a.seed, n) for p, c, s, m, n in combos]
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        results = pool.map(run, *zip(*args)) if a.jobs > 1 else map(run, *zip(*args))
        for (prof, cond, st, mm, n), r in zip(combos, results):
            rows.append("| %s | %s | %s | %s | %d | %.0f%% | %.0f%% | %.0f%% | %s | %s | %.0f%% | %.1f | %s |" % (
                prof, cond, st, mm, n, 100 * r["found"], 100 * r["success"], 100 * r["crash"],
                fmt(r["t_found"], "%.1f s"), fmt(r["t_total"], "%.1f s"), 100 * r["covered"], r["falses"],
                fmt(r["dist"], "%.0f m")))
            print(rows[-1], flush=True)
    note = "\n%d vuelos por nivel (bosque, ciudad, mixto) y combinación; zona de 30-40 × 20-30 m.\n" % a.flights
    print(note + "(%.0f s)" % (time.time() - t0))
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("# Resultados de la misión de búsqueda (fase 3)\n\n" + "\n".join(rows) + "\n" + note)


if __name__ == "__main__":
    main()
