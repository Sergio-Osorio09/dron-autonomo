"""Evaluación sin interfaz: muchos vuelos por perfil y condición, con los mismos mundos para todos.

    python eval/eval_headless.py                         # tabla completa (≈ 2-4 min)
    python eval/eval_headless.py --flights 3 --md eval/resultados.md
    python eval/eval_headless.py --maps desconocido        # solo con el mapa que construye el dron (fase 2)
"""
import argparse
import os
import statistics
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dron.sim import Simulation  # noqa: E402
from dron.world import LEVELS  # noqa: E402

CONDITIONS = {
    "calma, sin ruido": dict(wind_speed=0, gusts=0, noise="ideal"),
    "calma, ruido realista": dict(wind_speed=0, gusts=0, noise="realista"),
    "viento 6 m/s + ráfagas ligeras": dict(wind_speed=6, gusts=1, noise="realista"),
    "viento 10 m/s + ráfagas moderadas": dict(wind_speed=10, gusts=2, noise="realista"),
    "viento 10 m/s + ráfagas fuertes, ruido alto": dict(wind_speed=10, gusts=3, noise="alto"),
    "montañoso, meta en la cima": dict(terrain="montañoso", goal_kind="cima", wind_speed=4, gusts=1),
    "ciudad con colinas, meta en azotea": dict(level="ciudad", terrain="colinas", goal_kind="azotea", wind_speed=4, gusts=1),
    "precipicios, densidad alta": dict(terrain="precipicios", density="alta", wind_speed=4, gusts=1),
    "bosque extremo + viento 6 m/s": dict(level="bosque", density="extrema", wind_speed=6, gusts=1),
    "lluvia fuerte + viento 6 m/s": dict(rain="fuerte", wind_speed=6, gusts=1),
    "carrera, meta en el suelo": dict(mode="carrera", wind_speed=4, gusts=1),
    "carrera, meta en el aire, montañoso": dict(mode="carrera", goal_kind="aire", terrain="montañoso", wind_speed=4, gusts=1),
    "carrera, objetivo suave": dict(mode="carrera", motion="suave", terrain="colinas", wind_speed=4, gusts=1),
    "carrera, objetivo medio": dict(mode="carrera", motion="medio", terrain="colinas", wind_speed=4, gusts=1),
    "carrera, objetivo rápido": dict(mode="carrera", motion="rápido", terrain="colinas", wind_speed=4, gusts=1),
    "carrera, objetivo variable": dict(mode="carrera", motion="variable", terrain="colinas", wind_speed=4, gusts=1),
}


def run(profile, cond, levels, flights, seed, precision=True, map_mode="conocido"):
    res, times, land_err, est_err, vmax, energy, dist, replans, map_replans = Counter(), [], [], [], [], [], [], [], []
    cfg = dict(CONDITIONS[cond])
    if "level" in cfg:  # el escenario fija el nivel: mismos vuelos en total
        levels, flights = [cfg.pop("level")], flights * len(levels)
    for level in levels:
        for i in range(flights):
            s = Simulation(profile=profile, level=level, seed=seed + i, wind_dir=(i * 97) % 360,
                           precision_landing=precision, map_mode=map_mode, **cfg)
            errs = []
            while not s.done:
                s.step(0.5)
                errs.append(float(np.linalg.norm(s.est.p - s.drone.p)))
            res[s.status] += 1
            if s.status in ("success", "missed") and s.mode == "aterrizar":
                land_err.append(float(np.hypot(*(s.drone.p[:2] - s.mission.pad))))
            if s.status == "success":
                times.append(s.t)
            est_err.append(statistics.mean(errs))
            vmax.append(s.max_speed)
            energy.append(100 - s.battery())
            replans.append(s.mission.replans)
            map_replans.append(s.mission.map_replans)
            if s.status == "success":
                dist.append(s.distance)
    n = len(levels) * flights
    return {"profile": profile, "cond": cond, "success": res["success"] / n, "missed": res["missed"] / n,
            "crash": res["crash"] / n, "timeout": res["timeout"] / n,
            "time": statistics.mean(times) if times else None,
            "land": statistics.mean(land_err) if land_err else None, "est": statistics.mean(est_err),
            "vmax": max(vmax), "energy": statistics.mean(energy),
            "dist": statistics.mean(dist) if dist else None, "replans": statistics.mean(replans),
            "map_replans": statistics.mean(map_replans)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="mini,px4,matrice")
    ap.add_argument("--conditions", default=",".join(str(i) for i in range(len(CONDITIONS))),
                    help="índices de CONDITIONS (0 = calma sin ruido ...)")
    ap.add_argument("--flights", type=int, default=2, help="vuelos por nivel")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--no-precision", action="store_true", help="aterrizar solo con GPS")
    ap.add_argument("--maps", default="conocido,desconocido", help="conocido (fase 1), desconocido (fase 2) o ambos")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    t0 = time.time()
    rows = ["| perfil | condición | mapa | éxito | fuera de la plataforma | choca | tiempo medio | distancia "
            "| replanif. (por el mapa) | error de aterrizaje | error de estimación | vel. máx. | batería usada |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for prof in a.profiles.split(","):
        for cond in [list(CONDITIONS)[int(i)] for i in a.conditions.split(",")]:
            for mm in a.maps.split(","):
                r = run(prof, cond, list(LEVELS), a.flights, a.seed, not a.no_precision, mm)
                rows.append("| %s | %s | %s | %.0f%% | %.0f%% | %.0f%% | %s | %s | %.1f (%.1f) | %s | %.2f m | %.1f m/s | %.2f%% |" % (
                    prof, cond, mm, 100 * r["success"], 100 * r["missed"], 100 * r["crash"],
                    "%.1f s" % r["time"] if r["time"] else "—", "%.0f m" % r["dist"] if r["dist"] else "—",
                    r["replans"], r["map_replans"], "%.2f m" % r["land"] if r["land"] is not None else "—",
                    r["est"], r["vmax"], r["energy"]))
                print(rows[-1], flush=True)
    note = "\n%d vuelos por nivel (bosque, ciudad, mixto) y condición; aterrizaje de precisión %s.\n" % (
        a.flights, "desactivado" if a.no_precision else "activado")
    print(note + "(%.0f s)" % (time.time() - t0))
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("# Resultados: mapa conocido (fase 1) y mapa desconocido (fase 2)\n\n" + "\n".join(rows) + "\n" + note)


if __name__ == "__main__":
    main()
