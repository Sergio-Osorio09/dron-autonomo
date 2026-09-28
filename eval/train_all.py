"""Entrena con el algoritmo genético los pilotos de carrera (híbrido, reactivo y clásico × Mini, PX4 y Matrice) uno tras otro
y, al terminar cada uno, adopta lo que valida (eval/apply_evolved.py). Pensado para dejarlo entrenando solo: si se
para, lo ya entrenado y validado queda guardado (eval/evo_*.json y dron/entrenados.json).

    python eval/train_all.py --pop 20 --gens 16 --jobs 18         # varias horas
    python eval/train_all.py --only reactivo/mini                  # solo uno
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# un solo dron de entrenamiento (params.TRAINING: el Holybro X650) y los dos pilotos que se entrenan; "fábrica" no se
# entrena (es la referencia) y el reactivo puro se retiró (ver pilots.py)
RUNS = [("clasico", "x650"), ("hibrido", "x650")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=16)
    # 3 mundos por escenario (18 carreras por individuo): con 2, el reactivo del Matrice no chocaba en el
    # entrenamiento pero en la validación chocó más que antes (5 frente a 2) y se rechazó: sobreajuste
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--valid", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--only", default=None, help="piloto/dron, p. ej. hibrido/x650")
    ap.add_argument("--skip-done", action="store_true", help="salta los que ya tienen validación")
    a = ap.parse_args()
    t0 = time.time()
    for pilot, prof in RUNS:
        if a.only and a.only != "%s/%s" % (pilot, prof):
            continue
        out = os.path.join(HERE, "eval", "evo_%s_%s.json" % (pilot, prof))
        if a.skip_done and os.path.exists(out) and '"valid_best"' in open(out, encoding="utf-8").read():
            print("=== %s %s: ya entrenado y validado, se salta" % (pilot, prof), flush=True)
            continue
        print("=== %s %s (%.0f min desde el inicio)" % (pilot, prof, (time.time() - t0) / 60), flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "eval", "evolve.py"), "--pilot", pilot, "--profile", prof,
                        "--pop", str(a.pop), "--gens", str(a.gens), "--seeds", str(a.seeds), "--valid", str(a.valid),
                        "--jobs", str(a.jobs), "--out", out], cwd=HERE)
        subprocess.run([sys.executable, os.path.join(HERE, "eval", "apply_evolved.py")], cwd=HERE)
    print("FIN (%.0f min)" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
