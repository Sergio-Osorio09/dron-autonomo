"""Adopta lo que ha aprendido el algoritmo genético (eval/evolve.py) SOLO si lo ha validado en semillas nuevas.

Lee eval/evo_<piloto>_<dron>.json y, para cada uno con validación, adopta los parámetros entrenados si en la
validación la aptitud mejora y no choca más que antes. Los guarda en dron/entrenados.json, que dron/pilots.py carga
al arrancar (solo para la misión de carrera). Lo que no pasa la validación se queda como estaba.

    python eval/apply_evolved.py            # muestra y guarda
    python eval/apply_evolved.py --dry      # solo muestra
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "eval"))
from evolve import RANGES  # noqa: E402
OUT = os.path.join(HERE, "dron", "entrenados.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    trained = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            trained = json.load(f)
    for path in sorted(glob.glob(os.path.join(HERE, "eval", "evo_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        name = "%s/%s" % (d["pilot"], d["profile"])
        if "valid_best" not in d:
            print("%-18s sin validar todavía (entrenando o parado)" % name)
            continue
        vb, vt = d["valid_base"], d["valid_best"]
        ok = vt["fit"] > vb["fit"] and vt["crash"] <= vb["crash"]
        rng = RANGES.get(d["pilot"], {})
        out = [k for k, v in d["best"].items() if k in rng and not rng[k][0] - 1e-9 <= v <= rng[k][1] + 1e-9]
        if ok and out:        # entrenado con límites más amplios que los realistas de ahora
            print("%-18s fuera de los límites realistas (%s): no se adopta" % (name, ", ".join(out)))
            trained.get(d["pilot"], {}).pop(d["profile"], None)
            continue
        print("%-18s antes: aptitud %.3f, éxito %.0f %%, choques %d · entrenado: aptitud %.3f, éxito %.0f %%, "
              "choques %d → %s" % (name, vb["fit"], 100 * vb["success"], vb["crash"], vt["fit"], 100 * vt["success"],
                                   vt["crash"], "ADOPTADO" if ok else "no mejora: se queda como estaba"))
        if ok:
            trained.setdefault(d["pilot"], {})[d["profile"]] = {
                "params": d["best"], "valid_base": vb, "valid_best": vt}
    if not a.dry:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(trained, f, indent=2, ensure_ascii=False)
        print("guardado en", OUT)


if __name__ == "__main__":
    main()
