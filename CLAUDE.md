# Dron autónomo: guía para agentes

Proyecto autocontenido en esta carpeta. No toques nada fuera de `dron-autonomo/`: la carpeta padre `Laya/` aloja
otros proyectos (`laya-2048`, `laya-drone`). Este proyecto NO usa Laya ni redes neuronales: son algoritmos
clásicos de robótica como los de drones reales.

Lee primero `docs/CONTEXTO.md`, que tiene la arquitectura, los datos reales, las lecciones aprendidas y el plan de fases.
**En un chat nuevo, empieza por la sección 6 de `docs/CONTEXTO.md`:** estado actual, pendientes, qué partes del código
conocen el mapa (lo que cambia en la fase 2) y el plan de la fase 2.

## Reglas de trabajo

- Cada dato de un dron real va en `dron/params.py` con su fuente. Lo que no publica el fabricante se marca como ESTIMADO.
- El control y la misión usan SIEMPRE el estado estimado (`sim.est`), nunca el real (`sim.drone`). El estado real
  solo se usa para la física, los sensores y la evaluación.
- `dron/dynamics.py` es la única fuente de verdad de la física, y `dron/world.py` de la geometría.
- La web (`ui/`) solo dibuja; three.js se carga desde jsDelivr.
- Código y textos en español.
- Tras cambiar física, control, sensores o planificación: `python -m pytest -q` y `python eval/eval_headless.py`.

## Comandos

```bash
python -m pytest -q                                   # 18 tests, ~20 s
python server.py                                      # http://127.0.0.1:7873
python eval/eval_headless.py --flights 2 --md eval/resultados.md   # ~4 min
```
