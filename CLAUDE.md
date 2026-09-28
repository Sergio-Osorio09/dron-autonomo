# Dron autónomo: guía para agentes

Proyecto autocontenido en esta carpeta. No toques nada fuera de `dron-autonomo/`: la carpeta padre `Laya/` aloja
otros proyectos (`laya-2048`, `laya-drone`). Este proyecto NO usa Laya ni redes neuronales: son algoritmos
clásicos de robótica como los de drones reales.

Lee primero `docs/CONTEXTO.md`, que tiene la arquitectura, los datos reales, las lecciones aprendidas y el plan de fases.
**En un chat nuevo, empieza por la sección 6 de `docs/CONTEXTO.md`:** estado actual, pendientes, qué partes del código
conocen el mapa (lo que cambia en la fase 2) y el plan de la fase 2.

## Reglas de trabajo

- Cada dato de un dron real va en `dron/params.py` con su fuente. Lo que no publica el fabricante se marca como ESTIMADO.
- El control y la misión usan SIEMPRE el estado estimado (`sim.est`), nunca el real (`sim.drone`). Con
  `map_mode="desconocido"`, tampoco el mundo: planifican sobre `sim.map` (dron/mapping.py). El estado real
  solo se usa para la física, los sensores y la evaluación.
- `dron/dynamics.py` es la única fuente de verdad de la física, y `dron/world.py` de la geometría. La física por
  defecto es la de 4 motores (`Multirotor6DOF`, con el autopiloto interno de PX4); `DRON_PHYSICS=simple` usa la de
  antes. Las ganancias de cada dron las ajusta el autotune (`dron/autotune.json`; `DRON_AUTOTUNE=0` sin él).
- En la carrera, la embestida (`mission.RAM`) desbloquea inclinación y velocidad con la meta a la vista (`DRON_RAM=0`
  la quita). Con búsqueda, un objetivo en movimiento solo se conoce por la cámara (`Mission.camera_only`).
- La web (`ui/`) solo dibuja; three.js se carga desde jsDelivr.
- Código y textos en español.
- Un solo dron por vuelo: el enjambre (varios drones, fase 5) queda para el futuro. Su código sigue, pero la interfaz
  no lo ofrece y sus pruebas solo corren con `DRON_SWARM=1` (también el servidor).
- Se entrena UN dron: el Holybro X650 (`params.TRAINING`, perfil `x650`, carga útil con `payload=`); los demás perfiles
  quedan para pruebas y comparaciones.
- Los algoritmos de pilotaje elegibles están en `dron/pilots.py` (clásico, fábrica, híbrido); uno nuevo (p. ej. uno
  evolucionado con un algoritmo genético) se registra ahí y se compara con `bench.py ... pilot=<clave>`.
- Los parámetros de seguridad y velocidad salen de `dron/tuning.py` (ajustados por dron). Si cambias control o
  planificación, compara también con `DRON_TUNED=0` (los de fábrica) en el banco de pruebas.
- Tras cambiar física, control, sensores o planificación: `python -m pytest -q` y las evaluaciones de abajo. Para
  decidir entre dos variantes, no fiarse de 3 vuelos: `eval/bench.py` con 30 semillas y mirar el intervalo de Wilson.

## Comandos

```bash
python -m pytest -q                                   # 54 tests (+2 del enjambre con DRON_SWARM=1), ~1 min
python server.py                                      # http://127.0.0.1:7873
python eval/eval_headless.py --flights 1 --jobs 10 --md eval/resultados.md          # fases 1, 2 y 4, ~5 min
python eval/eval_search.py --jobs 10 --md eval/resultados_busqueda.md               # fase 3, ~5 min
python eval/eval_search.py --profiles px4 --conditions 0 --maps conocido --drones 1,2,3 --flights 2 --jobs 6 --md eval/resultados_enjambre.md   # fase 5 (aparcada)
python eval/bench.py --n 30 profile=px4 mode=carrera motion=huye level=mixto          # banco de pruebas (fase 6)
python eval/tune.py --profile px4 --out eval/ajuste_px4.json                          # ajuste automático por dron, ~1 h
python eval/evolve.py --pilot hibrido --profile x650 --out eval/evo_hibrido_x650.json  # entrenamiento (CMA-ES; --opt ga: genético), ~1 h
python eval/apply_evolved.py                                                           # adopta solo lo validado (dron/entrenados.json)
python eval/train_all.py                                                               # clásico e híbrido en el X650, varias horas
python eval/maniobras.py --autotune                                                    # escalón, parada, círculo y giro (destreza)
python eval/autotune.py --profile x650 --jobs 4                                         # autotune de las ganancias del dron (~10 min)
```
