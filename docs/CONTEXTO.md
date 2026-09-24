# Contexto del proyecto: Dron autónomo

Simulador de un dron autónomo construido con los **algoritmos que se usan en drones reales**: sin redes
neuronales y sin Laya (eso vive en `../laya-drone`). Lo desarrollamos por fases; cada fase se puede ver y medir.
Este documento es para personas y para agentes de IA que retomen el trabajo.

## 1. Arquitectura (igual que en PX4, Skydio o los drones del FAST Lab)

```
sensores con ruido → estimación (Kalman) → [mapa] → misión → planificación → trayectoria → control → motores
```

| Módulo | Archivo | Algoritmo | Referente real |
|---|---|---|---|
| Parámetros | `dron/params.py` | Perfiles con datos de fabricante | Fichas de DJI Mini 3 y Matrice 350 RTK; parámetros por defecto de PX4 |
| Física | `dron/dynamics.py` | Masa puntual con actitud; arrastre deducido de v_max; inercia de motores y actitud | Modelo de simuladores de planificación (FAST Lab, Flightmare) |
| Mundo | `dron/world.py` | Terreno con relieve (mapa de alturas), obstáculos apoyados en él, densidad, meta en suelo/azotea/cima/aire | — |
| Viento | `dron/wind.py` | Medio + cizalladura logarítmica + turbulencia **Dryden** + estelas de abrigo tras obstáculos y relieve, aceleración sobre azoteas y cimas | MIL-F-8785C / MIL-HDBK-1797; modelos de estela simplificados (no CFD) |
| Sensores | `dron/sensors.py` | IMU, GPS con deriva, barómetro, brújula, 40 telémetros, cámara inferior; la lluvia degrada alcance y precisión | Ruidos tipo EKF2 de PX4 |
| Estimación | `dron/estimator.py` | Filtro de Kalman de 9 estados con puertas de innovación | EKF2 de PX4 |
| Planificación | `dron/planning.py` | Campo de distancias + **A\*** + estirado de cuerda + Chaikin + perfil de velocidad óptimo en tiempo | Voxblox/FIESTA (ESDF), Theta*, TOPP |
| Control | `dron/control.py` | Cascada posición → velocidad (PID) → aceleración → inclinación y empuje; **Collision Prevention** | `mc_pos_control` y CollisionPrevention de PX4 |
| Objetivo móvil | `dron/target.py` | Vehículo que recorre el terreno (suave, medio, rápido, variable); **filtro de Kalman de velocidad constante** para seguirlo; **punto de intercepción** | Seguimiento de blancos y guiado con adelanto (lead pursuit) |
| Misión | `dron/mission.py` | Modo **aterrizar** (despegue → crucero → aproximación → aterrizaje de precisión) o **carrera** (cruzar la meta sin frenar); velocidad limitada por el alcance de los sensores; margen según la incertidumbre del filtro | Modos Takeoff/Mission/Land e IR-LOCK de PX4 |
| Simulación | `dron/sim.py` | Bucle a 200 Hz que une todo | — |

## 2. Datos reales usados (y lo estimado)

Todos están en `dron/params.py`, con fuente:

- **PX4** (valores por defecto en `mc_pos_control/*.yaml` y `ekf2/params_*.yaml`): 12 m/s máx., crucero 5 m/s,
  aceleración 3 m/s², tirón 4 m/s³, subida 3 m/s, bajada 1,5 m/s, inclinación 45°, despegue 1,5 m/s,
  aterrizaje 0,7 m/s, ganancias (XY_P 0,95; VEL P/I/D 1,8/0,4/0,2; Z_P 1,0; Z VEL P/I 4,0/2,0),
  ruido GPS 0,5 m y 0,3 m/s, ruido de acelerómetro 0,35 m/s².
- **DJI Mini 3**: 248 g, 16 m/s, subida 5 m/s, bajada 3,5 m/s, viento 10,7 m/s, 18,1 Wh / 38 min,
  precisión GNSS ±1,5 m.
- **DJI Matrice 350 RTK**: 23 m/s, subida 6 m/s, bajada 5 m/s, viento 12 m/s, inclinación 30°, giro 100°/s,
  cabeceo 300°/s, 2 × 263,2 Wh / 55 min, RTK ±0,1 m, detección de obstáculos hasta 40 m.
- **Estimado** (no lo publica el fabricante): relación empuje/peso, tamaño, constantes de tiempo de motores y
  actitud, inclinación máxima del Mini (35°), velocidad de crucero autónoma del Mini y del Matrice.
- El **arrastre** no se inventa: a velocidad máxima con la inclinación máxima, el empuje horizontal iguala al
  arrastre, así que `k = g·tan(inclinación) / v_max²`. Un test comprueba que cada perfil alcanza su v_max real.

## 3. Lecciones de la fase 1 (qué falló y cómo se resolvió)

1. **El GPS miente por medio metro y el planificador pasaba a 0,6 m de los árboles.** El Mini y el PX4 chocaban
   con ruido realista, mientras que el Matrice (RTK, ±0,1 m) nunca. Se resolvió como en los drones reales, con
   **Collision Prevention**: los telémetros miden distancias relativas, independientes del GPS, y limitan la
   velocidad hacia cada obstáculo para poder frenar siempre (v ≤ √(2·a·(d − d_seg))).
2. **El sensor inclinado hacia abajo veía el suelo como obstáculo** y no dejaba aterrizar. Los rayos que miran
   hacia abajo se excluyen de Collision Prevention.
3. **Troncos finos entre dos rayos.** Con 12 rayos cada 30°, un tronco de 60 cm a 1,5 m cabía en el hueco. Ahora
   son 36 rayos cada 10° (PX4 trabaja con 72 sectores de 5°).
4. **El viento desplaza el aterrizaje.** Si al bajar se desvía más de 35 cm, deja de bajar y se realinea con la
   cámara inferior, como el aterrizaje de precisión de PX4.

5. **Lluvia y carrera: ir más rápido de lo que se ve.** Con lluvia fuerte (alcance de 6 m) o en carrera (14 m/s con
   sensores de 12 m), el Mini no podía frenar a tiempo. La regla de oro de los drones autónomos es que la velocidad
   máxima sea la que permite frenar dentro del alcance de los sensores: √(2·a·(alcance − radio − 1 m)).
6. **El GPS del Mini se desvía 1,4 m y el camino pasaba a 0,6 m de un árbol.** El planificador deja ahora un margen
   de 0,6 m + 2σ de la incertidumbre del filtro (máx. 2,5 m), y si no hay camino prueba con uno menor.
   Collision Prevention frena con la mitad de la aceleración (cuenta con el retardo del dron) y guarda 0,8 m.
7. **Aterrizar en una azotea no es chocar con el edificio**, pero entrar de lado en una pared sí: el contacto con una
   superficie solo cuenta como aterrizaje si viene de arriba (penetración < 25 cm).

8. **Al replanificar hacia un objetivo móvil, el dron frenaba cada segundo** porque cada trayectoria nueva empezaba
   a velocidad 0. Ahora arranca a la velocidad actual. Las persecuciones bajaron de hasta 61 s a unos 15 s.
9. **La persecución en línea recta atravesaba el árbol que rodeaba el vehículo.** La persecución predictiva solo se
   usa con línea de visión libre; si no, sigue la trayectoria planificada que lo rodea.
10. **Distancia de seguridad dinámica:** margen fijo + velocidad × 0,3 s de reacción (antes, a 3 m/s empezaba a
    frenar demasiado tarde).
11. **Geovalla:** persiguiendo, el Matrice chocó contra el borde de la arena, que los telémetros no ven. Los bordes
    son ahora paredes virtuales en Collision Prevention, como la Geofence de PX4.
12. **El filtro divergía al rozar el borde de una meseta:** la física paraba el dron en seco, pero la IMU no lo
    registraba, así que el filtro seguía "moviéndose", rechazaba el GPS por no cuadrar y acababa con 23 m de error.
    Ahora el golpe lo registra la IMU, el filtro se reinicia con el GPS si lo rechaza durante 0,5 s (como EKF2) y en
    crucero se mantiene 1 m sobre la superficie de debajo y la de 0,8 s por delante.

13. **Tirones en la interfaz.** Había tres causas. (a) Un A* hacia un punto inalcanzable exploraba el mapa entero y
    congelaba la simulación hasta 18 s. Ahora comprueba antes si origen y destino están en la misma zona libre
    conectada (componentes conexas con scipy, al instante), usa arrays de numpy en vez de diccionarios y tiene un
    presupuesto de 60 000 nodos; el peor paso bajó a ~150 ms. (b) El navegador pedía los pasos de uno en uno: ahora
    el servidor simula en un hilo propio hasta 1 s por delante, y el navegador reproduce con 0,3 s de colchón y
    conexiones persistentes, así que los pasos lentos ya no se ven. (c) Calidad gráfica adaptativa: por debajo de
    ~35 fps baja la resolución interna y la de las sombras.

## 4. Resultados

Ver `eval/resultados.md` (3 drones × 16 escenarios: viento, ruido, terrenos, metas, lluvia, densidad, carrera y
objetivo en movimiento).

## 5. Fases del proyecto

| Fase | Contenido | Algoritmos |
|---|---|---|
| **1 (hecha)** | Física realista, viento, sensores con ruido, estimación, control, planificación y misión con mapa conocido | Dryden, EKF, cascada PX4, A*, TOPP, Collision Prevention |
| **1b (hecha)** | Terreno con relieve y precipicios, densidad, viento que interactúa con obstáculos, lluvia, meta en azotea/cima/aire, modo carrera | Mapa de alturas, estelas de viento, límite de velocidad por alcance de sensores, márgenes según la covarianza |
| **1c (hecha)** | Carrera contra un objetivo en movimiento (suave, medio, rápido, variable) | Kalman de velocidad constante, punto de intercepción, persecución predictiva con línea de visión, geovalla |
| 2 | El dron construye su mapa con sensores ruidosos (ya no conoce el mundo); replanificación; trayectorias B-spline | Mapa de ocupación / ESDF, **D\* Lite**, B-splines tipo **EGO-Planner** |
| 3 | Misión de búsqueda: meta desconocida, cámara con cono y oclusión, zona designada, niebla de guerra y mapa de calor | **Búsqueda bayesiana**, exploración por fronteras (**FUEL**), cobertura boustrophedon |
| 4 | Objetivo que HUYE del dron y se esconde tras edificios (el seguimiento de objetivos móviles ya existe desde la 1c) | Persecución-evasión, búsqueda desde la última posición vista |
| 5 | Varios drones que se reparten la búsqueda (activable) | Subastas **CBBA** / algoritmo húngaro, **Voronoi**, **ORCA** |
| 6 | Banco de pruebas, repeticiones y, opcionalmente, puente a PX4 SITL + Gazebo | — |

## 6. Estado actual y cómo continuar (para un chat nuevo)

**Estado:** la fase 1 (con 1b y 1c) está terminada. 18 tests en verde; `eval/resultados.md` con 3 drones × 16
escenarios (45 de 48 al 100 %). Repositorio privado: https://github.com/Sergio-Osorio09/dron-autonomo (rama `main`).
Comandos: `python -m pytest -q`, `python server.py` (http://127.0.0.1:7873) y
`python eval/eval_headless.py --flights 1 --md eval/resultados.md` (~10 min).

**Pendientes conocidos de la fase 1** (opcionales, no bloquean la fase 2):
- El tirón (jerk) del perfil de velocidad es aproximado (suavizado), no una curva en S exacta.
- PX4 genérico con viento de 10 m/s: aterriza a ~1 m de la plataforma o choca con ráfagas fuertes (está en su límite).
- Matrice en bosque de densidad extrema con viento: 1 choque de 3.
- La batería se gasta, pero no hay "volver a casa con batería baja" (RTL por batería, como PX4).
- Idea descartada por ahora: modo "carrera sin red" que ignore el límite de velocidad por alcance de sensores.

**Qué partes CONOCEN EL MAPA hoy (lo que la fase 2 tiene que cambiar):**
- `planning.ClearanceGrid(world)` se construye con TODOS los obstáculos y el terreno (`sim.py`, `make_flyable_world`).
- `Mission` planifica con esa rejilla completa (`plan(...)`), y también la replanificación hacia objetivos móviles.
- `estimator.range_down(dist, world.surface(...))` usa el mapa del terreno para convertir la distancia en altura.
- `sim._tick` usa `world.surface` para la altura mínima en crucero.
- La física, los sensores, la evaluación y el viento SÍ pueden usar el mundo real: son "la realidad", no el dron.

**Plan propuesto para la fase 2 (mapa desconocido):**
1. `dron/mapping.py`: mapa de ocupación 3D con log-odds (vóxeles de 0,5-1 m, como OctoMap) que se actualiza con los
   40 telémetros ruidosos (raycasting: celdas libres a lo largo del rayo y ocupada al final). El terreno también se
   aprende (el rayo inferior y los inclinados).
2. Campo de distancias (ESDF) incremental sobre ese mapa, para la holgura. Lo desconocido se trata de forma
   optimista para planificar lejos (como hacen los planificadores de exploración) y como obstáculo muy cerca.
3. Replanificación cuando el mapa invalida la trayectoria: A* incremental o **D\* Lite**; trayectorias locales
   suaves con B-splines tipo **EGO-Planner** si da tiempo.
4. Cambiar `Mission` y `sim` para usar el mapa aprendido en lugar de `ClearanceGrid(world)`. Mantener una opción
   "mapa conocido" para comparar.
5. Interfaz: dibujar el mapa que va construyendo el dron (vóxeles ocupados y zona explorada) frente al mundo real.
6. Evaluación: los mismos escenarios con mapa conocido y desconocido (éxito, tiempo, distancia recorrida,
   replanificaciones). Tests: el mapa converge al mundo real y no hay choques contra obstáculos no vistos.

**Preferencias del usuario:** todo en español; algoritmos que se usen en drones reales y datos reales con su fuente;
que se vea bien en pantalla; medir antes de afirmar (tests y evaluación tras cada cambio); explicar las decisiones.
Hacer commit o push solo cuando lo pida.

**Otros proyectos de la carpeta `Laya/`** (no tocar desde aquí): `laya-2048` y `laya-drone` (pilotos con el modelo
Laya). En `laya-drone` hay un ajuste v2 pausado al 92 %: `python training/finetune.py --base models/laya-drone
--out models/laya-drone-v2 --data data/drone_v2_train.jsonl --val data/drone_v2_val.jsonl --epochs 0.6
--lr-enc 2e-5 --lr-head 1e-4 --resume`.
