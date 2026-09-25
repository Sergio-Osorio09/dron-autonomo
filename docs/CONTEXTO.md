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
| Sensores | `dron/sensors.py` | IMU, GPS con deriva, barómetro, brújula, 40 telémetros, cámara inferior y (fase 2) **cámara de profundidad** de 24 × 14 rayos; la lluvia degrada alcance y precisión | Ruidos tipo EKF2 de PX4; campo de visión de la Intel RealSense D435 |
| Mapa (fase 2) | `dron/mapping.py` | **Mapa de ocupación 3D con log-odds** (vóxeles de 1 m) construido desde la posición estimada; suposición 2,5D; **ESDF** con transformada de distancia exacta; lo desconocido es libre para planificar | OctoMap (parámetros de octomap_server), Voxblox/FIESTA, Fast-Planner/EGO-Planner |
| Estimación | `dron/estimator.py` | Filtro de Kalman de 9 estados con puertas de innovación | EKF2 de PX4 |
| Planificación | `dron/planning.py` | Campo de distancias + **A\*** + estirado de cuerda + Chaikin + perfil de velocidad óptimo en tiempo; funciona igual con el mundo conocido o con el mapa aprendido | Voxblox/FIESTA (ESDF), Theta*, TOPP |
| Control | `dron/control.py` | Cascada posición → velocidad (PID) → aceleración → inclinación y empuje; **Collision Prevention** | `mc_pos_control` y CollisionPrevention de PX4 |
| Objetivo móvil | `dron/target.py` | Vehículo que recorre el terreno (suave, medio, rápido, variable); **filtro de Kalman de velocidad constante** para seguirlo; **punto de intercepción** | Seguimiento de blancos y guiado con adelanto (lead pursuit) |
| Misión | `dron/mission.py` | Con mapa desconocido, comprueba la trayectoria cada vez que cambia el mapa y **replanifica** sin frenar; si no hay camino, espera y reintenta. Modo **aterrizar** (despegue → crucero → aproximación → aterrizaje de precisión) o **carrera** (cruzar la meta sin frenar); velocidad limitada por el alcance de los sensores; margen según la incertidumbre del filtro | Modos Takeoff/Mission/Land e IR-LOCK de PX4 |
| Simulación | `dron/sim.py` | Bucle a 200 Hz que une todo; `map_mode` = `conocido` (fase 1) o `desconocido` (fase 2) | — |

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
- **Mapa (fase 2)**: cámara de profundidad con el campo de visión de la Intel RealSense D435 (87° × 58°, hoja de
  datos) y error del 2 % (D435: < 2 % a 2 m); su alcance es el del perfil (el Matrice 350 detecta obstáculos
  hasta 40 m). Mapa con los parámetros por defecto de octomap_server: acierto 0,7, fallo 0,4, límites 0,12 y 0,97.
  **Estimado:** la resolución de la cámara (24 × 14 rayos, submuestreada para que Python vaya en tiempo real), el
  3 % de píxeles sin dato y que el error de la cámara crezca linealmente con la distancia (en una estéreo real
  crece con el cuadrado).
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

## 3b. Lecciones de la fase 2 (mapa desconocido)

1. **Obstáculos fantasma a 15 m que convertían el cielo en una columna sólida.** Un rayo sin eco devolvía el
   alcance máximo *más ruido*, así que la mitad de las veces parecía un eco justo antes del alcance. Con la
   suposición 2,5D, todo lo que quedaba debajo pasaba a ser sólido, incluido el propio dron: se quedaba atascado
   o subía sin parar. Un sensor real dice "sin retorno", no da una distancia con ruido. Ahora es así.
2. **El suelo llano no aparecía en el mapa.** Los ecos del suelo a z = 0 con ruido negativo quedaban fuera de la
   rejilla y se descartaban. Ahora se llevan a la primera capa.
3. **Celdas ocupadas en parte.** Un árbol que ocupa media celda puede quedar marcado como libre, porque los rayos
   que pasan por el hueco la vacían (es lo que ocurre también en OctoMap). El 84 % de los ecos cae en una celda
   ocupada y el 98 % a menos de una celda de una; por eso la holgura resta media celda y se mantiene Collision
   Prevention, que usa las distancias de los telémetros sin pasar por el mapa.
4. **Sin mapa del terreno, el telémetro inferior no da la altura absoluta.** Solo da la distancia a lo que hay
   debajo. Con mapa desconocido, la altura sale del GPS y del barómetro (como en PX4 sin estimación del terreno). Por
   eso el aterrizaje baja, como el modo Land de PX4, hasta que el dron toca: la referencia puede quedar hasta 1,5 m
   por debajo de la plataforma. Antes paraba a 0,5 m y, con la altura estimada algo alta, se habría quedado
   flotando.
5. **D\* Lite no hace falta aquí.** Una replanificación completa (A\* + suavizado + perfil) tarda 15 ms de
   mediana y 19 ms como máximo (medido en bosque extremo, ciudad y montaña). D\* Lite reaprovecha la búsqueda
   anterior, pero cada celda nueva cambia la holgura de todas las cercanas, y en Python cada nodo es más lento. Con
   estos tiempos, A\* desde cero es lo más sencillo y rápido.
6. **Replanificar en bucle junto a un obstáculo recién visto.** Si el dron ve algo muy cerca, cualquier
   trayectoria nueva empieza dentro de su margen y la comprobación la rechazaría una y otra vez. Por eso la
   comprobación ignora el primer metro alrededor del dron (de eso se encarga Collision Prevention).
7. **Persiguiendo un objetivo, la línea de visión salía siempre "bloqueada" con el mapa aprendido.** La puerta va
   a 1 m del vehículo y el dron vuela a su altura; con celdas de 1 m, el suelo aprendido quita holgura a los dos
   extremos del segmento. El dron no pasaba a persecución predictiva y replanificaba detrás del objetivo (hasta
   67 s). Ahora se comprueba solo el tramo intermedio con radio + 0,3 m, suficiente para ver si hay un árbol o un
   edificio en medio. La carrera contra el objetivo rápido es muy variable en los dos modos: con 9 vuelos, el PX4
   tardó 32 s de media con mapa conocido y 21,5 s con desconocido.
8. **Hueco de la fase 1 que ha salido a la luz: descender sobre la copa de un árbol.** El anillo de telémetros es
   horizontal y el rayo inferior no cuenta en Collision Prevention (para poder aterrizar). Un dron que vuela a
   8,4 m y baja despacio sobre un árbol de 8,3 m no lo ve. Con el cambio del sensor sin eco (lección 1), un vuelo
   de carrera con mapa conocido (PX4, objetivo suave, mixto, semilla 500) cambió lo justo para caer en él. Está
   pendiente (ver sección 6).

## 4. Resultados

Ver `eval/resultados.md` (3 drones × 16 escenarios: viento, ruido, terrenos, metas, lluvia, densidad, carrera y
objetivo en movimiento; cada uno con mapa conocido y desconocido).

- **Mapa desconocido: 46 de 48 combinaciones al 100 %** (mapa conocido: 44 de 48). Solo falla el PX4 con viento de
  10 m/s, que es su límite en los dos modos.
- **Coste de no conocer el mundo:** en las 44 combinaciones que los dos modos completan al 100 %, el tiempo medio pasa
  de 21,2 s a 22,2 s (+5 %). Las mayores diferencias están en las carreras contra objetivos rápidos, que son muy
  variables en los dos modos (lección 3b.7).
- Replanificaciones por el mapa: de media entre 0,3 y 29 por vuelo (más en persecuciones). Cada una tarda ~15 ms.

## 5. Fases del proyecto

| Fase | Contenido | Algoritmos |
|---|---|---|
| **1 (hecha)** | Física realista, viento, sensores con ruido, estimación, control, planificación y misión con mapa conocido | Dryden, EKF, cascada PX4, A*, TOPP, Collision Prevention |
| **1b (hecha)** | Terreno con relieve y precipicios, densidad, viento que interactúa con obstáculos, lluvia, meta en azotea/cima/aire, modo carrera | Mapa de alturas, estelas de viento, límite de velocidad por alcance de sensores, márgenes según la covarianza |
| **1c (hecha)** | Carrera contra un objetivo en movimiento (suave, medio, rápido, variable) | Kalman de velocidad constante, punto de intercepción, persecución predictiva con línea de visión, geovalla |
| **2 (hecha)** | El dron construye su mapa con sensores ruidosos (ya no conoce el mundo); replanificación. Pendiente: trayectorias B-spline | Mapa de ocupación con log-odds (OctoMap), ESDF, A\* con replanificación (D\* Lite descartado, ver 3b.5); B-splines tipo **EGO-Planner** pendientes |
| 3 | Misión de búsqueda: meta desconocida, cámara con cono y oclusión, zona designada, niebla de guerra y mapa de calor | **Búsqueda bayesiana**, exploración por fronteras (**FUEL**), cobertura boustrophedon |
| 4 | Objetivo que HUYE del dron y se esconde tras edificios (el seguimiento de objetivos móviles ya existe desde la 1c) | Persecución-evasión, búsqueda desde la última posición vista |
| 5 | Varios drones que se reparten la búsqueda (activable) | Subastas **CBBA** / algoritmo húngaro, **Voronoi**, **ORCA** |
| 6 | Banco de pruebas, repeticiones y, opcionalmente, puente a PX4 SITL + Gazebo | — |

## 6. Estado actual y cómo continuar (para un chat nuevo)

**Estado:** fases 1 (con 1b y 1c) y 2 terminadas; de la fase 2 solo faltan las trayectorias B-spline. 23 tests en
verde. `eval/resultados.md` compara los dos modos de mapa (3 drones × 16 escenarios × mapa conocido/desconocido):
46 de 48 combinaciones al 100 % con mapa desconocido y 44 de 48 con conocido. Repositorio privado:
https://github.com/Sergio-Osorio09/dron-autonomo (rama `main`).
Comandos: `python -m pytest -q`, `python server.py` (http://127.0.0.1:7873) y
`python eval/eval_headless.py --flights 1 --md eval/resultados.md` (~20 min con los dos modos; `--maps desconocido`
para uno solo).

**Cómo funciona la fase 2** (detalle en el docstring de `dron/mapping.py`):
- `Simulation(map_mode="desconocido")` crea `sim.map` (OccupancyMap) y activa la cámara de profundidad. Cada barrido
  (telémetros a 20 Hz, cámara a 10 Hz) se inserta desde `est.p`. La interfaz arranca en este modo.
- `Mission._space()` devuelve el ESDF aprendido (`map.grid()`, recalculado solo si cambió el mapa) o el mundo real.
  `planning.plan` y `segment_clear` aceptan cualquiera de los dos.
- Cada 0,2 s, si el mapa cambió, `Mission._still_clear` comprueba la trayectoria restante; si choca, replanifica
  desde la posición estimada con la velocidad actual. Sin camino: espera en el sitio y reintenta cada 0,5 s.
- La altura mínima de crucero sale de `map.surface`; el telémetro inferior ya no corrige la altura absoluta;
  el objetivo móvil da su altura por GNSS.
- `make_flyable_world` sigue usando el mundo real (solo para generar un mundo con camino) y la física, los sensores,
  el viento y la evaluación también: son "la realidad".

**Pendientes** (opcionales):
- Fase 2: trayectorias locales suaves con B-splines tipo EGO-Planner (ahora: A* + estirado + Chaikin + TOPP).
- **Collision Prevention no ve hacia abajo en crucero** (lección 3b.8): descender sobre la copa de un árbol puede
  acabar en choque. Idea: en crucero, usar el rayo inferior solo para limitar la bajada, con una distancia de
  seguridad pequeña (radio + 0,3 m) para no estorbar al aterrizaje ni a la puerta de meta.
- **El planificador y Collision Prevention no usan el mismo límite de velocidad.** El plan usa
  √(2·a·(alcance − radio − 1)) y Collision Prevention frena con a/2 y más distancia de seguridad. El dron se queda
  atrás de la referencia, se desvía más de 3 m y replanifica (~8 veces por vuelo incluso sin ruido ni viento).
  Unificarlos daría trayectorias más fieles, pero cambia los resultados de la fase 1.
- El tirón (jerk) del perfil de velocidad es aproximado (suavizado), no una curva en S exacta.
- PX4 genérico con viento de 10 m/s: aterriza a ~1 m de la plataforma o choca con ráfagas fuertes (está en su límite).
- La batería se gasta, pero no hay "volver a casa con batería baja" (RTL por batería, como PX4).
- Con mapa desconocido la carrera contra objetivos rápidos es muy variable (15-40 s), igual que con mapa conocido.
- Idea descartada por ahora: modo "carrera sin red" que ignore el límite de velocidad por alcance de sensores.

**Siguiente: fase 3** (misión de búsqueda). Ya existe la base: el mapa de ocupación, la niebla de lo no explorado
(`map.seen`) y la cámara de profundidad. Falta la cámara con cono y oclusión para *detectar* el objetivo, el mapa de
probabilidad (búsqueda bayesiana) y la exploración por fronteras (FUEL) sobre `map.seen`.

**Preferencias del usuario:** todo en español; algoritmos que se usen en drones reales y datos reales con su fuente;
que se vea bien en pantalla; medir antes de afirmar (tests y evaluación tras cada cambio); explicar las decisiones.
Hacer commit o push solo cuando lo pida.

**Otros proyectos de la carpeta `Laya/`** (no tocar desde aquí): `laya-2048` y `laya-drone` (pilotos con el modelo
Laya). En `laya-drone` hay un ajuste v2 pausado al 92 %: `python training/finetune.py --base models/laya-drone
--out models/laya-drone-v2 --data data/drone_v2_train.jsonl --val data/drone_v2_val.jsonl --epochs 0.6
--lr-enc 2e-5 --lr-head 1e-4 --resume`.
