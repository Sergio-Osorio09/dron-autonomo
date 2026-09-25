# Dron autónomo

Simulador 3D de un dron autónomo hecho con los **algoritmos que usan los drones reales** y con **datos reales** de
DJI Mini 3, DJI Matrice 350 RTK y del autopiloto PX4. El dron despega y cruza bosques o ciudades sobre terreno con
colinas, montañas o precipicios, con viento que se abriga tras los edificios, lluvia y sensores imperfectos. Después
aterriza sobre una plataforma en el suelo, en una azotea o en una cima, o, en **modo carrera**, cruza la meta lo más
rápido posible.

**Fase 2:** el dron puede volar **sin conocer el mundo**. Con sus telémetros y una cámara de profundidad construye
un mapa de ocupación 3D mientras vuela, planifica sobre él y replanifica cada vez que descubre un obstáculo.

**Fases 3 a 6:** puede **buscar** una meta de la que solo sabe la zona (barrido, fronteras o búsqueda bayesiana),
**perseguir un vehículo que huye** y se esconde tras los edificios, **repartirse la búsqueda entre 2-4 drones**, y
cada vuelo se puede **repetir, guardar y abrir**. Un banco de pruebas mide todo con intervalos de confianza.

**Mapas densos y ajuste:** vuela dentro de un **almacén** sin GPS (estanterías, pasillos de hasta 2,4 m, techo) con
odometría visual-inercial, y en un **bosque de densidad máxima**. Sus parámetros de seguridad y velocidad se
**ajustan automáticamente por dron** en esos mapas (sin redes neuronales), y al replanificar **cose** la trayectoria
nueva a la que ya seguía, como Fast-Planner y EGO-Planner.

> Arquitectura, fuentes de los datos, lecciones aprendidas y plan de fases: **[docs/CONTEXTO.md](docs/CONTEXTO.md)**
> Piezas para construir un dron real con esta autonomía (~1000-1100 $): **[docs/HARDWARE.md](docs/HARDWARE.md)**

## Opciones de cada vuelo

| Opción | Valores |
|---|---|
| Mapa | **Desconocido** (lo construye con sus sensores, fase 2) o **conocido** (fase 1, para comparar) |
| Misión | **Aterrizar** en la meta o **Carrera** (cruzar la meta sin frenar; en cuanto ve la meta, **sprint a la velocidad máxima del fabricante**: 16 m/s el Mini 3, 23 m/s el Matrice 350, 12 m/s PX4; se guarda el mejor tiempo) |
| Dron | Mini 3 (249 g) · genérico PX4 · Matrice 350 RTK |
| Nivel y densidad | bosque / ciudad / mixto / **almacén** (interior, sin GPS) · densidad baja, normal, alta, extrema o máxima |
| Terreno | plano · colinas · montañoso · precipicios (mesetas con acantilados) |
| Meta | en el suelo · en una azotea · en una cima · en el aire (carrera) · al azar |
| Objetivo (carrera) | quieto · en movimiento sobre un vehículo: suave (~1 m/s), medio (~2,5 m/s), rápido (~5 m/s), variable (acelera, frena, se para y gira sin avisar) o **huye y se esconde** (fase 4: sin rastreador, solo lo ve la cámara) |
| Búsqueda | no (sabe dónde está la meta) · buscarla con **barrido** (cortacésped), **fronteras** (FUEL) o **bayesiana** (fase 3) |
| Drones | 1 a 4: se reparten la búsqueda o persiguen en equipo al vehículo que huye (fase 5) |
| Clima | viento 0-14 m/s y dirección · ráfagas (ninguna a fuertes) · lluvia (no, moderada, fuerte) |
| Sensores | ideales · realistas · ruido alto; aterrizaje de precisión y Collision Prevention activables |

## Qué incluye

- **Física realista (200 Hz):** el dron solo acelera inclinándose. Se modelan empuje limitado, inclinación
  máxima, arrastre (deducido de la velocidad máxima del fabricante), inercia de motores y actitud, y aterrizajes
  suaves o duros.
- **Tres drones reales:** pequeño tipo Mini 3 (249 g), genérico PX4 y de inspección tipo Matrice 350 RTK.
- **Terreno con relieve:** mapa de alturas en el que se apoyan edificios y árboles. Se puede aterrizar en azoteas y
  cimas, y chocar contra la pared de un acantilado.
- **Viento:** medio con cizalladura según la altura y ráfagas con el modelo de turbulencia **Dryden** (MIL-F-8785C).
  Los edificios, árboles y montañas crean **estelas de abrigo** (menos viento, más turbulencia) y el viento se
  acelera sobre azoteas y cimas.
- **Lluvia:** reduce el alcance y la precisión de telémetros, cámara y GPS, y añade arrastre.
- **Sensores con ruido:** IMU con sesgo, GPS con deriva, barómetro, brújula, 40 telémetros, cámara inferior y dos
  cámaras de profundidad (frontal e inferior).
  Niveles ideal, realista y alto.
- **Mapa que construye el propio dron (fase 2):** una cámara de profundidad (campo de visión de la RealSense
  D435) y los telémetros alimentan un **mapa de ocupación 3D con log-odds**, como OctoMap, insertado desde la
  posición *estimada*. Sobre él se calcula el **campo de distancias (ESDF)**. Lo desconocido se trata como libre
  para planificar; cada vez que el mapa cambia, el dron comprueba su trayectoria y **replanifica sin frenar** si ya
  no está libre.
- **Búsqueda (fase 3):** una cámara de detección (82°, inclinada 60°) ve la plataforma según su tamaño en la
  imagen, si nada la tapa, y a veces se equivoca (falsas alarmas). El dron mantiene un **mapa de probabilidad** que
  actualiza con la regla de Bayes, elige adónde ir con tres estrategias y **confirma cada detección en vuelo**, sin
  pararse: va hacia ella (en carrera, a toda velocidad) y la confirma con las imágenes que toma por el camino. Un
  filtro "M de N" (como el de un radar) ignora las falsas alarmas sueltas.
- **Objetivo que huye (fase 4):** al ver al dron, el vehículo escapa y se esconde tras obstáculos. El dron lo sigue
  con la cámara en un gimbal y, si lo pierde, lo busca **desde la última posición vista** con una creencia que se
  difunde a la velocidad del vehículo.
- **Enjambre (fase 5):** 2-4 drones completos comparten la creencia; cada uno busca en su **celda de Voronoi**, nunca
  donde va otro, en su capa de altura, y se evitan entre sí. Quien encuentra la meta aterriza. Contra el vehículo
  que huye, **persiguen en equipo**: comparten cada avistamiento y le cierran las salidas por los lados.
- **Banco de pruebas y repeticiones (fase 6):** evaluaciones en paralelo, `eval/bench.py` con intervalos de
  confianza de Wilson, y grabación de cada vuelo en el navegador (repetir, guardar en JSON y abrir).
- **Interior sin GPS:** en el almacén el filtro se alimenta de **odometría visual-inercial** (ruido de 5 cm y
  deriva del 1 % de lo recorrido, como una Intel T265). El mapa guarda el centroide de los ecos de cada celda para
  que un pasillo de 2,4 m no se estreche con celdas de 1 m.
- **Ajuste automático por dron:** `eval/tune.py` busca (al azar y refinando alrededor del mejor) los parámetros de
  Collision Prevention, márgenes y velocidad que dan más éxito y menos tiempo en mapas densos y con viento, y los
  valida en semillas nuevas; quedan en `dron/tuning.py` (`DRON_TUNED=0` vuelve a los de fábrica).
- **Filtro de Kalman** (como el EKF2 de PX4) con puertas de innovación. El dron vuela con lo que *cree*, no con la
  posición real.
- **Planificación:** A\* sobre un campo de distancias, estirado de cuerda, suavizado y **perfil de velocidad
  óptimo en tiempo**. Acelera en las rectas y frena antes de las curvas.
- **Control en cascada de PX4** con sus ganancias por defecto, integral contra el viento y **Collision
  Prevention** con los telémetros.
- **Seguridad realista:** nunca vuela más rápido de lo que le permiten frenar sus sensores; deja más margen a los
  obstáculos cuanto peor es su GPS (usa la incertidumbre del filtro de Kalman); la distancia de seguridad crece con
  la velocidad; y una **geovalla** impide salirse de la zona de vuelo.
- **Misión:** *aterrizar* (despegue → crucero, replanificando si una ráfaga lo desvía → aproximación →
  **aterrizaje de precisión**) o *carrera* (despega y cruza la puerta de meta lo más rápido que le permiten sus
  sensores; se guarda el mejor tiempo). Si el objetivo va sobre un vehículo, el dron lo sigue con un **filtro de
  Kalman**, vuela al **punto de intercepción** y, con línea de visión, pasa a **persecución predictiva**.

## En pantalla

- La escena 3D con terreno coloreado por altura (hierba, roca, nieve), la plataforma o la puerta de meta, lluvia, el dron inclinándose de verdad, la **trayectoria coloreada por velocidad**, **partículas de
  viento**, la posición **estimada** (naranja) frente a la real, las **lecturas de GPS** (rosa) y los rayos de los
  telémetros.
- **El mapa del dron:** celdas ocupadas coloreadas por altura, niebla sobre lo que aún no ha explorado y los ecos de
  la cámara de profundidad. El selector *Vista* muestra el mundo real, solo lo que sabe el dron o los dos.
- **La búsqueda:** mapa de calor de la probabilidad sobre el terreno (ámbar = aún probable), la zona, el cono de la
  cámara y las detecciones (naranja pendiente, verde confirmada, gris falsa alarma). Los demás drones del enjambre
  y la **zona de cada uno** (fronteras con su color).
- **Cámaras del enjambre:** el selector *Seguir* elige a qué dron siguen las cámaras, y la cámara **A bordo** muestra
  lo que ve ese dron (82°, como la del DJI Mini 3); el HUD muestra su fase, velocidad y altura.
- Botones **⟲ Repetir**, **⭳ Guardar** y **⭱ Abrir** para volver a ver un vuelo sin simularlo otra vez.
- HUD con fase, velocidad, altura, inclinación, viento y batería, más una brújula con el viento y el rumbo.
- Gráficas de **velocidad real frente a planificada** y de **error de estimación frente a viento**.
- Ficha del dron con sus datos reales y resultados de la sesión.
- Controles de todas las opciones de la tabla de arriba, más semilla, velocidad de simulación y cámara
  (persecución, a bordo, libre o cenital).

## Fluidez

La simulación corre en un hilo del servidor hasta 1 s por delante de lo que ves, y el navegador reproduce con un
pequeño colchón, así que las replanificaciones no provocan tirones. Si la GPU está ocupada por otros programas (por
ejemplo, entrenando un modelo), la calidad gráfica baja sola para mantener la fluidez.

## Resultados

Tablas completas (mismos mundos para todos) en [eval/resultados.md](eval/resultados.md) (3 drones × 20 escenarios
× mapa conocido y desconocido), [eval/resultados_busqueda.md](eval/resultados_busqueda.md) y
[eval/resultados_enjambre.md](eval/resultados_enjambre.md):

- **58 de 60 combinaciones al 100 % en los dos modos de mapa**: los tres drones en calma y con viento, montaña con
  meta en la cima, azotea, precipicios, bosque de densidad máxima, **almacén sin GPS con pasillos de 2,4 m**, lluvia
  fuerte, carreras, **objetivo en movimiento** y **objetivo que huye**.
- **Sprint de carrera:** al ver la meta vuela a su velocidad máxima real, una cámara en gimbal afina la posición de la
  puerta y en los últimos metros apunta directamente a ella: carreras un 5-19 % más rápidas y ya no roza la puerta
  sin cruzarla (antes, casi la mitad de las veces), sin choques.
- **A por todas:** en carrera con búsqueda, confirmar la detección en vuelo en vez de pararse baja el tiempo un
  13-36 % según el dron (PX4: de 35,9 a 23,1 s), sin choques.
- **Ajuste automático:** con los parámetros ajustados cada dron vuela un 8-17 % más rápido en mapas densos y con
  viento que con los de fábrica, sin añadir choques.
- **Búsqueda:** 54 de 54 casillas al 100 %, sin choques. Confirmar la detección en vuelo (sin bajar a mirarla), un
  detector a 10 imágenes por segundo con filtro de falsas alarmas "M de N" y bajar en vertical sobre la plataforma
  la hacen un 18-33 % más rápida según el dron (PX4 con mapa desconocido: de 40,9 a 27,9 s; 40 semillas), con
  0,1 falsas alarmas por vuelo en vez de 1,3.
- **Objetivo que huye:** con un dron lo captura el 83 % de las veces (30 semillas) y tarda ~1,5 min; **en equipo,
  97 % con 2 o 3 drones y en ~50 s** ([eval/resultados_persecucion.md](eval/resultados_persecucion.md)).
- **Límites reales:** el PX4 genérico, con viento de 10 m/s (su límite), aterriza fuera de la plataforma o choca con
  ráfagas fuertes.
- **Precisión:** aterriza a 3-32 cm del centro. Error del filtro: ~1,2 m con el GPS del Mini, ~0,7 m con PX4 y
  ~0,2 m con RTK.

## Uso

```bash
pip install -r requirements.txt
python server.py            # http://127.0.0.1:7873
python -m pytest -q         # 35 tests
python eval/eval_headless.py --flights 1 --jobs 10 --md eval/resultados.md     # fases 1, 2 y 4
python eval/eval_search.py --jobs 10 --md eval/resultados_busqueda.md          # fase 3
python eval/bench.py --n 30 profile=px4 search=bayesiana                       # banco de pruebas
python eval/tune.py --profile matrice --out eval/ajuste_matrice.json           # ajuste automático (~1 h)
```

## Estructura

```
dron-autonomo/
├── CLAUDE.md, docs/CONTEXTO.md     guía para agentes y contexto completo
├── dron/                           params, world (terreno y almacén), target (objetivo móvil), dynamics, wind, sensors,
│                                   estimator, mapping (mapa que construye el dron), search (búsqueda), planning,
│                                   control, tuning (parámetros por dron), mission, sim, swarm (enjambre)
├── eval/                           eval_headless.py, eval_search.py, bench.py, tune.py y resultados
├── tests/                          física, viento, filtro, planificación, mapa, búsqueda, evasión, enjambre y misión
├── server.py                       servidor local (puerto 7873)
└── ui/                             index.html, style.css, app.js (three.js)
```

## Fases

Las fases 1 a 6 están hechas. Queda pendiente: trayectorias B-spline (fase 2) y el puente a PX4 SITL + Gazebo
(fase 6). Detalle y siguientes pasos en `docs/CONTEXTO.md`.
