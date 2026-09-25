# Dron autónomo

Simulador 3D de un dron autónomo hecho con los **algoritmos que usan los drones reales** y con **datos reales** de
DJI Mini 3, DJI Matrice 350 RTK y del autopiloto PX4. El dron despega y cruza bosques o ciudades sobre terreno con
colinas, montañas o precipicios, con viento que se abriga tras los edificios, lluvia y sensores imperfectos. Después
aterriza sobre una plataforma en el suelo, en una azotea o en una cima, o, en **modo carrera**, cruza la meta lo más
rápido posible.

**Fase 2:** el dron puede volar **sin conocer el mundo**. Con sus telémetros y una cámara de profundidad construye
un mapa de ocupación 3D mientras vuela, planifica sobre él y replanifica cada vez que descubre un obstáculo.

> Arquitectura, fuentes de los datos, lecciones aprendidas y plan de fases: **[docs/CONTEXTO.md](docs/CONTEXTO.md)**

## Opciones de cada vuelo

| Opción | Valores |
|---|---|
| Mapa | **Desconocido** (lo construye con sus sensores, fase 2) o **conocido** (fase 1, para comparar) |
| Misión | **Aterrizar** en la meta o **Carrera** (cruzar la meta sin frenar; se guarda el mejor tiempo) |
| Dron | Mini 3 (249 g) · genérico PX4 · Matrice 350 RTK |
| Nivel y densidad | bosque / ciudad / mixto · densidad baja, normal, alta o extrema |
| Terreno | plano · colinas · montañoso · precipicios (mesetas con acantilados) |
| Meta | en el suelo · en una azotea · en una cima · en el aire (carrera) · al azar |
| Objetivo (carrera) | quieto · en movimiento sobre un vehículo: suave (~1 m/s), medio (~2,5 m/s), rápido (~5 m/s) o variable (acelera, frena, se para y gira sin avisar) |
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
- HUD con fase, velocidad, altura, inclinación, viento y batería, más una brújula con el viento y el rumbo.
- Gráficas de **velocidad real frente a planificada** y de **error de estimación frente a viento**.
- Ficha del dron con sus datos reales y resultados de la sesión.
- Controles de todas las opciones de la tabla de arriba, más semilla, velocidad de simulación y cámara
  (persecución, libre o cenital).

## Fluidez

La simulación corre en un hilo del servidor hasta 1 s por delante de lo que ves, y el navegador reproduce con un
pequeño colchón, así que las replanificaciones no provocan tirones. Si la GPU está ocupada por otros programas (por
ejemplo, entrenando un modelo), la calidad gráfica baja sola para mantener la fluidez.

## Resultados

Tabla completa (3 drones × 16 escenarios × mapa conocido y desconocido, mismos mundos para todos) en
[eval/resultados.md](eval/resultados.md):

- **Sin conocer el mundo, 46 de 48 combinaciones al 100 %** (igual que con el mapa conocido): los tres drones en
  calma y con viento, montaña con meta en la cima, azotea, precipicios, bosque extremo, lluvia fuerte, carreras y
  **objetivo en movimiento**.
- **Coste de construir el mapa:** +2 % de tiempo (22,4 s frente a 22,0 s de media). Cada replanificación tarda ~15 ms.
- **Límites reales:** el PX4 genérico, con viento de 10 m/s (su límite), aterriza a ~1 m de la plataforma con ráfagas
  moderadas y choca 2 de 3 veces con ráfagas fuertes.
- **Precisión:** aterriza a 3-36 cm del centro. Error del filtro: ~1,2 m con el GPS del Mini, ~0,7 m con PX4 y
  ~0,2 m con RTK.

## Uso

```bash
pip install -r requirements.txt
python server.py            # http://127.0.0.1:7873
python -m pytest -q         # 23 tests
python eval/eval_headless.py --flights 2 --md eval/resultados.md
```

## Estructura

```
dron-autonomo/
├── CLAUDE.md, docs/CONTEXTO.md     guía para agentes y contexto completo
├── dron/                           params, world (terreno), target (objetivo móvil), dynamics, wind, sensors, estimator,
│                                   mapping (mapa que construye el dron), planning, control, mission, sim
├── eval/                           eval_headless.py y resultados
├── tests/                          física, viento, filtro, planificación, mapa y misión completa
├── server.py                       servidor local (puerto 7873)
└── ui/                             index.html, style.css, app.js (three.js)
```

## Próximas fases

2 (hecha salvo las trayectorias B-spline) · 3 misión de búsqueda con niebla de guerra y mapa de probabilidad ·
4 objetivo que huye · 5 varios drones · 6 banco de pruebas y puente a PX4 SITL. Detalle en `docs/CONTEXTO.md`.
