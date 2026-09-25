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
| Mundo | `dron/world.py` | Terreno con relieve (mapa de alturas), obstáculos apoyados en él, densidad (hasta "máxima"), meta en suelo/azotea/cima/aire; **almacén** interior con estanterías, pasillos de 2,4-4,2 m, pilares, palés, paredes y techo | — |
| Viento | `dron/wind.py` | Medio + cizalladura logarítmica + turbulencia **Dryden** + estelas de abrigo tras obstáculos y relieve, aceleración sobre azoteas y cimas | MIL-F-8785C / MIL-HDBK-1797; modelos de estela simplificados (no CFD) |
| Sensores | `dron/sensors.py` | IMU, GPS con deriva, barómetro, brújula, 40 telémetros, cámara inferior y **cámaras de profundidad** frontal (24 × 14 rayos) e inferior (16 × 12); la lluvia degrada alcance y precisión | Ruidos tipo EKF2 de PX4; campo de visión de la Intel RealSense D435 |
| Mapa (fase 2) | `dron/mapping.py` | **Mapa de ocupación 3D con log-odds** (vóxeles de 1 m) construido desde la posición estimada; suposición 2,5D; **ESDF** con transformada de distancia exacta; lo desconocido es libre para planificar | OctoMap (parámetros de octomap_server), Voxblox/FIESTA, Fast-Planner/EGO-Planner |
| Estimación | `dron/estimator.py` | Filtro de Kalman de 9 estados con puertas de innovación; en interior se alimenta de **odometría visual-inercial (VIO)** en lugar de GPS | EKF2 de PX4; Intel T265 / VINS-Mono |
| Ajuste | `dron/tuning.py`, `eval/tune.py` | Parámetros de seguridad y velocidad **por dron**, ajustados automáticamente (búsqueda aleatoria + refinamiento, validados en semillas nuevas) | Ajuste por aparato de PX4 |
| Planificación | `dron/planning.py` | Campo de distancias + **A\*** + estirado de cuerda + Chaikin + perfil de velocidad óptimo en tiempo; al replanificar **cose** la trayectoria nueva a la actual; funciona igual con el mundo conocido o con el mapa aprendido | Voxblox/FIESTA (ESDF), Theta*, TOPP, Fast-Planner/EGO-Planner |
| Control | `dron/control.py` | Cascada posición → velocidad (PID) → aceleración → inclinación y empuje; **Collision Prevention** con telémetros, cámara frontal en 72 sectores y visión inferior | `mc_pos_control` y CollisionPrevention de PX4 |
| Objetivo móvil | `dron/target.py` | Vehículo que recorre el terreno (suave, medio, rápido, variable) o que **huye** del dron y se esconde (fase 4); **filtro de Kalman de velocidad constante** para seguirlo; **punto de intercepción** | Seguimiento de blancos y guiado con adelanto (lead pursuit); persecución-evasión |
| Búsqueda | `dron/search.py` | (fase 3) Zona de búsqueda, cámara de detección con cono, oclusión y falsas alarmas; **creencia bayesiana** en rejilla; estrategias **barrido** (boustrophedon), **fronteras** (FUEL) y **bayesiana**; confirmación de detecciones. (fase 4) Creencia de un blanco móvil que se difunde. (fase 5) Reparto por **Voronoi** + subasta voraz | Teoría de búsqueda de Koopman, IAMSAR; FUEL (Zhou et al. 2021); Cortés et al. 2004 |
| Enjambre | `dron/swarm.py` | (fase 5) 2-4 drones completos que comparten la creencia; capas de altura y evitación reactiva entre ellos | Cobertura con Voronoi, CBBA (versión voraz), reciprocidad de ORCA |
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
- **Búsqueda (fases 3-5)**: cámara de detección con el campo de visión del DJI Mini 3 (82,1°, ficha técnica),
  inclinada 60° hacia abajo; detector con entrada de 640 px (la estándar de YOLO). **Estimado:** la curva de
  detección (32 px fiable, 12 px nada, 0,9 como máximo), el 1 % de falsas alarmas por imagen, la altura de búsqueda
  (8 m), el vehículo que huye (5 m/s, 3 m/s², alerta a 30 m) y la separación entre capas del enjambre (2 m).
- **Interior (almacén)**: sin GPS; odometría visual-inercial con ruido de 5 cm y 0,05 m/s y deriva del 1 % de la
  distancia recorrida (**estimado**: la Intel T265 anunciaba < 1 % en bucle cerrado). Techo a 10 m, estanterías
  de 5-7 m y 1,2 m de fondo (**estimado**, del orden de las estanterías de palés habituales).
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
8. **Hueco de la fase 1: rozar la copa de un árbol o caer sobre un arbusto.** El anillo de telémetros es un plano
   horizontal: un dron a 8,4 m no ve la copa de un árbol de 8,3 m, pero su cuerpo (radio 0,3 m) la toca. Y
   persiguiendo un objetivo, bajaba a 1,6 m/s sobre un arbusto que tenía debajo y un poco delante. Se resolvió
   como en los drones reales: (a) la cámara de profundidad alimenta Collision Prevention en 72 sectores de 5°
   (lo más cercano dentro de una franja de ±(radio + 0,5 m) alrededor de la altura del dron), como el mensaje
   OBSTACLE_DISTANCE de PX4; (b) una segunda cámara de profundidad mira hacia abajo (visión inferior, como la de
   los DJI) y limita la bajada sobre lo que haya bajo el dron, con una distancia de seguridad pequeña (radio +
   0,3 m) para no estorbar a la puerta de meta. Solo actúa en crucero, carrera y persecución; no en el
   aterrizaje. Resultado: 46 de 48 combinaciones al 100 % en los dos modos de mapa.

## 3c. Lecciones de la fase 3 (búsqueda)

1. **Un punto de búsqueda pegado a un edificio que aún no conocía** dejaba al dron 16 s reintentando un camino
   imposible. Ahora, si no se puede llegar a un punto, se descarta (él y la zona que se quería mirar) y la estrategia
   elige otro.
2. **Dos falsas alarmas en el mismo sitio confirmaban una meta falsa** (aterrizaba a 41 m de la buena). Al mirar
   fijamente una zona para confirmar, las falsas alarmas caen justo ahí. Ahora hacen falta 3 detecciones a menos
   de 1,5 m de su mediana (con la plataforma real, de cerca, casi seguro; con falsas alarmas, ~10⁻⁵).
3. **Las fases nuevas no tenían las protecciones de vuelo.** La altura mínima sobre la superficie y la visión
   (frontal en sectores e inferior) solo se aplicaban en crucero y carrera: bajando a confirmar, un Mini cayó sobre
   la copa de un árbol. Ahora hay una lista única de fases de vuelo (`FLIGHT_PHASES`), y la visión inferior mira
   también hacia donde va el dron (0,6 s por delante), y la franja de los sectores se amplía hacia abajo al bajar.
4. **Idea descartada tras medirla: comprobar una detección sin desviarse.** Parecía que ahorraría los desvíos por
   falsas alarmas, pero mientras tanto se ignoraban las demás detecciones (también la de la plataforma de verdad):
   la búsqueda bayesiana pasó de 31 a 40 s de media y perdió 2 de 18 casillas. Se volvió a confirmar en el acto.

## 3d. Lecciones de las fases 4, 5 y 6

1. **(4) Perdido nada más despegar.** El dron no ve el vehículo a 40 m y lo daba por perdido a los 2,5 s. Ahora
   solo está "perdido" si está a menos de 25 m de donde lo predice y no lo ve; si no, sigue volando hacia allí.
2. **(4) Un recuadro de búsqueda alrededor de donde se perdió no sirve para un blanco rápido**: a 5 m/s sale de él
   en segundos y no se le volvía a encontrar nunca (300 s agotados). La creencia de un blanco móvil cubre toda la
   arena y se difunde a su velocidad máxima; solo se calcula la oclusión de lo que cae en el cono de la cámara.
3. **(4) Buscar a quien huye es una persecución.** Se buscaba a la velocidad de crucero: la del PX4 (5 m/s) es la
   del vehículo, así que nunca le recortaba distancia. Ahora busca a velocidad de carrera y 7 m más alto (se ve
   más terreno y tapan menos los obstáculos).
4. **(4) Chocar contra el borde persiguiendo un punto fuera de la arena.** El filtro de Kalman extrapolaba al
   vehículo más allá del borde; el dron iba a por él y rozaba la geovalla (4 choques en 30 vuelos). Ahora ninguna
   meta de planificación ni de persecución puede quedar fuera de la arena.
5. **(4) Perseguir un fantasma.** Sin verlo, el filtro lo seguía moviendo en línea recta para siempre: el dron lo
   perseguía sin llegar nunca a estar lo bastante cerca como para darlo por perdido y buscarlo. Al que huye no se
   le extrapola más de 1,5 s desde la última vez que se le vio. Banco de pruebas (PX4, mixto, 30 semillas): del
   70 % al 80 % de capturas y sin choques.
6. **(4) Ideas descartadas tras medirlas con 30 semillas:** interceptar desde 6 m más arriba (73 % frente a 70 %,
   y la mediana de tiempo empeora de 87 a 118 s), desplazar la creencia inicial en dirección contraria al dron
   (73 %) e ir primero a la última posición conocida antes de la búsqueda bayesiana (77 % frente a 83 %: la creencia
   ya empieza centrada allí y el vehículo ya se ha ido). Ninguna diferencia es significativa (los intervalos del 95 % se solapan casi por completo).
7. **(5) Estado por dron dentro de lo compartido.** Las franjas del barrido estaban en la creencia compartida: un
   dron se comía las del otro. Lo que es de cada dron (franjas, candidato que confirma) va con su identificador.
8. **(6) Las evaluaciones tardaban 20-25 min** en serie; con `--jobs` (procesos en paralelo, mismos resultados)
   tardan ~5 min con 10 procesos.
9. **Collision Prevention empujaba hacia la pared** (fallo de la fase 1 que salió al perseguir con el Matrice). El
   margen por tiempo de reacción crecía con la velocidad en TODAS las direcciones: un edificio que el dron acababa
   de dejar atrás contaba como "demasiado cerca" y el "aléjate" lo lanzaba contra el borde de enfrente. Ahora el
   margen usa solo la velocidad HACIA cada obstáculo, y una segunda pasada vuelve a aplicar todos los límites (sin
   empujones): ningún "aléjate" puede meterlo en otro obstáculo.
10. **Pedir menos velocidad no es frenar.** Si ya iba demasiado rápido hacia algo, Collision Prevention solo bajaba
    la velocidad pedida y el PID de velocidad (ganancia 1,8) frenaba flojo: llegaba tarde. Ahora añade la
    deceleración necesaria para parar en el espacio que queda (hasta la aceleración máxima). Con 9 y 10: de 46-48 a
    **49 de 51 casillas al 100 % en los dos modos**, y el objetivo que huye pasa a 100 % en todas sus casillas.
11. **Idea descartada tras medirla: planificar con el límite de Collision Prevention** (`control.cp_speed_limit`).
    Bajan un poco las replanificaciones (5,9 → 5,1) pero sube el tiempo 1-2 s y no mejora el éxito.
12. **(4 + 5) Un perseguidor no basta; varios, sí.** Con avistamiento compartido y cerco (los demás se abren 6 m a
    los lados), las capturas del vehículo que huye pasan del 80 % (1 dron) al 93 % (2) y al 100 % (3), y la mediana
    de tiempo de 120 s a 43 s (30 semillas; con el viento corregido de la lección 15: 83 %, 97 % y 97 %,
    `eval/resultados_persecucion.md`). Primero, con 3 drones, hubo 4 choques
    entre drones en 30 vuelos: convergían todos en la puerta y Collision Prevention ignoraba al compañero que tenía
    debajo (descarta los rayos hacia abajo). Ahora solo el primero baja a la puerta (los demás, +2 m por puesto) y
    los compañeros siempre cuentan, con su propia distancia de seguridad. 0 choques.
13. **Idea descartada tras medirla: B-splines tipo EGO-Planner** (`DRON_SMOOTHER=bspline`). La versión hecha
    (puntos de control cada metro, gradiente de suavidad + holgura del ESDF) no mejora a Chaikin: 97 frente a 98
    casillas al 100 %, algo más lenta y con más replanificaciones. Le faltaría lo que hace fuerte a EGO-Planner
    (factibilidad dinámica, reparto de tiempos y replanificación local a alta frecuencia). Queda como opción.
14. **Rendimiento**: los 40 telémetros usan ya el trazado de rayos vectorizado (mismos resultados): 4 drones con
    mapa desconocido pasan de 1,3× a 1,7× tiempo real.
15. **Ráfagas de 29 m/s con 6 m/s de viento medio** (lo encontró el banco de pruebas: un Matrice aterrizó a 1,2 m
    de la plataforma en una azotea). El multiplicador de turbulencia de las estelas se sumaba (+2,5 por edificio,
    +1,5 por el relieve) y llegaba a ×5. Ahora tiene un tope de ×2 (ESTIMADO: en una estela la intensidad de
    turbulencia crece del orden del doble). Barrido de robustez (4 escenarios difíciles × 40 semillas, mapa
    desconocido): 159/160 antes del arreglo, sin choques.
16. **Barridos de robustez con el banco de pruebas** (mapa desconocido, 40 semillas por escenario): Mini en bosque
    extremo con viento 40/40; Matrice en azotea con ráfagas moderadas 39/40 (antes del tope de turbulencia);
    búsqueda bayesiana con lluvia 40/40; carrera contra objetivo rápido en precipicios 40/40; búsqueda con 3 drones
    40/40; persecución en equipo (2 drones, ciudad) 39/40; con 4 drones (el máximo), búsqueda en ciudad 30/30 y
    persecución 30/30. Ningún choque.

## 3e. Lecciones de los mapas densos, el ajuste y el planificador

1. **Sin GPS dentro, y con GPS el mapa no sirve.** Construido con la posición del GPS (±0,5-1 m), un pasillo de 3,6 m
   se "emborronaba" y el dron no cabía. En interior, los drones reales usan odometría visual-inercial (precisa a
   corto plazo, con deriva); en el almacén el error baja a 0,1-0,4 m y la plataforma se corrige con la cámara.
2. **La suposición 2,5D rellenaba el almacén entero.** "Debajo de lo ocupado, sólido" es cierto fuera (no hay
   voladizos), pero al ver el techo se rellenaba todo hasta el suelo, incluido el aire donde vuela el dron. Ahora
   solo se rellena debajo de lo visto con rayos horizontales o que bajan; lo visto con rayos que suben (aunque sea
   3°: el Matrice ve el techo a 40 m con rayos casi horizontales) no dice nada de lo que tiene debajo. La altura
   mínima de crucero tampoco cuenta ya el techo como "superficie" (lo empujaba hacia arriba, contra él).
3. **Celdas de 1 m y pasillos de 3,6 m no encajan.** Con la holgura de peor caso (distancia a la celda ocupada −
   media celda), el centro del pasillo tenía 0,5 m y el dron no cabía en su propio mapa. Ahora cada celda guarda el
   centroide de sus ecos y la holgura se mide hasta él (como las distancias por debajo de la celda de Voxblox):
   ~1,1-1,3 m en el pasillo, sin pasar a celdas de 0,5 m (que costarían ~8 veces más).
4. **Coser las trayectorias al replanificar** (Fast-Planner / EGO-Planner): con mapa desconocido, los choques de la
   evaluación bajan de 2 a 0 y el tiempo un 14 %; almacén estrecho (PX4, 30 semillas) del 77 % al 80 % y el p90 de
   93 a 69 s. Antes, cada trayectoria nueva salía de la posición actual en la dirección del camino nuevo y, con la
   inercia, el dron se desviaba, se quedaba atrás y volvía a replanificar.
5. **Velocidad inicial en la dirección del camino nuevo.** Al replanificar se usaba la rapidez actual aunque el
   camino nuevo fuera hacia atrás: la referencia pedía −6 m/s mientras el dron iba a +7 y se pasaba de largo (un
   Matrice rozó un edificio yendo a confirmar una detección). Ahora la velocidad inicial es la componente de la
   actual en esa dirección: si hay que dar media vuelta, empieza frenando.
6. **Ajuste automático de parámetros, POR DRON.** Un único ajuste con el Mini y el PX4 (−15 % de tiempo en
   validación) hizo chocar al Matrice, que no estaba en el ajuste: más pesado, a 17 m/s no frenaba a tiempo. Como
   en PX4, cada dron tiene sus parámetros (`dron/tuning.py`), ajustados en mapas densos y con viento de 8 m/s y
   validados en 60 vuelos con semillas nuevas: Mini −15 %, PX4 −8 %, Matrice −17 % de tiempo, sin perder éxito ni
   añadir choques. Lo que aprende: frenar más fuerte (90 % de la aceleración en vez del 50 %), menos margen de
   reacción y de seguridad, más velocidad de crucero. Con margen de planificación ≥ 0,72 m, en cambio, chocaba más:
   en pasillos estrechos se quedaba sin camino.
7. **Lo que el ajuste no vio, hay que protegerlo aparte.** Regresiones en escenarios que no estaban en el ajuste,
   halladas con el banco de pruebas (40 semillas): (a) buscando, el Matrice iba a 17 m/s a los puntos de búsqueda;
   ahora la búsqueda vuela a la velocidad de crucero de fábrica (el detector necesita tiempo para ver);
   (b) persiguiendo, choques contra el borde (3 de 40 en carrera rápida del Matrice): la geovalla usa ahora una
   distancia de seguridad fija (radio + 1 m) y frena con la aceleración de fábrica, no con la ajustada, y la
   referencia del reintento se recorta dentro de la arena; (c) un Matrice persiguiendo a 4 m/s se metió en un
   árbol: con 0,1 s de reacción y frenando al 90 %, no contaba con lo que tarda la frenada en llegar a esa
   aceleración (la rampa del tirón, a/jerk). Ahora el tiempo de reacción de Collision Prevention nunca baja de
   ½·a/jerk (0,36 s en el Matrice ajustado; los de fábrica no cambian). Con eso: carrera rápida del Matrice 36/40 →
   40/40 y persecución en equipo con 2 PX4 38/40 → 40/40, sin choques.
8. **Qué queda de "entrenar" sin red neuronal.** Todo lo anterior es ajuste de parámetros medido, no aprendizaje:
   el dron no mejora con la experiencia de vuelo, se eligen mejores parámetros en simulación y se validan en
   semillas nuevas. Es lo que se hace con los drones reales (ajuste por aparato en PX4); una red neuronal se deja
   para más adelante (lección 3d y sección 6).

## 4. Resultados

Ver `eval/resultados.md` (3 drones × 20 escenarios: viento, ruido, terrenos, metas, lluvia, densidad, carrera,
objetivo en movimiento o que huye, bosque de densidad máxima y almacén normal y estrecho; cada uno con mapa conocido y
desconocido). Con los parámetros ajustados por dron y las trayectorias cosidas (25-09-2026):

- **56 de 60 combinaciones al 100 % en los dos modos de mapa**, incluidos los 9 escenarios densos nuevos (bosque de
  densidad máxima, almacén y almacén de pasillos de 2,4 m, los tres drones). Falla el PX4 con viento de 10 m/s (su
  límite: aterriza fuera o, con ráfagas fuertes, choca) y, alguna vez, la persecución del vehículo que huye (sin
  choques; 3 vuelos por casilla son pocos para ese escenario: ver el banco de pruebas).
- **Tiempo medio** en las 56 combinaciones: 22,1 s con mapa conocido y 24,5 s con mapa desconocido (+11 %: en el
  almacén estrecho, sin conocerlo, replanifica ~70 veces). Replanificaciones por vuelo: 3,8 y 10,3.
- **Ajuste por dron frente a fábrica** (validación en 60 vuelos con semillas nuevas en mapas densos y con viento):
  Mini 28,9 → 24,7 s, PX4 36,7 → 33,7 s, Matrice 27,4 → 22,7 s, sin perder éxito ni añadir choques. Banco de
  robustez de 40 semillas tras las correcciones de la lección 3e.7: carrera rápida del Matrice en precipicios con
  mapa desconocido 40/40 y persecución en equipo con 2 PX4 40/40, sin choques.
- **Objetivo que huye (fase 4):** con 30 semillas (PX4, mixto, `eval/bench.py`): 83 % de capturas con un dron, con
  una mediana de 103 s (lo pierde y lo vuelve a buscar varias veces).
- **Búsqueda (fase 3, `eval/resultados_busqueda.md`):** 53 de 54 casillas al 100 %, sin choques (en la otra, un
  Matrice con lluvia no encontró la meta a tiempo en 1 de 3 vuelos). 1,3-2 falsas alarmas por vuelo, todas descartadas. En esa tabla la
  bayesiana parece la más rápida (30 s frente a 40 y 43 s), pero son siempre los mismos 3 mundos por casilla: **con
  60 semillas (`eval/bench.py`, PX4, mixto, colinas) las tres tardan lo mismo de media** (~24 s ± 3-6 s) y la
  diferencia está en el peor caso: percentil 90 de 31 s la bayesiana, 33 s fronteras y 40 s el barrido. Tampoco
  cambian nada una altura de búsqueda de 12 m ni pasadas del barrido cada 16 m (30 semillas).
- **Persecución en equipo (fases 4 + 5, `eval/resultados_persecucion.md`, 30 semillas):** 1 dron 83 % (mediana
  103 s), 2 drones 97 % (49 s), 3 drones 97 % (60 s), sin choques entre ellos.
- **Enjambre (fase 5, `eval/resultados_enjambre.md`, PX4 en calma):** 100 % y ningún choque entre drones. Con 2 drones
  el barrido encuentra la meta en 17 s (29,8 s con 1); la bayesiana apenas mejora (19 → 17 s): a partir de ~16 s
  el tiempo es despegar, llegar a la zona y confirmar. Con 60 semillas (bayesiana, mixto): media 23,9 → 21,1 → 21,7 s
  con 1, 2 y 3 drones; lo que sí mejora con 3 es el peor caso (percentil 90: 31 → 24 s). 0 choques en 120 vuelos.
  En zonas de 40 × 30 m el enjambre rinde poco en búsqueda; donde marca la diferencia es persiguiendo.
- Replanificaciones por el mapa: de media entre 0,3 y 34 por vuelo (más en persecuciones). Cada una tarda ~15 ms.

## 5. Fases del proyecto

| Fase | Contenido | Algoritmos |
|---|---|---|
| **1 (hecha)** | Física realista, viento, sensores con ruido, estimación, control, planificación y misión con mapa conocido | Dryden, EKF, cascada PX4, A*, TOPP, Collision Prevention |
| **1b (hecha)** | Terreno con relieve y precipicios, densidad, viento que interactúa con obstáculos, lluvia, meta en azotea/cima/aire, modo carrera | Mapa de alturas, estelas de viento, límite de velocidad por alcance de sensores, márgenes según la covarianza |
| **1c (hecha)** | Carrera contra un objetivo en movimiento (suave, medio, rápido, variable) | Kalman de velocidad constante, punto de intercepción, persecución predictiva con línea de visión, geovalla |
| **2 (hecha)** | El dron construye su mapa con sensores ruidosos (ya no conoce el mundo); replanificación. Pendiente: trayectorias B-spline | Mapa de ocupación con log-odds (OctoMap), ESDF, A\* con replanificación (D\* Lite descartado, ver 3b.5); B-splines tipo **EGO-Planner** pendientes |
| **3 (hecha)** | Misión de búsqueda: meta desconocida, cámara con cono y oclusión, zona designada, niebla de guerra y mapa de calor | **Búsqueda bayesiana**, exploración por fronteras (**FUEL**), cobertura boustrophedon, confirmación de detecciones |
| **4 (hecha)** | Objetivo que HUYE del dron y se esconde tras edificios; ya no emite su posición | Persecución-evasión, cámara en gimbal, búsqueda desde la última posición vista (creencia que se difunde) |
| **5 (hecha)** | Varios drones (2-4) que se reparten la búsqueda | **Voronoi**, subasta voraz (CBBA sin consenso), capas de altura y evitación reactiva (reciprocidad de **ORCA**, sin su programa lineal) |
| **6 (hecha, salvo SITL)** | Banco de pruebas en paralelo con intervalos de confianza, repeticiones de vuelos (grabar, guardar, abrir). Puente a PX4 SITL + Gazebo: no implementado (ver sección 6) | Intervalo de Wilson, registros de vuelo tipo ULog |

## 6. Estado actual y cómo continuar (para un chat nuevo)

**Estado:** fases 1 a 6 terminadas (de la 2 faltan las B-splines; de la 6, el puente a PX4 SITL). Después, mapas
densos (bosque de densidad máxima, almacén interior sin GPS con VIO), trayectorias cosidas al replanificar y ajuste
automático de parámetros por dron (lección 3e). 35 tests en verde. Resultados en la sección 4. Repositorio privado: https://github.com/Sergio-Osorio09/dron-autonomo (`main`).
Comandos en `CLAUDE.md` (tests, servidor, las tres evaluaciones con `--jobs` y el banco de pruebas).

**Cómo funcionan las fases 3-6** (detalle en los docstrings de `search.py`, `target.py` y `swarm.py`):
- `Simulation(search="barrido" | "fronteras" | "bayesiana")`: la zona sale de `search.make_area`; `Mission` va por
  búsqueda → confirmación → crucero/carrera. `pad`, `pad_z` y `gate` son la verdad (evaluación); el dron usa
  `pad_est`, `pad_z_est` y `gate_est`. `FLIGHT_PHASES` (sim.py) dice en qué fases protegen la altura mínima y la visión.
- `motion="huye"`: `target.EvaderMotion` se simula en vuelo (`step`); el dron solo lo ve con `sensors.detect_vehicle`
  (gimbal); `Mission._track` limita la extrapolación y `_lost` crea una búsqueda de blanco móvil sobre toda la arena.
- `swarm.Swarm(n, **cfg)`: n simulaciones con `shared` (mundo, creencia, salida, id, capa de altura); cada misión
  pide su punto con `_team_info` (Voronoi + puntos ajenos). Se usa igual que una `Simulation` (servidor y evaluación).
  Con `motion="huye"` es persecución en equipo: `share_sighting`, `_flank` (cerco) y el vehículo huye del más cercano
  (`EvaderMotion.threats`).
- Repeticiones: todo en el navegador (`recorded`, `startReplay`, `saveFlight`); el servidor no guarda nada.

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
- Tomas cinematográficas (Reveal, paralaje, Pedestal, Dronie, órbita): pedidas por el usuario, aplazadas. Serían
  trayectorias de cámara definidas respecto a un sujeto, con el planificador y Collision Prevention de siempre.
- Red neuronal: aplazada por decisión del usuario. Si se hace, como experimento aparte y comparado en el banco de
  pruebas con lo clásico (por ejemplo, una política que elija la velocidad de crucero, entrenada por imitación).
- Almacén: con mapa desconocido y pasillos de 2,4 m replanifica ~70 veces por vuelo y tarda +10 % (celdas de 1 m;
  celdas de 0,5 m solo en interior costarían ~8× el ESDF).
- Fase 2: trayectorias B-spline de verdad al estilo EGO-Planner. Hay una versión sencilla (`DRON_SMOOTHER=bspline`)
  que no mejora a Chaikin (lección 3d.13); le faltan la factibilidad dinámica y el reparto de tiempos.
- Fase 6: puente a PX4 SITL + Gazebo (enviar las referencias p/v/a por MAVSDK en modo offboard y leer el estado de
  EKF2). No se ha hecho: hay que instalar PX4 y Gazebo, fuera de este proyecto en Python.
- El planificador y Collision Prevention no usan el mismo límite de velocidad (el dron replanifica ~5 veces por
  vuelo al quedarse atrás). Unificarlos se probó y no mejora (lección 3d.11).
- Fase 4 con un solo dron: 83 % de capturas y ~1,5 min (con 2-3 drones, 97 % y ~50 s). Se probaron tres ideas para
  un solo dron (interceptar desde arriba, sesgar la creencia, ir a la última posición conocida): ninguna mejora.
- Rendimiento: 4 drones con mapa desconocido van a 1,7× tiempo real (lo que más cuesta es recalcular el ESDF).
- El tirón (jerk) del perfil de velocidad es aproximado (suavizado), no una curva en S exacta.
- PX4 genérico con viento de 10 m/s: aterriza a ~1 m de la plataforma o choca con ráfagas fuertes (está en su límite).
- La batería se gasta, pero no hay "volver a casa con batería baja" (RTL por batería, como PX4).
- Con mapa desconocido la carrera contra objetivos rápidos es muy variable (15-40 s), igual que con mapa conocido.
- Idea descartada por ahora: modo "carrera sin red" que ignore el límite de velocidad por alcance de sensores.

**Siguiente:** lo pendiente de arriba, o pasar a hardware real: lista de piezas y atributos en `docs/HARDWARE.md`
(~1000-1100 $ por dron: X500 V2, Pixhawk 6C Mini con PX4, RPLIDAR C1, OAK-D Lite, Raspberry Pi 5).

**¿Laya puede mejorar la búsqueda?** (analizado el 25-09-2026) El repositorio github.com/aayushch/laya no es un
modelo: es una aplicación de escritorio que ordena notificaciones (Slack, Gmail, GitHub...) llamando a LLMs externos;
no aporta nada al dron. El modelo Laya de `../laya-drone` (convaiinnovations/laya, 322 M parámetros, solo texto)
pilota a partir de sensores en texto: 78 % de éxito frente al 100 % del experto clásico, a ~25 ms por decisión en una
RTX 4060. Para la búsqueda no mejora lo que hay: no ve imágenes (el detector tiene que ser de visión, tipo YOLO) y la
elección del siguiente punto ya la resuelve la búsqueda bayesiana, que es óptima para su modelo, explicable y tarda
milisegundos. Único uso con sentido: convertir un aviso en texto ("se perdió cerca del río, iba hacia el norte") en
una creencia a priori sobre el mapa; para eso rinde más un LLM general que un modelo de 322 M ajustado a pilotar.

**Preferencias del usuario:** todo en español; algoritmos que se usen en drones reales y datos reales con su fuente;
que se vea bien en pantalla; medir antes de afirmar (tests y evaluación tras cada cambio); explicar las decisiones.
Hacer commit o push solo cuando lo pida.

**Otros proyectos de la carpeta `Laya/`** (no tocar desde aquí): `laya-2048` y `laya-drone` (pilotos con el modelo
Laya). En `laya-drone` hay un ajuste v2 pausado al 92 %: `python training/finetune.py --base models/laya-drone
--out models/laya-drone-v2 --data data/drone_v2_train.jsonl --val data/drone_v2_val.jsonl --epochs 0.6
--lr-enc 2e-5 --lr-head 1e-4 --resume`.
