# Entrenamiento de los pilotos con un algoritmo genético (misión de carrera)

Qué se entrena, cómo, con qué datos reales y qué salió. El código está en `eval/evolve.py` (entrenamiento),
`eval/apply_evolved.py` (adopción solo si valida) y `dron/pilots.py` (los pilotos y sus parámetros entrenados, que
se cargan de `dron/entrenados.json`).

## Referencias: cómo se entrenan los drones rápidos de verdad

| Trabajo | Qué aporta aquí |
|---|---|
| Kaufmann et al., *Champion-level drone racing using deep reinforcement learning*, Nature 620, 2023 (Swift) | La recompensa: **progreso** hacia la siguiente puerta en cada paso (más un término de percepción y otro de suavidad de mandos) y **−5 al chocar**, que acaba el intento. Dron de 870 g con empuje/peso 4,1; 100 agentes en paralelo, 10⁸ interacciones (50 min en una estación de trabajo). Ganó a tres campeones humanos. |
| Loquercio et al., *Learning high-speed flight in the wild*, Science Robotics 2021 | Referencia de lo realista: en un bosque de 1 árbol por 25 m², su política aprendida completa el 100 % de los vuelos hasta 5 m/s y el 60 % a 10 m/s; los métodos clásicos, 0 % a 10 m/s. Volar rápido entre obstáculos tiene un límite. |
| Foehn et al., *Time-optimal planning for quadrotor waypoint flight*, Science Robotics 2021 | El tiempo mínimo se consigue en el **límite de los actuadores**: por eso el genoma deja que la carrera use hasta el 90 % de la aceleración física. |
| Chen et al., *What matters in learning a zero-shot sim-to-real RL policy for quadrotor control?* (SimpleFlight), 2024 | Para que lo aprendido en simulación valga en un dron real: identificar lo que se puede medir (masa) y **aleatorizar lo incierto** (el coeficiente de empuje ±30 %); aleatorizar de más solo dificulta el aprendizaje. |
| Estudios de ajuste de PID de cuadricópteros con GA, estrategias evolutivas, evolución diferencial y PSO (p. ej. *Evolutionary Intelligence*, Springer 2019; MDPI Electronics 2024) | Los algoritmos evolutivos ajustan bien parámetros de control continuos con una función de coste de simulación; el GA consigue menos sobreoscilación y tiempo de establecimiento que el ajuste manual. |

## Qué es aquí "entrenar" (y qué no)

No hay red neuronal: se evolucionan los **parámetros** de dos pilotos clásicos para la carrera, como hacen los
trabajos de ajuste evolutivo de controladores. Lo aprendido son números con significado físico (márgenes en metros,
aceleraciones, tiempos), así que se pueden llevar a un dron real con PX4 y revisar uno a uno.

**Genoma** (rangos en `RANGES` de `eval/evolve.py`):

- Comunes: frenada de Collision Prevention (`cp_acc`, fracción de la aceleración), margen (`cp_margin`, m) y reacción
  (`cp_react`, s); descuento del alcance de los sensores (`sense`, m); fracción de la velocidad máxima sin ver la meta
  (`race`); cuánto se puede superar el límite de "frenar dentro de lo que ven los sensores" (`vs_k`, 1-2); factor de
  la aceleración del planificador (`acc_k`, hasta el 90 % de g·tan(inclinación máxima)); medio ancho del pasillo del
  sprint (`corridor`, m); segundos de guía terminal (`term`).
- Clásico: además, la holgura del planificador (`margin`) y cuántas σ de incertidumbre se le suman (`sigma_k`).
- Reactivo: sus 7 ganancias (repulsión, radio de influencia, peso lateral, frenada ante un obstáculo, cuándo se da
  por atascado, cuánto rodea y durante cuánto).

**Aptitud** de un individuo = media sobre sus vuelos de:

- llega → 1 + (1 − tiempo/60): más rápido, más puntos;
- no llega en 45 s → la fracción del camino recorrida (progreso, como Swift);
- choca → progreso − 5 (la penalización de Swift).

**Mundos de entrenamiento**: carreras SIN conocer el mapa en bosque extremo, ciudad densa con colinas, precipicios,
almacén (sin GPS), montaña con meta en el aire y carrera con búsqueda (tiene que identificar la meta), 2 mundos de
cada uno, **los mismos para todos los individuos y generaciones**.

**Aleatorización de la dinámica** en cada mundo (el piloto no lo sabe; lo "real" es otro dron):
empuje ±20 % (la relación empuje/peso es un dato estimado), arrastre ±20 %, retardos de motores y de actitud ±30 %,
viento de 0 a 4 m/s en dirección al azar y a veces con ráfagas. La masa no se aleatoriza (se mide).

**Algoritmo**: GA de codificación real. Población de 16 (el piloto actual, 4 mutaciones suyas y el resto al azar),
torneo de 3, cruce BLX-α con α = 0,3, mutación gaussiana con probabilidad 0,3 por gen y σ del 12 % al 5 % del rango,
élite de 2, 12 generaciones. Como la simulación es determinista, un individuo ya volado no se repite (caché).

**Validación**: el mejor y el de partida vuelan 8 mundos NUEVOS por escenario (48 vuelos). `eval/apply_evolved.py`
solo lo adopta si la aptitud mejora y no choca más.

## Lecciones

1. **Pocos mundos y cambiantes = seleccionar suerte.** La primera versión volaba 1 mundo por escenario, distinto en
   cada generación: la aptitud dependía más del mundo que del individuo y el "entrenado" validó PEOR (55 % de
   carreras completadas frente a 62 %). Con los mismos mundos para todos, la comparación es justa.
2. **"Llega o no llega" no enseña nada.** Con esa aptitud el algoritmo se estancó en la 2.ª generación: un
   individuo que se quedaba a 1 m valía lo mismo que uno que no despegaba. La aptitud por progreso (Swift) da
   dirección.
3. **Antes de entrenar, arreglar el algoritmo.** Entrenando se vio que el reactivo se quedaba parado a 23 m de la
   meta: al agotarse la duración de su recta de referencia, la misión creía haber pasado la meta. Ningún ajuste de
   parámetros podía arreglar eso; el arreglo lo llevó del 82 % al 100 % en carrera (mixto) y del 52 % al 85 % en la
   ciudad densa.
4. **Simulador más rápido = entrenamiento más rápido.** Los rayos contra los obstáculos (tablas precalculadas y solo
   los obstáculos al alcance), los choques (solo los obstáculos cercanos) y Collision Prevention (aritmética simple en
   vez de vectores de numpy de 3 elementos) dan los mismos resultados (diferencia 0 y 10⁻¹⁷) y el simulador va ~2
   veces más rápido (los 40 tests: 145 → 71 s).
5. **La validación es la que manda.** El reactivo del Matrice entrenado no chocaba en sus 12 carreras de
   entrenamiento, pero en 48 nuevas chocó más que antes (5 frente a 2): sobreajuste a pocos mundos con un dron pesado
   y agresivo. `apply_evolved.py` lo rechazó. La ronda 2 usa 3 mundos por escenario (18 carreras por individuo).

6. **Límites realistas, aunque la validación diga que sí.** El clásico del Matrice aprendió 0,06 s de reacción y a
   correr un 72 % por encima del límite de sus sensores: validó sin choques, pero con ráfagas y precipicios chocaba
   el 5 % (2 de 40). La latencia real de un dron (percepción + cálculo + motores) ronda 0,1 s: ahora la reacción no
   baja de 0,1 s ni se supera en más de un 50 % el límite de los sensores, y `apply_evolved.py` rechaza lo
   entrenado fuera de esos límites (se retiraron el clásico del Matrice y del Mini y el híbrido del Mini; la ronda 2
   los reentrena dentro de ellos).

## Resultados (validación en 48 carreras con semillas que el entrenamiento no vio)

| piloto / dron | antes | entrenado | adoptado |
|---|---|---|---|
| reactivo / PX4 | 77 % completadas, aptitud 1,441, 0 choques | **81 %**, aptitud **1,466**, 0 choques | sí |
| reactivo / Mini | 75 %, aptitud 1,364, 1 choque | igual (el mejor era el de partida) | no |
| reactivo / Matrice | 65 %, aptitud 1,094, 2 choques | 69 %, pero 5 choques (aptitud 0,813) | no: choca más |
| clásico / PX4 | 100 %, 20,0 s de media, 0 choques | **100 %, 18,4 s (−8 %)**, 0 choques | sí |
| clásico / Mini | 100 %, 17,4 s, 0 choques | 98 %, **15,3 s (−12 %)**, 0 choques | sí (más rápido; 1 de 48 sin tiempo) |
| clásico / Matrice | 100 %, 17,7 s, 0 choques | **100 %, 16,9 s (−5 %)**, 0 choques | sí |
| híbrido / PX4 (ronda 2) | 94 %, 17,0 s, 0 choques | 90 %, 16,2 s, 0 choques | no: más rápido pero llega menos |
| híbrido / Mini (ronda 2) | 92 %, 13,8 s, 1 choque | **94 %, 14,3 s, 0 choques** (aptitud 1,582 → 1,687) | sí |

Lo que aprendió el reactivo del PX4 (antes → después): frenar más fuerte (`cp_acc` 0,89 → 0,95) con menos margen
(0,47 → 0,36 m) y menos tiempo de reacción (0,20 → 0,12 s); correr un 32 % por encima del límite de sus sensores
(`vs_k` 1,32) y con un 9 % más de aceleración; darse por atascado antes (2,5 → 1,7 s) y rodear durante más tiempo
(6 → 7,6 s). El clásico del PX4 aprendió sobre todo a **acelerar**: usa un 68 % más de aceleración al planificar
(`acc_k` 1,68: 3 → 5 m/s², justo el MPC_ACC_HOR_MAX de PX4), frena más fuerte (0,89 → 0,94) con menos margen (0,47 →
0,36 m) pero más holgura por la incertidumbre del GPS (`sigma_k` 1,34 → 1,85), un pasillo de sprint más ancho
(0,6 → 1,0 m) y más guía terminal (0,7 → 1,0 s). Los demás (reactivo con Mini y Matrice, clásico con los tres) se entrenan con `eval/entrenar.ps1`; sus
resultados quedan en `eval/entrenamiento.log`, `eval/evo_*.json` y, si validan, en `dron/entrenados.json`.

**Generaliza a lo que no entrenó.** Los parámetros de carrera se entrenaron con metas quietas, pero se aplican a toda
carrera. Con el clásico entrenado (40 semillas, 0 choques en todos): PX4 contra un objetivo rápido 17,7 → 15,4 s,
Mini 16,2 → 13,0 s; PX4 contra el que huye, 92 % de capturas como antes y media 74 → 64 s; 2 PX4 persiguiéndolo en la
ciudad 46,0 → 34,8 s (40/40).

## Cómo lanzarlo y aplicarlo

```bash
python eval/evolve.py --pilot reactivo --profile px4 --out eval/evo_reactivo_px4.json   # ~30-60 min con 18 procesos
python eval/apply_evolved.py                                                          # adopta solo lo validado
```

`eval/train_all.py` entrena los 6 (2 pilotos × 3 drones) seguidos y aplica lo validado tras cada uno.
`eval/entrenar.ps1` lo lanza como proceso independiente (sigue aunque se cierre todo): una ronda con los que faltan y
otra más larga (20 individuos, 20 generaciones) que parte de lo ya adoptado, como un entrenamiento por etapas.

## El piloto híbrido

El Mini reactivo fallaba 8 de sus 12 carreras de validación en el almacén: avanzaba (62-95 % del camino) pero se
perdía en los pasillos. Ningún parámetro arregla un mínimo local grande: hace falta guía global. El piloto
**híbrido** es el reactivo yendo hacia un punto de mira `look` metros por delante en un camino A* sobre el mapa
(recalculado cada segundo, ~15-35 ms), la arquitectura de la navegación de ROS (planificador global + local). Sin
entrenar, en las 48 carreras de validación del PX4: 45/48 en 17,0 s de media (clásico 48/48 en 20,0 s; reactivo
entrenado 39/48 en 21,0 s), 0 choques. El algoritmo genético ajusta además `look`.

## Hacia un dron real

Lo aprendido son parámetros de PX4 y del planificador con unidades físicas. Para llevarlo a un dron real: medir su
masa, empuje y retardos (identificación del sistema), entrenar con su perfil y una aleatorización centrada en esos
valores, y empezar con las velocidades de validación más bajas. El modelo de empuje del simulador es colectivo (no
simula cada hélice): para evolucionar un control de motores haría falta antes un modelo de 6 grados de libertad
(lección 3i de `CONTEXTO.md`).
