# Bitácora: física de 4 motores, maniobras y entrenamiento (26-09-2026)

Hitos y aprendizajes en orden, con lo medido. El detalle técnico está en los docstrings de `dron/dynamics.py`
(`Multirotor6DOF`) y en `docs/ENTRENAMIENTO.md`.

## Hito 1 · El dron vuela con cuatro empujes independientes (≈03:40)

`dynamics.Multirotor6DOF`: sólido rígido de 6 grados de libertad con los 4 motores en X y el autopiloto interno de
PX4 a 1 kHz:

    aceleración pedida → ACTITUD (MC_ROLL_P 6,5; rumbo MC_YAW_P 2,8 × MC_YAW_WEIGHT 0,4)
    → VELOCIDAD ANGULAR (PID por eje: MC_ROLLRATE_P/I/D 0,15/0,2/0,003) → MEZCLADOR (±0,707 balanceo y cabeceo,
    ±1 guiñada; si satura, cede primero la guiñada, luego el empuje y por último balanceo/cabeceo)
    → 4 MOTORES (primer orden, 12,5 ms acelerando / 25 ms frenando) → pares (brazo × empuje; guiñada = ±k_m·empuje)
    → J·ω̇ = τ − ω × Jω → orientación → empuje total a lo largo del eje del cuerpo

Datos reales: el X500 de PX4 en Gazebo (PX4-gazebo-models: masa 2,0 kg, inercias 0,0217/0,0217/0,040 kg·m², motores a
±0,174 m, momentConstant 0,016, timeConstantUp/Down 0,0125/0,025 s) y las ganancias de fábrica de PX4. Mini 3 y
Matrice 350: brazos de sus especificaciones (el Matrice, 895 mm de diagonal), inercias con las proporciones del X500
(ESTIMADO) y ganancias escaladas por su autoridad de par (par máximo / inercia).

Es lo que hace un piloto con las dos palancas: izquierda empuje y guiñada, derecha cabeceo y balanceo. Para acelerar
hay que inclinarse Y dar más empuje; para frenar, inclinarse al revés antes de llegar. `sticks` guarda esas 4
órdenes y `u` la salida de cada motor (se ven en la interfaz).

Medido (15° de inclinación pedidos): 90 % en 0,24-0,27 s y sin pasarse, en los tres drones; en vuelo estacionario,
los cuatro motores iguales.

## Aprendizaje · La guiñada es el eje débil

1. Con MC_YAW_P entero, el bucle de rumbo y el de velocidad de giro se peleaban (oscilaba 15-30°): PX4 lo pondera
   con MC_YAW_WEIGHT = 0,4.
2. Sin feedforward de la velocidad de giro pedida, el dron iba por detrás y el integrador se cargaba: se añade como
   PX4.
3. El Matrice se pasaba ~20° con cualquier ganancia: no era el control sino la FÍSICA. Frenar un giro de 100°/s con
   poco par de guiñada lleva esos 20° (ω²/2α). PX4 lo evita limitando también la ACELERACIÓN del rumbo pedido
   (MPC_YAWRAUTO_ACC, 60°/s²): ahora los tres giran 90° pasándose menos de 3°.
4. El par de reacción de una hélice por unidad de empuje crece con su diámetro: k_m ≈ 0,016·D/10" (Mini ~6":
   0,0096; Matrice 21": 0,03; ESTIMADO).

## Hito 2 · Toda la simulación vuela con los 4 motores (≈04:00)

Física por defecto `physics="6dof"` (`DRON_PHYSICS=simple` vuelve a la de antes, para comparar). Aterrizaje,
carrera, bosque extremo sin mapa, Mini con viento de 8 m/s y ráfagas fuertes y Matrice en precipicios: todo llega.
Los 41 tests pasan con ella.

Banco de pruebas (40 semillas, 6 grados de libertad frente a simple): carrera PX4 mixto sin mapa 11,8 / 11,6 s;
búsqueda 27,1 / 27,2 s; carrera contra un objetivo rápido 15,1 / 15,7 s, todos 40/40. Pero 1 choque del Mini en el
bosque extremo y 2 del Matrice en precipicios (0 con la física simple): los parámetros de seguridad se ajustaron con
un dron que se inclinaba sin inercia de rotación; ahora tarda en girar y en frenar. → Reentrenar con la física real.

## Aprendizaje · Rendimiento

El bucle interno a 1 kHz con matrices de numpy de 3×3 costaba 0,37 ms por paso de 5 ms (un 25-50 % más de cálculo):
en floats de Python (orientación como 9 números, Rodrigues y Gram-Schmidt a mano), 0,09 ms, 4 veces menos, con las
mismas respuestas. Los tests (41): 57 s.

## Hito 3 · Las palancas y los motores, en pantalla (≈04:05)

Tarjeta flotante en la vista 3D (junto a la brújula): las dos palancas tal como las movería un piloto (izquierda:
empuje ↕ y guiñada ↔; derecha: cabeceo ↕ y balanceo ↔) y la potencia de cada motor en X (verde poco, rojo a tope).
Se ve, por ejemplo, que para acelerar el autopiloto cabecea hacia delante y a la vez sube el empuje, y que en un giro
los motores de un lado trabajan más que los del otro.

## Aprendizaje · El viento fuerte es un límite físico, no de control

PX4 genérico con viento medio de 8 m/s y ráfagas fuertes: 7 choques en 40 vuelos con las dos físicas. Las ráfagas
del modelo Dryden llegan a 17-19 m/s, casi el doble de lo que soporta el dron (10 m/s según su perfil): va a la
inclinación máxima (45°) y aun así el viento lo arrastra. Ningún ajuste del piloto lo evita; lo realista sería no
despegar o abortar la misión (pendiente: "no volar si el viento supera el límite del dron", como las comprobaciones
previas al vuelo de DJI y PX4).

## Hito 4 · Entrenamiento con la física real (≈04:04)

`eval/entrenar.ps1` (proceso independiente): los 9 pilotos×drones con la física de 4 motores. El genoma añade el
"autotune" del autopiloto: factores sobre las ganancias de actitud (att_k) y de velocidad angular P/I/D (rate_pk,
rate_ik, rate_dk) y sobre las de posición y velocidad (xy_p_k, vel_p_k: con qué firmeza se inclina para acelerar y
frenar). Ronda 1: 20 individuos, 16 generaciones, 18 carreras por individuo; ronda 2: 24 × 24 partiendo de lo
adoptado. Los resultados de la física simple quedan archivados en `eval/evo_simple/`.

## Hito 5 · Prueba de maniobras (`eval/maniobras.py`, ≈04:15)

Como los vuelos de prueba con los que se ajusta un dron real, en campo abierto: escalón de 10 m (subida, cuánto se
pasa, cuándo queda estable a ±0,3 m), parada desde el 80 % de la velocidad máxima, círculo de 8 m de radio a 6 m/s y
giro de 180°. Con las ganancias de fábrica y 4 motores:

| dron | escalón 10 m: subida · se pasa · estable | frenada | círculo: error RMS | giro 180° |
|---|---|---|---|---|
| Mini | 1,62 s · 0,50 m · 7,3 s | 12,0 m | 0,64 m | 2,9 s |
| PX4 | 1,67 s · 0,50 m · 6,1 s | 6,2 m | 1,21 m | 3,5 s |
| Matrice | 1,62 s · 0,53 m · 7,5 s | 27,1 m | 0,55 m | 2,9 s |

La frenada es física (v²/2a: el Matrice a 18 m/s necesita 27 m). Lo mejorable era el posicionamiento: las ganancias
de posición y velocidad de fábrica de PX4 tardan 6-7 s en estabilizarse y se pasan medio metro.

## Hito 6 · Autotune con algoritmo genético (`eval/autotune.py` → `dron/autotune.json`, ≈04:25)

Genoma: factores sobre las ganancias de actitud y de velocidad angular (P/I/D) del autopiloto y sobre las de posición
y velocidad. Coste: tiempo en estabilizarse tras el escalón + 2 × lo que se pasa + 3 × error RMS del círculo +
0,3 × tiempo del giro de 180°, sobre 3 drones con la dinámica aleatorizada (empuje ±20 %, retardos ±30 %, inercias
±20 %: tiene que valer aunque el dron real no sea exactamente el del modelo). Validación en otros 3 drones
aleatorizados; se adopta solo si mejora. Se aplica al dron en TODAS las misiones (es un ajuste del vehículo); en
carrera, si el piloto tiene ganancias entrenadas, esas lo sustituyen (se entrenan partiendo de él).

PX4 (validación: coste 11,8 → 4,5): escalón subida 1,67 → 1,32 s, se pasa 0,50 → 0,01 m, estable 6,1 → 2,3 s;
círculo 1,21 → 0,40 m (−67 %). Aprendió a ser más firme en posición (×1,46) y sobre todo en velocidad (×2,0: se
inclina con más decisión para acelerar y frenar), con más P y D en la velocidad angular (×1,8 y ×2,2: más "reflejos"
y más amortiguación), menos integral (×0,67) y algo menos de P de actitud (×0,74).

Efecto del autotune del PX4 en las misiones (30 semillas, física de 4 motores, fábrica → autotune; 0 fallos nuevos):
aterrizaje en mixto 20,8 → 18,2 s; con viento de 6 m/s y ráfagas 26,0 → 19,4 s (−25 %, y 29/30 → 30/30: con
ganancias más firmes aguanta mejor las ráfagas); azotea en la ciudad sin mapa 20,6 → 17,9 s; carrera en bosque
extremo sin mapa 16,4 → 16,0 s; búsqueda 27,0 → 28,0 s (igual dentro del ruido).

Mini (validación: coste 11,05 → 3,99, −64 %). En las misiones (30 semillas): aterrizaje en mixto 19,0 → 18,3 s; con
viento de 8 m/s y ráfagas fuertes 26,5 → 18,3 s (−31 %; 28/30 → 29/30); carrera en bosque extremo sin mapa
16,1 → 15,5 s y desaparece el choque de la semilla 3007 (que ya aparecía con 4 motores y ganancias de fábrica);
almacén estrecho sin mapa 37,8 → 34,9 s.

## Hito 7 · Autotune de los tres drones (≈04:45)

| dron | coste (validación) | escalón 10 m: estable · se pasa | círculo: error RMS |
|---|---|---|---|
| Mini | 11,05 → 3,99 | 7,3 → 2,5 s · 0,50 → 0,00 m | 0,64 → 0,21 m |
| PX4 | 11,79 → 4,52 | 6,1 → 2,3 s · 0,50 → 0,01 m | 1,21 → 0,40 m |
| Matrice | 11,11 → 4,06 | 7,5 → 2,6 s · 0,53 → 0,00 m | 0,55 → 0,18 m |

Se colocan casi 3 veces más rápido, sin pasarse, y siguen curvas con 3 veces más precisión. Los tres aprendieron lo
mismo: ganancia de velocidad al doble (se inclinan con decisión para acelerar y frenar; está en el tope del rango,
2×: MPC_XY_VEL_P_ACC pasaría de 1,8 a 3,6, dentro de lo que admite PX4) y más derivada en la velocidad angular.

## Hito 8 · Guiado de interceptor y reserva de empuje (≈04:55)

Con 4 motores, el Matrice chocaba en carreras por precipicios con ráfagas (3 de 40): pasaba a 1,2 m de la puerta a
9 m/s y su inercia lo llevaba contra lo que hubiera detrás. Cuatro cambios, todos de drones reales:
1. **Navegación proporcional** en los últimos metros (la ley de guiado de los interceptores: aceleración =
   N · velocidad de cierre · giro de la línea de visión, N = 3): corrige antes. Carreras un 1-4 % más rápidas.
2. **La curva que queda tiene que caber**: si la aceleración lateral necesaria para llegar a la puerta
   (≈ 2·v²·sen θ / d) pasa de la que da la inclinación máxima, velocidad justa para tomarla, como un piloto que
   frena para una curva cerrada. La semilla 3002 ya llega.
3. **El sprint vigila también hacia donde SE MUEVE**, no solo hacia donde quiere ir: con inercia no es lo mismo.
4. **MPC_THR_XY_MARG = 0,3** (PX4): subiendo a tope, se reserva un 30 % del empuje para el control horizontal. Sin
   ella, un Matrice subiendo junto a un edificio no podía apartarse (la semilla 3017 ya llega).
Matrice en precipicios con ráfagas: 37/40 y 17,5 s → 39/40 y 15,0 s. Queda 1 choque: un arbusto bajo que ve tarde
mientras baja en diagonal (queda por debajo de su línea de visión).

## Hito 9 · Primer piloto entrenado con la física real (≈04:44)

Clásico con el PX4 (validación en 48 carreras nuevas): 100 % → 100 %, 0 choques, 18,3 → 17,0 s. Aprendió a usar toda
la aceleración que se le permite (acc_k 1,8), a ser más firme en velocidad (×1,8) y en velocidad angular (P ×1,44) y
con menos derivada (×0,6). El entrenamiento se reinició a las 04:59 con el código final para los demás.

## Hito 10 · Decisiones más rápidas (≈05:10)

Medido en vuelo (con la CPU cargada por el entrenamiento): cada decisión de control (posición → empuje, con
Collision Prevention) tarda 0,19 ms; la de la misión, 1,1 ms de media; la parte lenta era replanificar (~56 ms). Dos
aceleraciones sin cambiar ningún resultado:
- **Perfil de velocidad vectorizado:** 9,3 → 0,5 ms (18×), diferencia 0 en 300 trayectorias aleatorias.
- **A\* con listas de Python** (en vez de leer elementos sueltos de arrays de numpy) guardadas por mapa y holgura, y
  diccionarios solo para las celdas visitadas: 145 → 24 ms por búsqueda en mundos densos (6×), los mismos caminos en
  60 de 60 búsquedas.

## Evaluación principal con todo lo nuevo (≈05:15)

`eval/eval_headless.py` (3 vuelos por casilla, 4 motores + autotune + guiado): 56 de 60 combinaciones al 100 %. Lo
mejor: el PX4 con viento de 10 m/s y ráfagas moderadas pasa de 0 % a 100 % (el autotune aguanta el viento); error
medio de aterrizaje 11 cm. Fallos (1 de 3 vuelos cada uno): Mini y PX4 con viento de 10 m/s y ráfagas fuertes (su
límite) y, nuevos, el Matrice persiguiendo al que huye y en el almacén de pasillos de 2,4 m sin mapa: se miden con 40
semillas.

## Hito 11 · Persecución sin choques: consignas como las de PX4 (≈06:15)

Con 40 semillas, el Matrice en el almacén de pasillos estrechos sin mapa daba 37/40 sin choques (bien: el 1 de 3 era
mala suerte). Persiguiendo al vehículo que huye, en cambio, 36/40 con 2 choques. Mirando los últimos segundos de cada
choque aparecieron cuatro causas, todas de **consignas imposibles para la inercia del dron**:
1. La velocidad pedida saltaba de 2 a 14 m/s: el control se saturaba a la inclinación máxima y no quedaba fuerza
   lateral para esquivar. → **Rampa** a la aceleración que el dron puede dar (MPC_ACC_HOR_MAX), solo en horizontal.
2. La consigna de posición era el propio vehículo, lejos: kp × distancia anulaba la rampa y todo lo demás. →
   **Zanahoria** a ≤ 3 m del dron (como el generador de trayectorias de PX4). Con 1 m capturaba más tarde (79 s de
   media frente a 67 s con 3 m), con el mismo éxito.
3. Pegado al borde, el vehículo "huía" hacia fuera y la consigna lo seguía. → Hacia cada borde, nunca más rápido de
   √(2·a·distancia).
4. Tras bajar en picado en la carrera (−5 m/s), se pasaba de la altura de la puerta y tocaba el suelo. → Lo mismo en
   vertical: √(2 · 0,3 g · diferencia de altura).

Resultado (40 semillas, carrera con objetivo que huye, sin mapa): **Matrice 36/40 con 2 choques → 39/40 sin choques**;
PX4 38/40 sin choques. El único fallo que queda (semilla 3011, en ambos) es un tiempo agotado, no un choque.

**Aprendizaje:** un piloto humano de drones no mueve las palancas a saltos: anticipa la inercia y "suaviza". Para
el autopiloto es lo mismo: cada consigna tiene que ser alcanzable con la aceleración y la inclinación que el dron
tiene; si no, el control se satura y pierde justo el margen que necesita para esquivar.

## Hito 12 · El error del GPS también cuenta cerca del borde (≈06:20)

Comprobando que lo anterior no empeoraba nada, el Mini persiguiendo un objetivo "rápido" chocaba contra el borde en
2 de 30 vuelos. El dron creía estar a 1,6 m de donde estaba de verdad (GPS del Mini: ±0,75 m, 1 sigma) y el límite
hacia el borde se calculaba con la posición estimada. → Margen de **2 sigmas del GPS** además del radio y 1 m.
Resultado (60 semillas cada uno, carrera con objetivo "rápido"): Mini, PX4 y Matrice **60/60**; PX4 "variable" 60/60;
Mini que huye sin mapa 39/40 sin choques; enjambre de 3 PX4 contra el que huye 20/20.

**Aprendizaje:** toda distancia de seguridad calculada con la posición estimada tiene que incluir cuánto se puede
equivocar la estimación (un dron real no sabe dónde está, solo lo estima).

## Evaluación principal (≈06:25)

`eval/eval_headless.py` (3 vuelos por casilla, semillas 500-502): **117 de 120 casillas al 100 %** (antes 116; la del
Matrice persiguiendo al que huye sin mapa pasa a 100 %). Quedan 3 casillas al 67 % (1 choque de 3 vuelos):
- Mini y PX4 con viento de 10 m/s y ráfagas fuertes: su límite físico (ver arriba).
- Matrice en el almacén de pasillos de 2,4 m sin mapa (semilla 500): viene por encima de las estanterías a 7 m/s y
  baja al pasillo; entra a 0,45 m de la estantería (el dron mide 0,9 m y el pasillo 2,4 m: le sobran 0,75 m por lado,
  y el VIO se equivoca 0,25 m) y roza su arista superior. Con 40 semillas nuevas, 37/40 sin choques: es raro, pero
  queda **pendiente**: bajar a un pasillo estrecho más despacio o centrado en él.

**Pendiente también:** en la semilla 3011 del objetivo que huye, ningún dron llega a ver el vehículo en 300 s (se
agota el tiempo buscando): revisar dónde se esconde.

## Hito 13 · Entrenamiento con la física de 4 motores, en marcha (≈06:30)

Ronda 1 (`eval/entrenar.ps1`, 20 individuos × 16 generaciones, validación en 48 vuelos con semillas nuevas):
- clásico/PX4: aptitud 1,695 → 1,717, 100 % sin choques → **adoptado**.
- clásico/Mini: 1,730 → 1,753, 100 % sin choques → **adoptado**.
- clásico/Matrice: el entrenado era peor (1,611 y 1 choque) → **rechazado**, se queda como estaba. El filtro de
  adopción hace su trabajo: el GA encontró algo que iba bien en sus mundos de entrenamiento pero no en otros.
- Siguen híbrido y reactivo × 3 drones y luego la ronda 2 (24 × 24, partiendo de lo adoptado). Progreso en
  `eval/entrenamiento.log`; cada resultado adoptado se guarda solo en `dron/entrenados.json`.

## Hito 14 · Buscar un objetivo en movimiento (carrera con búsqueda)

**Problema (visto en la interfaz):** con "Búsqueda: sí" y objetivo en movimiento, el dron sabía dónde estaba el
objetivo desde el despegue. La simulación quitaba la búsqueda si el objetivo se movía (y la interfaz la ponía en "No"),
y un objetivo en movimiento emitía su posición por radio (rastreador GNSS) a 5 Hz.
**Ahora:** en la carrera, con búsqueda, el vehículo no emite nada (`Mission.camera_only`, como el que huye). El dron lo
busca con la cámara en gimbal por toda la arena (la creencia se difunde a 6 m/s: se mueve) y, en cuanto lo ve, va a
por él a toda velocidad; si lo pierde, vuelve a buscarlo desde donde lo vio por última vez.
Resultado (60 semillas cada uno, objetivo "rápido", sin mapa, búsqueda bayesiana): **Mini, PX4 y Matrice 60/60**.
Enjambre de 3 PX4: 15/15. Barrido y fronteras también (PX4, 30/30).

Correcciones que salieron al medirlo (también mejoran el resto):
- **Cámara de profundidad según la senda de planeo:** bajando en diagonal, cuenta lo que hay por debajo de la
  trayectoria (lo que el dron habrá bajado al llegar ahí), no solo lo que baja en 0,8 s. La azotea de un edificio
  2,3 m más abajo no contaba hasta tenerla a 5,8 m, y a 10 m/s el Matrice ya no podía frenar.
- **Deja de bajar sobre lo que ve delante y debajo:** la cámara frontal limita también la bajada (la inferior no ve
  la azotea que tiene 7 m por delante). En vez de intentar frenar en horizontal, lo sobrevuela.
- **Persecución: consigna vertical a ≤ 1 m y frenada vertical de 0,2 g:** con el vehículo 9 m más abajo, el control
  pedía bajar a 9 m/s y tocaba el suelo; con 0,3 g el Matrice seguía chocando con arbustos y con 0,15 g la persecución
  en equipo era más lenta (141 s frente a menos de 120 s).
- `control.cp_active` nunca se actualizaba (siempre falso): no fiarse de él para depurar.

Sin regresiones: 52 tests; evaluación principal 117 de 120 casillas al 100 % (las 3 restantes, 1 fallo de 3 vuelos:
viento extremo y un tiempo agotado persiguiendo al que huye); Matrice contra el que huye, 38/40 sin choques.
Pendiente: PX4 con búsqueda y objetivo "rápido", semilla 3019 (1 choque en 90 vuelos): baja en vertical sobre el
vehículo, que da vueltas junto a un árbol de 9,5 m, y roza la copa.

## Hito 15 · Embestida: todo el empuje de los 4 motores hacia la meta

**Idea (del usuario):** cuando hay que llegar como sea, que el dron use la fuerza de sus 4 hélices para ir directo a
la meta, muy inclinado, como un FPV de carreras o un interceptor.
**Física:** el arrastre del modelo sale de la velocidad máxima del fabricante a su inclinación máxima, así que
inclinarse más da, de forma natural, más aceleración (g·tan θ) y más velocidad (v_max·√(tan θ / tan θ_fábrica)). El
límite real es sostener el peso: con TWR 2, a más de acos(1/TWR) = 60° el empuje vertical no basta y el dron cae.
**Implementación** (`mission.RAM`, `Mission.ram_tilt/ram_speed/ram_acc`, `control.PositionController.ram`): en la
carrera, con la meta a la vista y el pasillo libre, la inclinación sube a acos(1,1/TWR) (Mini 60°, PX4 57°, Matrice
55°, frente a 35°, 45° y 30° de fábrica) y la velocidad a la que da ese empuje, con un tope de 1,5 × la del fabricante
(Mini 24 m/s, PX4 14,8, Matrice 34,5; ESTIMADO con drones reales desbloqueados: DJI Avata 2 16 → 27 m/s, DJI FPV
27 → 39 m/s). Guía directa con navegación proporcional desde que ve la meta y, contra un objetivo en movimiento, sin
frenar para igualar su velocidad (impactar es llegar). En la interfaz: casilla "Embestida" y "· ¡embestida!" en el HUD.
En un dron real: PX4 lo permite (MPC_TILT_MAX_AIR hasta 85°, MPC_XY_VEL_MAX); un DJI no, con su firmware.

Lo que se aprendió probándolo (60 semillas por caso):
- **Con 80° el dron cae:** empuje vertical cos 80° ≈ 0,17 del total; el Matrice chocó contra el suelo en 4 de 30
  carreras. "Las hélices apuntando a la meta" solo vale picando desde arriba; en vuelo nivelado, el tope es acos(1/TWR).
- **Contra el que huye, no:** esquiva, y lanzarse sin frenar junto a obstáculos subió los choques del Matrice (2 → 5
  en 30) sin capturarlo antes. Desactivada ahí.
- **Pasillo más ancho según el dron (`tuning.py`, `ram_clear`):** a más velocidad, la inercia de un giro se come el
  margen antes. Mini y PX4 no necesitan más (0 choques); el Matrice (6,5 kg, motores lentos) chocaba en 3 de 60 en
  el bosque denso sin ganar tiempo: solo embiste con 1,5 m más de holgura por lado.

Resultado (carrera, mixto, 60 semillas): **Mini 9,6 → 8,2-8,6 s (−10 a −15 %), PX4 10,7 → 9,6-9,9 s (−7 a −10 %)**,
sin choques; el Matrice, igual que antes (su holgura casi nunca se da en el mixto). Con búsqueda y objetivo "rápido",
PX4 60/60. Evaluación principal: 117 de 120 casillas al 100 %, las mismas que sin embestida; 53 tests.
Pendiente (no es de la embestida, pasa igual sin ella): PX4 con objetivo "variable" sin mapa, semilla 7018, 1 choque.

## Entrenamiento: estado

Ronda 1 hasta el momento: adoptados clásico/PX4, clásico/Mini e **híbrido/PX4** (aptitud 1,584 → 1,674 y de 1
choque a 0 en la validación); clásico/Matrice rechazado. El entrenamiento del híbrido del Mini se cortó (el grupo
de procesos murió mientras se lanzaban a la vez bancos de pruebas y la evaluación) y los siguientes terminaron sin
hacer nada. Se relanzó con el código nuevo (embestida y búsqueda de objetivos en movimiento) y 16 procesos en vez de
18, para dejar margen: `eval/entrenar.ps1`, progreso en `eval/entrenamiento.log`.
**Aprendizaje:** no lanzar bancos pesados mientras entrena con casi todos los núcleos.

## Hito 16 · Un solo dron de entrenamiento: Holybro X650; tres pilotos

**Decisión (del usuario):** entrenar un único dron práctico, que se arme por piezas y lleve carga, sin DJI. Elegido
el **Holybro X650** (perfil `x650`, `params.TRAINING`): kit de desarrollo de referencia de PX4, 650 mm, 2,0 kg sin
batería, despegue máximo 6,3 kg, MN4014 330KV con hélices de 15", 6S (datos de Holybro y T-Motor; empuje con hélice
de 15", inercias y motores, ESTIMADOS). Con batería de 10000 mAh y la electrónica de autonomía: 3,65 kg, empuje/peso
2,3, ~25 min. **Carga útil:** `Simulation(payload=kg)` y selector "Carga" en la interfaz (más masa, menos empuje por
kilo, más inercia y potencia); como mucho ~2,6 kg (con 4 kg pesaría 7,7 kg, empuje/peso 1,1: no despega con
seguridad y la simulación lo rechaza). La interfaz solo ofrece el X650; los perfiles mini, px4 y matrice siguen en
el código para las pruebas y las comparaciones.

**Pilotos:** se quedan **clásico** y **híbrido** (los que se entrenan) y **fábrica** (la referencia, sin entrenar).
Retirado el **reactivo** puro: 81-85 % de éxito en validación, sin mejora con el genético; su límite (atascarse en
pasillos en U) es de diseño. Su código sigue como planificador local del híbrido. El algoritmo genético no es un
piloto: es el método (`eval/evolve.py`) que ajusta los parámetros del clásico y del híbrido.

**Primeros datos del X650:**
- Autotune (`eval/autotune.py`): coste 11,8 → 4,5 en la validación. Escalón de 10 m: estable en 2,3 s en vez de
  6,1 s, sin pasarse (antes 0,5 m); círculo a 6 m/s: error 1,21 → 0,40 m.
- Carrera sin mapa, 30 semillas, antes de entrenar: fábrica 16,4 s, clásico 15,5 s, **híbrido 12,9 s**, los tres
  30/30. Con carga (clásico, 20 semillas): 0 kg 12,7 s · 1 kg 13,0 · 2 kg 13,2 · 2,6 kg 13,6 s, todas 20/20.
- Entrenamiento lanzado (`eval/entrenar.ps1`): 3 rondas de clásico e híbrido en el X650 (la última, 32 × 32 con 4
  mundos por escenario), varias horas. Progreso en `eval/entrenamiento.log`.

## Hito 17 · Buscar el cuello de botella real de la velocidad (plan de mejoras, docs/PLAN_MEJORAS.md)

Entrenamiento detenido (a petición) tras la ronda 1 del X650: adoptados clásico (98 → 100 %, 22,9 → 18,3 s en 48
carreras nuevas) e híbrido (92 → 98 %, 18,3 → 15,4 s). Se midió qué frena al dron (60 semillas por caso):
- **Fallo corregido:** el planificador local del híbrido solo usaba las lecturas de los telémetros en el paso en que
  llegaban (20 Hz de 200): en 9 de cada 10 decisiones iba recto, sin repulsión. Ahora usa la última lectura, con la
  repulsión y la frenada a 1/10 (el mismo efecto medio, sin pulsos): mixto 11,4 → 11,3 s, bosque 15,1 → 14,7 s.
  Reescalados también los parámetros guardados (entrenados.json, evo_*.json) y los rangos del genético.
- **Tirón de carrera a 8 m/s³** (MPC_JERK_MAX de PX4, antes el de misión, 4): clásico persiguiendo el objetivo rápido
  18,1 → 16,4 s (−9 %), sin más choques.
- **Probado y descartado** (queda en el código, desactivado): vuelo más horizontal en A* (DRON_NEAR=h), ventana dinámica
  en el híbrido (DRON_DWA=1), aproximación nivelada a la puerta (DRON_APPROACH=6) y un LiDAR de 40 m: ninguno acelera
  más de un 2-4 %.
**Aprendizaje:** medir antes de optimizar. Tres hipótesis razonables (sensores, altura, anticolisión) no eran el
límite; el siguiente es el seguimiento de la trayectoria (el dron va por detrás de la velocidad pedida).

## Hito 18 · Mejora A: que el dron siga lo que se le pide (prealimentaciones de planitud diferencial)

Medido: el clásico iba por detrás de la velocidad pedida un 30 % del tiempo (1,5 m/s de error medio).
- **Prealimentación del arrastre** (`control.DRAG_FF`, Faessler et al., RA-L 2018): el control suma de antemano la
  aceleración que se come el arrastre a la velocidad pedida (a 8 m/s, ~4,4 m/s² en el X650), en vez de esperar al
  error. Es lo que más aporta.
- **Prealimentación del giro** (`dynamics.TILT_FF`, Mellinger y Kumar, ICRA 2011): la velocidad angular con la que
  gira el eje de empuje pedido va directa al lazo de velocidad angular (acotada a 1 rad/s: un escalón no es una
  trayectoria y su "derivada" hacía pasarse 1-2° de inclinación). En carrera no cambia el tiempo; en maniobras, sí.
- **Autotune rehecho** con el control nuevo, partiendo de lo adoptado (ahora también compara con las ganancias
  anteriores, no solo con las de fábrica): coste −1 a −2 % en los cuatro drones.
- Probado y descartado: rampa en la velocidad del híbrido (`DRON_REACT_RAMP`): más lento y con choques; al híbrido
  le sirve pedir de más.

Resultado (X650, carrera sin mapa, 60 semillas por caso, antes → después):

| | clásico | híbrido |
|---|---|---|
| mixto | 12,52 → 12,04 s (−4 %) | 10,74 → 10,39 s (−3 %) |
| bosque denso | 17,35 → 16,62 s (−4 %), choques 1 → 0 | 14,05 → 13,97 s, choques 0 → 1 |
| ciudad | 12,68 → 12,25 s (−3 %) | 10,82 → 10,56 s (−2 %) |
| objetivo rápido | 16,91 → 15,28 s (**−10 %**) | 14,35 → 14,06 s (−2 %) |

Maniobras del X650 (`eval/maniobras.py --autotune`): círculo de 8 m a 6 m/s, error 0,40 → **0,14 m**; frenada desde
el 80 % de la velocidad máxima 6,0 → 5,6 m; escalón de 10 m se pasa 0,09 m (antes 0,01) y se estabiliza en 2,25 s.
Evaluación principal: 117 de 120 casillas al 100 % (las 3 de siempre: viento extremo y pasillos de 2,4 m sin mapa,
sin choques en esta última). 56 tests.

## Hito 19 · Mejora B: que replanificar y planificar no le cuesten velocidad al clásico

Medido: con el mapa conocido el clásico era un 13-16 % más rápido que sin él (mixto 10,6 frente a 12,0 s; bosque
14,0 frente a 16,6 s): 19-24 replanificaciones por carrera; el 15-21 % hacían caer la velocidad pedida más de 2 m/s.
- **Lo que funcionó: planificar con la aceleración física del dron.** El genético no podía pasar de acc_k 1,8
  (5,1 m/s² en el X650) y el dron da 8,8 (90 % de g·tan 45°). Con la prealimentación del arrastre (hito 18) ya sigue
  trayectorias más agresivas: acc_k 2,9 → −5 a −7 % en mixto, bosque, ciudad, objetivo rápido, montaña y viento
  (60 semillas por caso, sin más choques). Adoptado a mano para el clásico del X650 (`entrenados.json`, con nota) y
  rango del genético ampliado a (1,0, 3,0).
- **Probado y descartado** (queda en el código, desactivado): variable de progreso en el seguimiento (MPCC,
  `DRON_PROGRESS`), coste de lo no visto en A* (`DRON_UNKNOWN_COST`: con 2, 4 choques de 60), perfil de velocidad con
  arrastre (TOPP, `DRON_DRAG_PROFILE`: más fiel pero 1-4 % más lento; pedir algo más de lo que da lo lleva al
  máximo), la misma aceleración en el reactivo del híbrido (sin efecto).

Clásico del X650, carrera sin mapa, 60 semillas por caso (antes de A → después de A → después de B):

| | antes | A | B |
|---|---|---|---|
| mixto | 12,52 s | 12,04 | **11,42 (−9 %)** |
| bosque denso | 17,35 | 16,62 | **15,65 (−10 %)** |
| ciudad | 12,68 | 12,25 | **11,36 (−10 %)** |
| objetivo rápido | 16,91 | 15,28 | **14,72 (−13 %)** |

La distancia al clásico con el mapa conocido (10,6 / 14,0 s) se ha reducido a la mitad. El híbrido sigue siendo el
más rápido (mixto 10,4 s, bosque 14,0 s).
**Hallazgo para el pendiente:** el X650 con viento de 10 m/s y ráfagas fuertes choca en 15-18 de 60 carreras (con
8 m/s, 3-5): su límite de viento de 12 m/s (ESTIMADO) es optimista; hace falta la comprobación de viento antes de
despegar o bajarlo.

## Decisión · Un solo dron; el enjambre, para el futuro

Nos quedamos con un solo dron por vuelo (el X650). El enjambre (fase 5: búsqueda repartida y persecución en equipo)
queda aparcado: su código sigue, pero la interfaz ya no ofrece varios drones y sus 2 pruebas solo corren con
`DRON_SWARM=1` (también el servidor). No acorta el entrenamiento (el genético siempre ha volado un solo dron), pero
sí la validación de cada cambio (tests de ~77 a ~44 s) y la complejidad al diseñar mejoras.

## Hito 20 · Comprobación de viento antes de despegar (X650)

Medido (X650, clásico, mixto sin mapa, 30 carreras y 30 aterrizajes por casilla de viento × ráfagas): sin ráfagas no
choca ni con 12 m/s; lo peligroso son las RÁFAGAS (moderadas: choques desde 6 m/s; fuertes: 13 de 30 a 8 m/s). La
racha pico prevista, viento medio × (1 + 0,6 × nivel de ráfagas) (≈3σ de la turbulencia Dryden a baja altura),
separa perfectamente lo seguro de lo peligroso con un límite de 13 m/s. Ahora, como COM_WIND_MAX de PX4 o un operador
que mira la previsión, si la racha prevista supera el `gust_max` del dron, **no despega** (estado `grounded`, con la
causa; en la interfaz "No despega: demasiado viento"; columna nueva en `eval/eval_headless.py`). Solo el X650 tiene
el límite validado (13 m/s); en los demás perfiles (datos del fabricante) no se aplica. `DRON_WIND_CHECK=0`: sin ella.

## Hito 21 · Mejora C: CMA-ES en el entrenamiento y reentrenamiento nocturno

`eval/evolve.py --opt cmaes` (ahora el optimizador por defecto; `--opt ga`, el genético de antes): CMA-ES (Hansen,
2016) sobre los genes normalizados a sus rangos realistas, partiendo de los parámetros actuales; misma aptitud, mismos
mundos, misma validación en semillas nuevas y mismo formato (la adopción, `apply_evolved.py`, no cambia).
Comparación a igual presupuesto (clásico del X650, 12 × 10 = 120 individuos, 12 carreras cada uno): ninguno mejora
al clásico actual en la validación (±1 %: ya está cerca de su óptimo tras los hitos 18-19); CMA-ES concentra mejor la
población (aptitud media 1,718 frente a 1,628). Se usa CMA-ES por eficiencia; donde más se espera del entrenamiento
es en el híbrido, cuyos parámetros cambiaron con la corrección de las lecturas (hito 17).
Entrenamiento lanzado (`eval/entrenar.ps1`, 27-09-2026 08:43): ronda 1 (λ 14, 30 generaciones, 3 mundos por escenario)
y ronda 2 (λ 16, 40 generaciones, 4 mundos), clásico e híbrido del X650, ~6 h.

## Hito 22 · Resultado del reentrenamiento con CMA-ES (27-09-2026, 08:43-22:33)

- **Híbrido del X650, ronda 1: ADOPTADO.** En 48 carreras nuevas, 15,6 → 14,9 s y de 1 choque a 0 (éxito 96 %).
- Híbrido, ronda 2: en entrenamiento llegó a 14,3 s sin choques, pero en la validación chocó 2 veces → rechazado
  (sobreajuste a sus mundos de entrenamiento). Se queda el de la ronda 1.
- Clásico del X650, rondas 1 y 2: no mejora (16,8 frente a 16,4-16,9 s, y en la ronda 1 con 1 choque) → se queda el
  de la mejora B (acc_k 2,9): ya estaba en su óptimo.
- La ronda 2 fue ~4-5 veces más lenta de lo previsto (~11 min por generación en vez de ~2,4): Windows limitaba los
  procesos en segundo plano mientras el equipo no se usaba. Para entrenamientos largos: plan de energía de alto
  rendimiento o `powercfg /powerthrottling disable /path <python.exe>`.
**Aprendizaje:** el filtro de validación en semillas nuevas es imprescindible: 2 de los 4 entrenamientos "mejoraron"
en sus mundos y habrían empeorado el dron de verdad.
