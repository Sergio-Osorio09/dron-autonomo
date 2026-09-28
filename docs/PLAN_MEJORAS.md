# Plan de mejoras (26-09-2026): el X650 más rápido y eficiente

Punto de partida, medido en 10 carreras sin mapa del X650 (`eval/bench.py` y un perfil de tiempos):

| | Híbrido | Clásico |
|---|---|---|
| Tiempo medio | 10,3 s | 12,8 s |
| Tiempo a ≥ 80 % de la velocidad máxima | 0,2 % | 0 % |
| Tiempo por debajo de 3 m/s | 18 % | 32 % |
| Tiempo subiendo o bajando a > 1 m/s | 26 % | 21 % |
| Tiempo con la velocidad pedida recortada por Collision Prevention | 65 % | 13 % |
| Recorrido / línea recta | 1,04 | 1,03 |

Las rutas ya son casi rectas. Lo que se pierde: no va rápido (el límite de frenar dentro de lo que ven los sensores,
12 m, lo deja en ~8 m/s de 12), sube y baja demasiado, y el híbrido pide velocidades que la anticolisión le recorta.

Tiempos estimados de trabajo, sin contar el reentrenamiento (una noche, ~10 h, al final).

| # | Mejora | Qué se espera | Tiempo | Estado |
|---|---|---|---|---|
| 1 | **Vuelo horizontal**: la cercanía en A* se mide en horizontal (el suelo no cuenta) y subir/bajar cuesta 2× | −8 % en el clásico, la mitad de tiempo subiendo/bajando, menos energía | 1-2 h | probado: no acelera ni ahorra; desactivado |
| 2 | **Híbrido coordinado con Collision Prevention**: el planificador local solo pide lo que la anticolisión aceptaría | menos frenazos (65 % del tiempo recortado) | 1-2 h | hecho: corregido un fallo (el híbrido usaba 1 de cada 10 lecturas); la ventana dinámica no acelera, desactivada |
| 3 | **Ir más rápido que el alcance de los sensores con seguridad** (FASTER, Tordesillas et al., T-RO 2021): trayectoria rápida hacia lo desconocido + trayectoria de frenado de emergencia siempre dentro de lo ya visto libre | la mayor ganancia de velocidad sin cambiar hardware | 4-6 h | |
| 4 | **Trayectorias de tiempo mínimo** con límites de empuje e inclinación (MINCO/GCOPTER, Wang et al., T-RO 2022; EGO-Planner, Zhou et al., RA-L 2021) | curvas y aceleraciones al límite del dron | 6-10 h | |
| 5 | **Control para maniobras agresivas**: INDI + planitud diferencial (Tal & Karaman, T-CST 2021) y compensación del arrastre de los rotores (Faessler et al., RA-L 2018) | menos error siguiendo la trayectoria a alta velocidad | 3-5 h | |
| 6 | **CMA-ES en lugar del genético** y objetivo múltiple (tiempo, energía, riesgo) | converge con menos vuelos | 2-3 h (+ entrenamiento) | |
| 7 | **Más realismo**: pérdida de empuje de las hélices con la velocidad, caída de tensión de la batería, efecto suelo; después, puente con PX4 SITL + Gazebo e identificación con registros ULog del dron real | que lo aprendido sirva en el dron real | 2-3 h (física); 1-2 días (SITL) | |
| 8 | **Hardware**: probar en simulación un LiDAR de más alcance (p. ej. ~40 m frente a 12 m) | sube directamente el límite de velocidad | 30 min | probado: solo −2 a −4 %; no compensa |

Al terminar: reentrenar clásico e híbrido en el X650 (`eval/entrenar.ps1`) con el código mejorado.

Resultados de cada paso: en `docs/HITOS.md`.

## Lo aprendido al medir (26-09-2026, 60 semillas por caso)

La hipótesis inicial (el límite es el alcance de los sensores) era falsa. Medido en el X650:
- Con un LiDAR de 40 m en lugar de 12 m, solo −2 a −4 %; la velocidad máxima media sigue en ~7,5 m/s.
- Volar más horizontal (la mitad de tiempo subiendo o bajando) no acelera ni ahorra energía.
- Que la anticolisión no recorte (ventana dinámica) no acelera.
- Aproximarse nivelado a la puerta (la trayectoria la cruza a 7 m/s en vez de 3,3) no acelera: el dron replanifica
  20-25 veces por carrera y al final manda la guía terminal.
- **Lo que sí:** el tirón de la carrera a MPC_JERK_MAX (8 m/s³): el clásico persiguiendo el objetivo rápido −9 %
  (18,1 → 16,4 s), bosque −6 % en una tanda y igual en otra, sin más choques.
- El híbrido ya es un 15-20 % más rápido que el clásico: no depende del perfil de velocidad de la trayectoria.

## Plan revisado

| # | Mejora | Por qué ahora | Tiempo |
|---|---|---|---|
| A | **Seguimiento de la trayectoria (5)**: el dron va por detrás de la velocidad pedida (p. ej. pide 10,6 m/s y lleva 4,4); control con prealimentación de aceleración y planitud diferencial, compensación del arrastre (INDI si hace falta) | es el siguiente cuello de botella medido | 3-5 h · **hecho** (hito 18): −2 a −10 %, círculo 0,40 → 0,14 m |
| B | **Trayectorias de tiempo mínimo (4)** que se cosen sin volver a acelerar en cada replanificación | el clásico replanifica 20-25 veces y cada vez rehace el perfil | 6-10 h · **hecho** (hito 19): planificar con la aceleración física, −5 a −7 % (−9 a −13 % con la A) |
| C | **CMA-ES + objetivo múltiple (6)** y reentrenar clásico e híbrido en el X650 con todo lo nuevo | las ganancias del híbrido cambian con la corrección de las lecturas | 2-3 h + una noche · **hecho** (hitos 21-22): híbrido 15,6 → 14,9 s y 0 choques; el clásico ya estaba en su óptimo |
| (hecho) | **Comprobación de viento antes de despegar** (hito 20): racha prevista > 13 m/s → no despega | el X650 chocaba con ráfagas | 30 min |
| D | **Realismo (7)**: pérdida de empuje con la velocidad, caída de tensión de la batería, efecto suelo | antes de pasar al dron real | 2-3 h |
| E | FASTER (3) | baja prioridad: el alcance de los sensores no es el límite | 4-6 h |
