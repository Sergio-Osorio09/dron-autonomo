# Construir un dron real con esta autonomía (piezas baratas)

Lista pensada para reproducir en un dron real lo que hace el simulador: mapa propio (fase 2), búsqueda con cámara
(fase 3), persecución (fase 4) y enjambre (fase 5). Arquitectura como la de los drones reales (y la del simulador):

```
controladora de vuelo (PX4: EKF2, control en cascada, Collision Prevention, geovalla, aterrizaje)
      ↑ MAVLink (offboard: referencias p/v/a)          ↓ estado estimado
ordenador de a bordo (mapa de ocupación + ESDF, A*, trayectoria, búsqueda bayesiana, detector)
      ↑ cámara de profundidad + detección, LiDAR 360°, telémetro inferior
```

Precios consultados el 25-09-2026 (EE. UU., sin impuestos ni envío); la memoria ha encarecido mucho los ordenadores
de a bordo en 2026.

## Atributos que necesita (sacados de lo que pide el simulador)

| Atributo | Valor mínimo | Por qué |
|---|---|---|
| Relación empuje/peso | ≥ 2 | margen para frenar y aguantar ráfagas (los perfiles usan 1,9-2,2) |
| Autonomía con carga | ≥ 15 min | búsquedas de 30-60 s por zona de 40 × 30 m, varias zonas y reintentos |
| Alcance de los sensores de obstáculos | ≥ 10-12 m | limita la velocidad: v ≤ √(2·a·(alcance − radio − 1 m)) ≈ 6-9 m/s |
| Cobertura de obstáculos | 360° horizontal + frente/abajo | anillo de telémetros + cámara frontal + visión inferior (lección 3d.8) |
| Ordenador de a bordo | ≥ 4 núcleos ARM A76 o GPU pequeña | mapa + ESDF (~15 ms) + A* (~15 ms) a 5 Hz, y detector a ≥ 2 Hz |
| GNSS | ±1-2 m (M10) | el filtro baja el error a ~1 m; RTK (±0,1 m) solo si hace falta precisión de centímetros |
| Enlace de datos entre drones | ~10 Hz, < 100 m | el enjambre comparte posición, punto elegido y avistamientos |

## EL DRON DE ENTRENAMIENTO: Holybro X650 (26-09-2026)

Se entrena un solo dron, práctico, que se arma por piezas y lleva carga: el **Holybro X650** (perfil `x650`), el kit
de desarrollo de referencia de PX4. Datos de Holybro: 650 mm entre ejes, 2,0 kg sin batería, despegue máximo 6,3 kg,
motores T-Motor MN4014 330KV, hélices de 15", ESC 45 A, 6S, 30 min en estacionario (10000 mAh, sin carga).

**Carga útil real:** Holybro anuncia 4,3 kg sin batería, pero con la batería de 10000 mAh (1,2 kg) y la electrónica
de autonomía de abajo (~0,45 kg) el dron pesa ~3,65 kg: caben **~2,6 kg** hasta los 6,3 kg (≈3 kg con la batería de
5200 mAh). Para volar con soltura (empuje/peso ≈ 2), 1-1,5 kg. 4 kg con margen piden un chasis mayor (p. ej. Tarot X4
de 960 mm, hasta 8 kg al despegue). En el simulador: `Simulation(payload=kg)` y el selector "Carga" de la interfaz.

| Pieza | Modelo | Precio aprox. | En el simulador |
|---|---|---|---|
| Chasis + motores + ESC + hélices | Holybro X650 ARF (fibra de carbono, MN4014 330KV, Tekko32 45 A, Gemfan 1555, placa de distribución) · o solo el chasis (249 $) y el resto por piezas | 899 $ | física (`params.py`, perfil `x650`) |
| Batería | LiPo 6S 10000 mAh (XT60) | ~150-200 $ (ESTIMADO) | batería (`battery_wh` 222 Wh) |
| Controladora de vuelo | Holybro Pixhawk 6C Mini (PX4) | 131 $ | EKF2, control PX4, Collision Prevention, geovalla |
| GNSS + brújula | Holybro M10 GPS | ~45 $ | GPS con ruido y deriva |
| Telémetro inferior | Benewake TF-Luna (0,2-8 m) | 20-25 $ | rayo inferior (altura sobre el suelo, aterrizaje) |
| LiDAR 360° | Slamtec RPLIDAR C1 (12 m, 5000 muestras/s, 10 Hz) | ~80 $ | el anillo de 36 telémetros (PX4 lo usa como OBSTACLE_DISTANCE) |
| Cámara de profundidad + detección | Luxonis OAK-D Lite (estéreo + color 4K + red neuronal a bordo) | 269 $ | cámara de profundidad frontal y cámara de detección (el detector corre en la propia cámara) |
| Ordenador de a bordo | Raspberry Pi 5 4 GB | ~65-110 $ | mapa, ESDF, planificación, búsqueda, misión |
| Radio de control + receptor | ExpressLRS (p. ej. emisora de gama baja + receptor ELRS) | ~80-100 $ | seguridad: piloto humano que puede tomar el control |
| Radio de telemetría | SiK 433/915 MHz o la propia ELRS con MAVLink | ~30 $ | enlace con la estación de tierra |
| Varios | regulador 5 V/5 A para la Pi, cables, soportes impresos en 3D | ~30 $ | — |
| **Total** | | **~1900-2000 $** | |

(La lista anterior, de ~1000-1100 $, usaba el Holybro X500 V2 ARF, 259 $, con batería 4S de 5000 mAh: el perfil `px4`
del simulador. Más barato, pero con ~0,5-1 kg de carga útil.)

Para la visión inferior (fase 2) y el aterrizaje de precisión se puede añadir una cámara barata mirando hacia abajo
con marcadores AprilTag en la plataforma (PX4 acepta LANDING_TARGET desde el ordenador de a bordo) en lugar de un
IR-LOCK. Para el enjambre, multiplicar por el número de drones (la estación de tierra y la emisora pueden ser
compartidas; cada dron necesita su enlace de datos).

### Alternativas

- **Más potencia de cálculo** (detector más grande, mapa más fino, 0,5 m): NVIDIA Jetson Orin Nano Super en lugar de
  la Raspberry Pi: 399 $ en la tienda de NVIDIA a mediados de 2026 (salió a 249 $).
- **Más barato**: sin LiPo nueva ni emisora (si ya se tienen), o con una controladora compatible con PX4 más
  sencilla. No recomiendo quitar el LiDAR 360°: es lo que hace Collision Prevention, la red de seguridad de todo.
- **Por debajo de 250 g** (menos trámites): no hay hueco para el ordenador y la cámara de profundidad con esta
  autonomía; el perfil `mini` del simulador corresponde a un DJI Mini 3, que es cerrado.

### Antes de volar

- Probar primero en **PX4 SITL + Gazebo** (el puente del simulador a PX4 queda pendiente: ver `CONTEXTO.md`).
- Registrar el dron y cumplir la normativa local (peso, altura máxima, línea de vista, zonas prohibidas).
- Geovalla y modo de retorno a casa (RTL) configurados en PX4, y la emisora siempre a mano.

## Fuentes de los precios

- [Holybro X650: especificaciones oficiales](https://docs.holybro.com/drone-development-kit/x650-development-kit/overview)
- [Holybro X650 Kits (precios)](https://holybro.com/products/x650-kits)
- [T-Motor MN4014 330KV](https://store.tmotor.com/product/mn4014-kv330-motor-navigator-type.html)
- [Holybro X500 V2 Kits](https://holybro.com/products/x500-v2-kits)
- [Holybro Pixhawk 6C Mini](https://holybro.com/products/pixhawk-6c-mini)
- [Holybro M10 GPS](https://holybro.com/products/m10-gps)
- [Benewake TF-Luna (IR-LOCK)](https://irlock.com/products/benewake-tf-luna-rangefinder-8m)
- [Slamtec RPLIDAR C1 (RobotShop)](https://www.robotshop.com/products/slamtec-rplidar-c1-360-dtof-laser-scanner)
- [Luxonis OAK-D Lite](https://shop.luxonis.com/products/oak-d-lite-1)
- [Raspberry Pi 5 y subidas de precio por la memoria](https://www.raspberrypi.com/news/more-memory-driven-price-rises/)
- [Subida de precio de los Jetson (CNX Software, 22-07-2026)](https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/)
