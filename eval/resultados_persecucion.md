# Persecución de un vehículo que huye: 1, 2 y 3 drones (fases 4 y 5)

Banco de pruebas (`eval/bench.py`), 30 semillas (1000-1029) por fila, PX4 genérico, nivel mixto, colinas, viento
4 m/s con ráfagas ligeras, mapa conocido. El vehículo huye a 5 m/s del dron más cercano y se esconde tras los
obstáculos; los drones solo lo ven con la cámara (sin rastreador GNSS).

| drones | capturas | IC 95 % (Wilson) | tiempo medio | mediana | p90 | fallos |
|---|---|---|---|---|---|---|
| 1 | 25/30 = 83 % | 66-93 % | 116.3 s | 103.3 s | 237.8 s | 5 tiempo agotado (300 s) |
| 2 | 29/30 = 97 % | 83-99 % | 74.4 s | 49.1 s | 160.5 s | 1 tiempo agotado |
| 3 | 29/30 = 97 % | 83-99 % | 68.2 s | 60.0 s | 117.7 s | 1 tiempo agotado |

Con varios drones: avistamiento compartido (si uno lo ve, todos actualizan su filtro), cerco (los demás se abren
6 m a los lados del punto de intercepción) y, en la persecución final, solo el primero baja a la puerta (los demás
cierran 2 m más arriba por puesto). Antes de esa separación vertical, con 3 drones hubo 4 choques entre drones en
30 vuelos.

    python eval/bench.py --n 30 --drones 2 profile=px4 mode=carrera motion=huye terrain=colinas wind_speed=4 gusts=1 level=mixto
