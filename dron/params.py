"""Perfiles de dron con datos REALES y ganancias del autopiloto PX4.

Fuentes (consultadas el 24-09-2026):
  * PX4-Autopilot, valores por defecto de src/modules/mc_pos_control/*.yaml y src/modules/ekf2/params_*.yaml
    (MPC_XY_VEL_MAX 12 m/s, MPC_XY_CRUISE 5 m/s, MPC_ACC_HOR 3 m/s², MPC_JERK_AUTO 4 m/s³,
     MPC_Z_V_AUTO_UP 3 m/s, MPC_Z_V_AUTO_DN 1,5 m/s, MPC_TILTMAX_AIR 45°, MPC_TKO_SPEED 1,5 m/s,
     MPC_LAND_SPEED 0,7 m/s, MPC_YAWRAUTO_MAX 60°/s, ganancias MPC_XY_P 0,95, MPC_XY_VEL_P/I/D_ACC 1,8/0,4/0,2,
     MPC_Z_P 1,0, MPC_Z_VEL_P/I_ACC 4,0/2,0, EKF2_GPS_P_NOISE 0,5 m, EKF2_GPS_V_NOISE 0,3 m/s).
  * DJI Mini 3, especificaciones oficiales: 248 g, 16 m/s (Sport), subida 5 m/s, bajada 3,5 m/s, viento 10,7 m/s,
    18,1 Wh / 38 min, precisión de posición GNSS ±1,5 m horizontal y ±0,5 m vertical.
  * DJI Matrice 350 RTK, especificaciones oficiales: 23 m/s, subida 6 m/s, bajada 5 m/s, viento 12 m/s,
    inclinación máx. 30°, giro 100°/s, 2 × 263,2 Wh / 55 min, 3,77 kg sin baterías.

Lo que el fabricante no publica se marca como ESTIMADO (relación empuje/peso, tamaño, constantes de tiempo).
El arrastre se deduce de la velocidad máxima: a velocidad máxima y máxima inclinación, empuje horizontal = arrastre,
g·tan(inclinación) = k·v², así que k = g·tan(inclinación) / v_max².
"""
import math
from dataclasses import dataclass, field
from typing import Dict

G = 9.81


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    mass: float                 # kg
    radius: float               # m, esfera que envuelve el dron (colisiones)
    v_max: float                # m/s, velocidad horizontal máxima
    v_cruise: float             # m/s, crucero en vuelo autónomo
    acc_hor: float              # m/s², aceleración horizontal que usa el planificador
    jerk: float                 # m/s³, tirón del planificador
    v_up: float                 # m/s, subida máxima
    v_down: float               # m/s, bajada máxima
    tilt_max_deg: float         # inclinación máxima
    yaw_rate_deg: float         # °/s de giro en vuelo autónomo
    twr: float                  # relación empuje/peso (ESTIMADO)
    battery_wh: float
    hover_power_w: float        # potencia en vuelo estacionario = batería / autonomía
    wind_max: float             # m/s de viento que soporta según el fabricante
    sensor_range: float         # m, alcance de los telémetros
    gps_sigma: float            # m, ruido horizontal del GPS (1 sigma)
    acc_max: float = 0.0        # m/s², aceleración del sprint de carrera (0: la de acc_hor)
    tko_speed: float = 1.5      # MPC_TKO_SPEED
    land_speed: float = 0.7     # MPC_LAND_SPEED
    tau_att: float = 0.08       # s, respuesta de actitud (ESTIMADO, bucle interno rápido)
    tau_motor: float = 0.03     # s, respuesta de los motores (ESTIMADO)
    # ganancias del control en cascada (PX4 por defecto)
    gains: Dict[str, float] = field(default_factory=lambda: dict(
        xy_p=0.95, xy_vel_p=1.8, xy_vel_i=0.4, xy_vel_d=0.2, z_p=1.0, z_vel_p=4.0, z_vel_i=2.0, z_vel_d=0.0))

    @property
    def tilt_max(self) -> float:
        return math.radians(self.tilt_max_deg)

    @property
    def drag(self) -> float:
        """Coeficiente de arrastre cuadrático por unidad de masa (1/m), deducido de v_max."""
        return G * math.tan(self.tilt_max) / self.v_max ** 2

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "gains"}
        d["drag"] = self.drag
        return d


PROFILES = {
    "mini": Profile(
        key="mini", name="Pequeño (tipo DJI Mini 3, 249 g)", mass=0.249, radius=0.18,
        v_max=16.0, v_cruise=8.0, acc_hor=4.0, jerk=6.0, v_up=5.0, v_down=3.5, tilt_max_deg=35.0,
        yaw_rate_deg=120.0, twr=2.2, battery_wh=18.1, hover_power_w=18.1 / (38 / 60), wind_max=10.7,
        sensor_range=12.0, gps_sigma=1.5 / 2,
        acc_max=4.8),   # ESTIMADO: 70 % de g·tan(35°) (DJI no publica la aceleración; el resto se va en el arrastre)
    "px4": Profile(
        key="px4", name="Genérico PX4 (valores por defecto)", mass=1.5, radius=0.3,
        v_max=12.0, v_cruise=5.0, acc_hor=3.0, jerk=4.0, v_up=3.0, v_down=1.5, tilt_max_deg=45.0,
        yaw_rate_deg=60.0, twr=2.0, battery_wh=80.0, hover_power_w=200.0, wind_max=10.0,
        sensor_range=15.0, gps_sigma=0.5,
        acc_max=5.0),   # MPC_ACC_HOR_MAX por defecto (la de los modos manuales; la de misión es MPC_ACC_HOR, 3)
    "matrice": Profile(
        key="matrice", name="Inspección (tipo DJI Matrice 350 RTK)", mass=6.47, radius=0.45,
        v_max=23.0, v_cruise=10.0, acc_hor=4.0, jerk=5.0, v_up=6.0, v_down=5.0, tilt_max_deg=30.0,
        yaw_rate_deg=100.0, twr=1.9, battery_wh=2 * 263.2, hover_power_w=2 * 263.2 / (55 / 60), wind_max=12.0,
        sensor_range=40.0, gps_sigma=0.1,  # RTK: ±0,1 m
        acc_max=4.0),   # ESTIMADO: 70 % de g·tan(30°) ≈ la que ya usa (6,5 kg y 30° de inclinación máxima)
}
