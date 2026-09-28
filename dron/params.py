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
from dataclasses import dataclass, field, replace
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
    gust_max: float = 0.0       # m/s, racha pico prevista con la que aún despega (0: sin comprobación; ver gust_peak)
    tko_speed: float = 1.5      # MPC_TKO_SPEED
    land_speed: float = 0.7     # MPC_LAND_SPEED
    tau_att: float = 0.08       # s, respuesta de actitud (ESTIMADO, bucle interno rápido)
    tau_motor: float = 0.03     # s, respuesta de los motores (ESTIMADO)
    # ganancias del control en cascada (PX4 por defecto)
    gains: Dict[str, float] = field(default_factory=lambda: dict(
        xy_p=0.95, xy_vel_p=1.8, xy_vel_i=0.4, xy_vel_d=0.2, z_p=1.0, z_vel_p=4.0, z_vel_i=2.0, z_vel_d=0.0))
    # --- modelo de 6 grados de libertad (dynamics.Multirotor6DOF): cuatro motores en X ---
    arm: float = 0.246          # m, del centro a cada motor
    inertia: tuple = (0.0163, 0.0163, 0.030)   # kg·m² (Ixx, Iyy, Izz)
    k_m: float = 0.016          # m, par de reacción de cada rotor / su empuje (momentConstant de Gazebo)
    tau_up: float = 0.0125      # s, constante de tiempo del motor acelerando
    tau_down: float = 0.025     # s, frenando
    # bucle interno de PX4 (mc_att_control y mc_rate_control): actitud P (rad/s por rad) y velocidad angular PID.
    # Guiñada: 4 veces MC_YAWRATE_P con poca integral (ajuste propio: con 0,2/0,1 se pasaba 15-30° de rumbo)
    # (par normalizado por rad/s). Los de fábrica de PX4 están ajustados para el X500; en otros drones se escalan
    # por la autoridad de par (par máximo / inercia) para la misma respuesta (ESTIMADO: en uno real, autotune)
    inner: Dict[str, float] = field(default_factory=lambda: dict(
        att_p=6.5, yaw_p=2.8, rate_p=0.15, rate_i=0.2, rate_d=0.003, yawrate_p=0.8, yawrate_i=0.08,
        rate_max=220.0, yawrate_max=200.0))

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


# El dron con el que se entrena y el único que ofrece la interfaz. Los demás perfiles (mini, px4, matrice) se quedan
# para las pruebas y las comparaciones con resultados anteriores
TRAINING = ("x650",)
# COMPROBACIÓN DE VIENTO ANTES DE DESPEGAR (como COM_WIND_MAX de PX4 o un operador que mira la previsión): la racha
# pico prevista, viento medio × (1 + GUST_K × intensidad de ráfagas), unos 3σ de la turbulencia Dryden a baja altura
# (dron/wind.py: σ ≈ 0,2 × viento medio por nivel). Si pasa del `gust_max` del dron, no despega.
GUST_K = 0.6


def gust_peak(wind_speed: float, gusts: int) -> float:
    return wind_speed * (1.0 + GUST_K * gusts)


TWR_MIN = 1.2   # por debajo, no despega con seguridad (sin margen para frenar ni para las ráfagas)


def with_payload(p: "Profile", kg: float) -> "Profile":
    """El mismo dron con `kg` de carga útil colgada bajo el centro (a ~0,1 m): más masa, menos empuje por kilo
    (TWR), algo más de inercia y más potencia en estacionario (∝ peso^1,5). El arrastre por kilo se deja igual
    (prudente: con más masa, el mismo arrastre frena menos por kilo). ValueError si pasa del peso máximo."""
    if kg <= 0:
        return p
    m = p.mass + kg
    k = m / p.mass
    twr = p.twr / k
    if twr < TWR_MIN:
        raise ValueError("Con %.1f kg de carga el %s pesa %.1f kg: empuje/peso %.2f < %.1f, no despega con seguridad"
                         % (kg, p.name, m, twr, TWR_MIN))
    ixx, iyy, izz = p.inertia
    return replace(p, mass=m, twr=twr, hover_power_w=p.hover_power_w * k ** 1.5,
                   inertia=(ixx + kg * 0.01, iyy + kg * 0.01, izz + kg * 0.0025))


PROFILES = {
    # EL DRON DE ENTRENAMIENTO. Holybro X650 (kit de desarrollo de referencia de PX4, por piezas o ARF), con la
    # electrónica de docs/HARDWARE.md. Datos de Holybro (docs.holybro.com, X650 Development Kit): 650 mm entre ejes,
    # 2,0 kg sin batería, despegue máximo 6,3 kg (al 70 % de acelerador), carga útil 4,3 kg sin batería / 3,5 kg con
    # 5200 mAh (750 g) / 3,1 kg con 10000 mAh (1200 g), motores T-Motor MN4014 330KV, hélices de 15" (Gemfan 1555),
    # ESC 45 A, 6S, 30 min en estacionario con 10000 mAh sin carga.
    # Aquí: con la batería de 10000 mAh (6S, 222 Wh) y la electrónica de a bordo (ordenador, LiDAR 360°, cámara de
    # profundidad, telémetro, soportes: ~0,45 kg, ESTIMADO): 3,65 kg sin carga útil. La carga útil se añade aparte
    # (Simulation(payload=kg)): como mucho ~2,6 kg hasta los 6,3 kg de despegue máximo.
    "x650": Profile(
        key="x650", name="Holybro X650 (PX4, por piezas)", mass=3.65,
        radius=0.52,        # brazo 0,325 m + media hélice de 15" (0,19 m)
        # velocidades, aceleraciones y giro: los valores por defecto de PX4 (MPC_XY_VEL_MAX 12, MPC_ACC_HOR 3,
        # MPC_ACC_HOR_MAX 5, MPC_JERK_AUTO 4, MPC_Z_VEL_MAX_UP 3, MPC_Z_VEL_MAX_DN 1,5, MPC_TILT_MAX_AIR 45°)
        v_max=12.0, v_cruise=5.0, acc_hor=3.0, jerk=4.0, v_up=3.0, v_down=1.5, tilt_max_deg=45.0,
        yaw_rate_deg=60.0,
        # empuje: 2,6 kg por motor con hélice de 17" (T-Motor, MN4014 330KV a 6S); con la de 15", ~2,1 kg (ESTIMADO,
        # coherente con los 6,3 kg de Holybro al 70 %): 8,4 kg entre los cuatro → TWR 2,3 sin carga
        twr=8.4 / 3.65,
        battery_wh=222.0,   # 6S 10000 mAh
        # estacionario: 30 min con 222 Wh a 3,2 kg (Holybro) → 444 W; a 3,65 kg, × (3,65/3,2)^1,5 ≈ 540 W (la potencia
        # de estacionario crece con el peso^1,5, teoría del disco actuador): ~25 min
        hover_power_w=540.0,
        # viento: Holybro no lo publica. Medido en simulación (26-09-2026, carrera y aterrizaje, 30 vuelos por
        # casilla): sin ráfagas no choca ni con 12 m/s; los choques los deciden las ráfagas, y la racha pico
        # (gust_peak) separa lo seguro de lo peligroso con 13 m/s: ligeras hasta 8 m/s, 0 choques (10: 3 y 1);
        # moderadas hasta 5,9 (4: 0; 6: 1 de 60); fuertes hasta 4,6 (4: 0; 6: 5 y 5)
        wind_max=13.0, gust_max=13.0,
        sensor_range=12.0,  # Slamtec RPLIDAR C1 (docs/HARDWARE.md)
        gps_sigma=0.5,      # Holybro M10
        acc_max=5.0,        # MPC_ACC_HOR_MAX por defecto
        # 6 grados de libertad: brazo 0,325 m (650 mm entre ejes); inercias con las proporciones del X500
        # (Ixx ≈ 0,18·m·brazo², Izz ≈ 0,33·m·brazo², ESTIMADO); k_m ≈ 0,016·15/10 (hélices de 15"); motores más
        # lentos que los del X500 (hélices más grandes, ESTIMADO). Ganancias del autopiloto interno: las de PX4
        # escaladas por la autoridad de par (brazo × empuje / inercia: ~5 % menos que el X500) y de guiñada
        arm=0.325, inertia=(0.069, 0.069, 0.127), k_m=0.024, tau_up=0.020, tau_down=0.040,
        inner=dict(att_p=6.5, yaw_p=2.8, rate_p=0.16, rate_i=0.21, rate_d=0.0032, yawrate_p=0.75, yawrate_i=0.075,
                   rate_max=220.0, yawrate_max=200.0)),

    "mini": Profile(
        key="mini", name="Pequeño (tipo DJI Mini 3, 249 g)", mass=0.249, radius=0.18,
        v_max=16.0, v_cruise=8.0, acc_hor=4.0, jerk=6.0, v_up=5.0, v_down=3.5, tilt_max_deg=35.0,
        yaw_rate_deg=120.0, twr=2.2, battery_wh=18.1, hover_power_w=18.1 / (38 / 60), wind_max=10.7,
        sensor_range=12.0, gps_sigma=1.5 / 2,
        acc_max=4.8,    # ESTIMADO: 70 % de g·tan(35°) (DJI no publica la aceleración; el resto se va en el arrastre)
        # ESTIMADO: brazo de ~0,10 m (plegado mide 148 × 90 mm; desplegado, ~0,2 m entre motores en diagonal) e
        # inercias con las mismas proporciones que el X500 (I ≈ 0,18·m·brazo² e Izz ≈ 0,33·m·brazo²); motores
        # pequeños y rápidos
        # el par de reacción de una hélice por unidad de empuje crece con su diámetro: ~6" → k_m ≈ 0,016·6/10
        arm=0.10, inertia=(0.00045, 0.00045, 0.00082), tau_up=0.010, tau_down=0.020, k_m=0.0096,
        inner=dict(att_p=6.5, yaw_p=2.8, rate_p=0.056, rate_i=0.075, rate_d=0.0011, yawrate_p=0.2, yawrate_i=0.02,
                   rate_max=220.0, yawrate_max=200.0)),
    "px4": Profile(
        key="px4", name="Genérico PX4 (valores por defecto)", mass=1.5, radius=0.3,
        v_max=12.0, v_cruise=5.0, acc_hor=3.0, jerk=4.0, v_up=3.0, v_down=1.5, tilt_max_deg=45.0,
        yaw_rate_deg=60.0, twr=2.0, battery_wh=80.0, hover_power_w=200.0, wind_max=10.0,
        sensor_range=15.0, gps_sigma=0.5,
        acc_max=5.0),   # MPC_ACC_HOR_MAX por defecto (la de los modos manuales; la de misión es MPC_ACC_HOR, 3)
        # 6 grados de libertad: los valores por defecto de la clase, del X500 de PX4 (PX4-gazebo-models, x500 y
        # x500_base: motores a ±0,174 m → brazo 0,246 m; inercias 0,0217/0,0217/0,040 para 2,0 kg, aquí escaladas a
        # 1,5 kg; momentConstant 0,016; timeConstantUp/Down 0,0125/0,025 s) y las ganancias de fábrica de PX4
    "matrice": Profile(
        key="matrice", name="Inspección (tipo DJI Matrice 350 RTK)", mass=6.47, radius=0.45,
        v_max=23.0, v_cruise=10.0, acc_hor=4.0, jerk=5.0, v_up=6.0, v_down=5.0, tilt_max_deg=30.0,
        yaw_rate_deg=100.0, twr=1.9, battery_wh=2 * 263.2, hover_power_w=2 * 263.2 / (55 / 60), wind_max=12.0,
        sensor_range=40.0, gps_sigma=0.1,  # RTK: ±0,1 m
        acc_max=4.0,    # ESTIMADO: 70 % de g·tan(30°) ≈ la que ya usa (6,5 kg y 30° de inclinación máxima)
        # distancia diagonal entre motores de 895 mm (especificación de DJI) → brazo 0,4475 m; inercias ESTIMADAS
        # con las proporciones del X500; hélices de 21" y motores más lentos (ESTIMADO)
        # k_m ≈ 0,016·21/10 ≈ 0,03 (hélices de 21"; ESTIMADO)
        arm=0.4475, inertia=(0.233, 0.233, 0.43), tau_up=0.030, tau_down=0.060, k_m=0.03,
        inner=dict(att_p=6.5, yaw_p=2.8, rate_p=0.29, rate_i=0.39, rate_d=0.0058, yawrate_p=1.5, yawrate_i=0.15,
                   rate_max=220.0, yawrate_max=200.0)),
}
