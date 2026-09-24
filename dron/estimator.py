"""Estimación de estado con un filtro de Kalman, simplificación del EKF2 de PX4.

Estado (9): posición, velocidad y sesgo del acelerómetro, en ejes del mundo.
  * Predicción a 200 Hz con la aceleración medida por la IMU (menos el sesgo estimado).
  * Correcciones con GPS (posición y velocidad), barómetro (altura) y el telémetro inferior (distancia a la superficie
    de debajo + mapa del terreno = altura; muy preciso cerca del suelo: es lo que permite aterrizar suave).
  * Cada medida pasa una "puerta" de innovación (test χ² de 5 sigmas, como EKF2_*_GATE): si no cuadra con lo
    esperado (p. ej. el telémetro inferior mide la azotea de un edificio y no el suelo), se descarta.
  * Si el GPS se rechaza de forma sostenida (0,5 s), el que se ha perdido es el FILTRO: se reinicia posición y
    velocidad con el GPS, como hace EKF2 cuando falla la comprobación de innovación demasiado tiempo.

Con la actitud conocida, el EKF de PX4 se reduce a este filtro lineal para la parte de traslación.
Ruidos de proceso: EKF2_ACC_NOISE = 0,35 m/s², EKF2_ACC_B_NOISE = 0,003 m/s³ (valores por defecto de PX4).
"""
import numpy as np

ACC_NOISE, ACC_B_NOISE = 0.35, 0.003
GATE = 25.0  # 5 sigmas al cuadrado


class Estimator:
    def __init__(self, p0, gps_sigma: float, noise_level: float):
        self.x = np.zeros(9)
        self.x[:3] = p0
        self.P = np.diag([1.0] * 3 + [0.1] * 3 + [0.01] * 3)
        m = max(noise_level, 0.05)  # sin ruido, el filtro confía casi del todo en las medidas
        self.r_gps = (max(gps_sigma, 0.1) * m) ** 2
        self.r_gps_z = (1.5 * max(gps_sigma, 0.1) * m) ** 2
        self.r_gps_v = (0.1 * m) ** 2 + 1e-4
        self.r_baro = (0.3 * m) ** 2 + 1e-4
        self.r_rng = (0.05 * m) ** 2 + 1e-4
        self.rejected = 0
        self.gps_fails = 0
        self.resets = 0

    @property
    def p(self):
        return self.x[:3]

    @property
    def v(self):
        return self.x[3:6]

    def predict(self, acc_meas: np.ndarray, dt: float):
        a = acc_meas - self.x[6:9]
        x = self.x
        x[:3] += x[3:6] * dt + 0.5 * a * dt * dt
        x[3:6] += a * dt
        I = np.eye(3)
        F = np.eye(9)
        F[0:3, 3:6] = I * dt
        F[0:3, 6:9] = -0.5 * I * dt * dt
        F[3:6, 6:9] = -I * dt
        Gu = np.zeros((9, 3))
        Gu[0:3] = 0.5 * I * dt * dt
        Gu[3:6] = I * dt
        Q = Gu @ Gu.T * ACC_NOISE ** 2
        Q[6:9, 6:9] += I * ACC_B_NOISE ** 2 * dt
        self.P = F @ self.P @ F.T + Q

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> bool:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        Si = np.linalg.inv(S)
        if float(y @ Si @ y) > GATE * len(z):
            self.rejected += 1
            return False
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        return True

    def gps(self, pos: np.ndarray, vel: np.ndarray):
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        ok = self._update(H, pos, np.diag([self.r_gps, self.r_gps, self.r_gps_z]))
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)
        ok = self._update(H, vel, np.eye(3) * self.r_gps_v) and ok
        self.gps_fails = 0 if ok else self.gps_fails + 1
        if self.gps_fails >= 5:  # 0,5 s rechazando el GPS: el filtro se ha perdido -> reinicio con el GPS (como EKF2)
            self.x[:3], self.x[3:6] = pos, vel
            self.P[:6, :6] = np.diag([self.r_gps] * 3 + [self.r_gps_v] * 3) * 4
            self.P[:6, 6:] = self.P[6:, :6] = 0.0
            self.gps_fails = 0
            self.resets += 1

    def height(self, z: float, r: float):
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        return self._update(H, np.array([z]), np.array([[r]]))

    def baro(self, z: float):
        self.height(z, self.r_baro)

    def range_down(self, dist: float, surface: float):
        """El telémetro inferior mide la distancia a la superficie de debajo. Con el mapa del terreno (conocido en
        la fase 1) eso da la altura absoluta: z = superficie + distancia. Si el mapa no cuadra (p. ej. debajo hay un
        árbol que no es superficie), la puerta de innovación descarta la medida."""
        if dist < 8.0:
            self.height(surface + dist, self.r_rng)
