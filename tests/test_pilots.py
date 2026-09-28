"""Los algoritmos de pilotaje elegibles (dron/pilots.py): cada uno completa una misión sencilla con sus parámetros."""
import pytest

from dron import pilots, tuning
from dron.sim import Simulation


@pytest.mark.parametrize("pilot", list(pilots.PILOTS))
def test_each_pilot_lands(pilot):
    s = Simulation(profile="px4", level="mixto", seed=5, pilot=pilot)
    while not s.done:
        s.step(1.0)
    assert s.status == "success", (pilot, s.status, s.cause)
    assert s.config["pilot"] == pilot


def test_factory_pilot_uses_factory_params():
    s = Simulation(profile="mini", seed=1, pilot="fabrica")
    factory = tuning.params("mini", tuned=False)
    assert {k: s.mission.P[k] for k in factory} == factory
    assert all(s.mission.P[k] == v for k, v in pilots.RACE_DEFAULTS.items())   # aterrizando: sin nada entrenado
    assert s.ctrl.P is s.mission.P                     # Collision Prevention con los mismos parámetros
    assert not s.mission.sprint_on and not s.mission.confirm_in_flight


def test_unknown_pilot_is_rejected():
    with pytest.raises(ValueError):
        Simulation(seed=1, pilot="no-existe")


def test_ram_uses_the_thrust_of_the_four_motors_beyond_factory_limits():
    """Embestida (carrera, meta a la vista y pasillo libre): se inclina más que el límite de fábrica, sin pasar de
    lo que aún sostiene el peso, y va más rápido que la velocidad máxima de fábrica, hasta chocar con la meta."""
    import math
    from dron.sim import Simulation
    s = Simulation(profile="mini", mode="carrera", level="mixto", seed=2)
    tilt = speed = 0.0
    while not s.done:
        s.step(0.05)
        if s.mission.ram:
            tilt = max(tilt, math.degrees(s.drone.tilt))
            speed = max(speed, math.hypot(s.drone.v[0], s.drone.v[1]))
    p = s.prof
    assert s.status == "success"
    assert p.tilt_max_deg + 10 < tilt < math.degrees(math.acos(1.0 / p.twr)) + 3   # ~60°: más que 35°, sin caer
    assert speed > 1.15 * p.v_max                                                   # 16 → ~20 m/s
