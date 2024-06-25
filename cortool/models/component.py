from enum import Enum, auto

from cortool.models.substance import Substance, create_substance


class Phase(Enum):
    LIQUID = auto()
    GAS = auto()


class Component:
    substance: Substance  # Химический элемент
    temperature: float  # Температура
    pressure: float  # Давление
    fraction: float  # Доля компонента в потоке
    phase: Phase  # Агрегатное состояние компонента
    velocity: float  # Скорость компонента в потоке

    def __init__(self, substance_name: str, temperature: float, pressure: float, velocity: float, fraction: float,
                 phase: Phase):
        self.substance = create_substance(substance_name)
        self.temperature = temperature
        self.pressure = pressure
        self.fraction = fraction
        self.phase = phase
        self.velocity = velocity

    @property
    def density(self) -> float:
        return self.substance.get_density(self.temperature, self.pressure)

    @property
    def viscosity(self) -> float:
        return self.substance.get_viscosity(self.temperature)
