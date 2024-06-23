from dataclasses import dataclass
from enum import Enum, auto

from cortool.models.substance import Substance, Ethanol, Nitrogen


class Phase(Enum):
    LIQUID = auto()
    GAS = auto()


@dataclass
class Component:
    name: str
    substance: Substance
    temperature: float
    fraction: float  # Доля компонента в потоке
    phase: Phase
    density: float = None  # Плотность, будет рассчитываться динамически
    viscosity: float = None  # Вязкость, будет рассчитываться динамисески

    def __post_init__(self):
        self.update_properties()

    def update_properties(self):
        """Обновление свойств компонента на основе температуры."""
        self.density = self.calculate_density(self.temperature)
        self.viscosity = self.substance.get_viscosity(self.temperature)

    def calculate_density(self, temperature: float) -> float:
        """Метод для расчета плотности, может включать зависимость от температуры."""
        # Примерный расчет, нужна конкретная формула или метод
        return self.substance.molar_mass / (0.0821 * temperature)

    def set_temperature(self, temperature: float):
        self.temperature = temperature
        self.update_properties()


# Использование:
ethanol_component = Component("Ethanol", Ethanol(), 46.07, 0.7, Phase.LIQUID)
nitrogen_component = Component("Nitrogen", Nitrogen(), 28.02, 0.3, Phase.GAS)
ethanol_component.set_temperature(300)
nitrogen_component.set_temperature(300)

print(f"Density and viscosity of Ethanol at 300K: {ethanol_component.density}, {ethanol_component.viscosity}")
print(f"Density and viscosity of Nitrogen at 300K: {nitrogen_component.density}, {nitrogen_component.viscosity}")
