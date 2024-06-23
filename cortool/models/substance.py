import math
from abc import ABC, abstractmethod


class Substance(ABC):
    name: str
    molar_mass: float  # Молярная масса вещества в г/моль

    def __init__(self, name, molar_mass: float):
        self.name = name
        self.molar_mass = molar_mass

    @abstractmethod
    def get_viscosity(self, temperature: float) -> float:
        """Метод для расчёта вязкости в зависимости от температуры"""
        pass


class Ethanol(Substance):
    def __init__(self):
        super().__init__(name="ethanol", molar_mass=46.07)

    def get_viscosity(self, temperature: float) -> float:
        A = 0.00201 * 1e-6
        B = 1614
        C = 0.00618
        D = -1.132 * 1e-5
        return A * math.exp(B / temperature + C * temperature + D * temperature ** 2)


class Nitrogen(Substance):
    def __init__(self):
        super().__init__(name="nitrogen", molar_mass=28.02)

    def get_viscosity(self, temperature: float) -> float:
        VISCOSITY_INIT = 1.7e-5
        T_INIT = 273
        S = 104.7
        return (VISCOSITY_INIT * (temperature / T_INIT) ** 1.5) * (T_INIT + S) / (temperature + S)


def create_substance_dictionary():
    substance_dict = {}
    for subclass in Substance.__subclasses__():
        instance = subclass()
        substance_dict[instance.name] = instance
    return substance_dict


# Использование словаря
substance_dict = create_substance_dictionary()
print("All available substances:", list(substance_dict.keys()))
