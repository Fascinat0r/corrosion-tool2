import math
from abc import ABC, abstractmethod


class Substance(ABC):
    @abstractmethod
    def viscosity_from_temperature(self, temperature: float) -> float:
        """Метод для расчёта вязкости в зависимости от температуры"""
        pass


class Ethanol(Substance):
    def viscosity_from_temperature(self, temperature: float) -> float:
        A = 0.00201 * 1e-6
        B = 1614
        C = 0.00618
        D = -1.132 * 1e-5
        return A * math.exp(B / temperature + C * temperature + D * temperature ** 2)


class Nitrogen(Substance):
    def viscosity_from_temperature(self, temperature: float) -> float:
        VISCOSITY_INIT = 1.7e-5
        T_INIT = 273
        S = 104.7
        return (VISCOSITY_INIT * (temperature / T_INIT) ** 1.5) * (T_INIT + S) / (temperature + S)


# Для использования:
ethanol = Ethanol()
nitrogen = Nitrogen()
print("Viscosity of Ethanol at 300K:", ethanol.viscosity_from_temperature(300))
print("Viscosity of Nitrogen at 300K:", nitrogen.viscosity_from_temperature(300))
