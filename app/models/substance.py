import math
from abc import ABC, abstractmethod

from app.config.constants import UNIVERSAL_GAS_CONSTANT
from repository.density_repo import density_module


class Substance(ABC):
    """
    Абстрактный класс, описывающий химическое вещество.
    Описывает химические свойства вещества, такие как молярная масса и удельная теплоёмкость и т.д.
    """

    def __init__(self, thermopack_id, name, molar_mass, specific_heat_capacity):
        self.thermopack_id = thermopack_id  # Идентификатор флюида в базе Thermopack
        self.name = name  # Название вещества
        self.molar_mass = molar_mass  # Молярная масса вещества в кг/моль
        self.specific_heat_capacity = specific_heat_capacity  # удельная теплоёмкость Дж/(кг·К)

    @abstractmethod
    def get_viscosity(self, temperature: float) -> float:
        """Метод для расчёта вязкости в зависимости от температуры"""
        pass

    @abstractmethod
    def get_density(self, temperature: float, pressure: float) -> float:
        """Метод для расчёта плотности в граммах на метр кубический"""
        pass

    def __str__(self):
        return f"{self.name} ({self.thermopack_id})"


class Ethanol(Substance):
    def __init__(self):
        super().__init__("ETOH", "ethanol", 0.04607, 2470)

    def get_viscosity(self, temperature: float) -> float:
        A = 0.00201 * 1e-6
        B = 1614
        C = 0.00618
        D = -1.132 * 1e-5
        return A * math.exp(B / temperature + C * temperature + D * temperature ** 2)

    def get_density(self, temperature: float, pressure: float) -> float:
        return density_module.get_density(self.name, temperature)


class Nitrogen(Substance):
    def __init__(self):
        super().__init__("N2", "nitrogen", 0.02802, 1040)

    def get_viscosity(self, temperature: float) -> float:
        VISCOSITY_INIT = 1.7e-5
        T_INIT = 273
        S = 104.7
        return (VISCOSITY_INIT * (temperature / T_INIT) ** 1.5) * (T_INIT + S) / (temperature + S)

    def get_density(self, temperature: float, pressure: float) -> float:
        """Вычислние плотности азота, используя уравнение состояния идеального газа"""
        return pressure * self.molar_mass / (UNIVERSAL_GAS_CONSTANT * temperature)


def create_substance_dictionary():
    """
    Функция создания словаря, содержащего все классы веществ, унаследованные от Substance.
    Ключи словаря - названия веществ, значения - объекты классов веществ.
    """
    substance_dict = {}
    for subclass in Substance.__subclasses__():
        instance = subclass()
        substance_dict[instance.name] = instance
    return substance_dict


def create_substance(name: str) -> Substance:
    """ Функция создания объекта Substance на основе строки. """
    if name in substance_dict:
        return substance_dict[name]
    else:
        raise ValueError(f"No substance class defined for {name}")


# Использование словаря
substance_dict = create_substance_dictionary()
print("All available substances:", list(substance_dict.keys()))
