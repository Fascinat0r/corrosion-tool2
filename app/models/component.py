import numpy as np

from app.models.substance import Substance, create_substance


class Component:
    """
    Класс, представляющий компонент потока.
    Описывает физические свойства компонента, такие как плотность, вязкость и т.д.
    """
    substance: Substance  # Химический элемент
    temperature: np.float64  # Температура
    pressure: np.float64  # Давление
    composition: np.float64  # Доля компонента в потоке  в мольных долях
    velocity: np.float64  # Скорость компонента в потоке
    fractions: (np.float64, np.float64)  # Доля жидкой и газовой фазы в компоненте (жидкая, газовая)
    densities: (np.float64, np.float64)  # Плотности жидкой и газовой фазы в компоненте (жидкая, газовая)

    def __init__(self, substance_name: str, temperature: np.float64, pressure: np.float64, velocity: np.float64,
                 composition: np.float64):
        self.substance = create_substance(substance_name)
        self.temperature = temperature
        self.pressure = pressure
        self.composition = composition
        self.velocity = velocity
        self.fractions = (None, None)
        self.densities = (None, None)

    def copy(self):
        """
        Создает копию компонента.
        """
        comp = Component(self.substance.name, self.temperature, self.pressure, self.velocity, self.composition)
        comp.fractions = self.fractions
        comp.densities = self.densities
        return comp

    @property
    def density(self) -> np.float64:
        """
        Вычисляет плотность компонента потока на основе двух фаз (жидкой и газовой).
        """
        return sum(density * fraction for density, fraction in zip(self.densities, self.fractions))

    @property
    def viscosity(self) -> np.float64:
        """
        Вычисляет вязкость компонента потока.
        """
        return self.substance.get_viscosity(self.temperature)
