from cortool.models.substance import Substance, create_substance


class Component:
    """
    Класс, представляющий компонент потока.
    Описывает физические свойства компонента, такие как плотность, вязкость и т.д.
    """
    substance: Substance  # Химический элемент
    temperature: float  # Температура
    pressure: float  # Давление
    composition: float  # Доля компонента в потоке  в мольных долях
    velocity: float  # Скорость компонента в потоке
    fractions: (float, float)  # Доля жидкой и газовой фазы в компоненте (жидкая, газовая)
    densities: (float, float)  # Плотности жидкой и газовой фазы в компоненте (жидкая, газовая)

    def __init__(self, substance_name: str, temperature: float, pressure: float, velocity: float, composition: float):
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
    def density(self) -> float:
        """
        Вычисляет плотность компонента потока на основе двух фаз (жидкой и газовой).
        """
        return sum(density * fraction for density, fraction in zip(self.densities, self.fractions))

    @property
    def viscosity(self) -> float:
        """
        Вычисляет вязкость компонента потока.
        """
        return self.substance.get_viscosity(self.temperature)
