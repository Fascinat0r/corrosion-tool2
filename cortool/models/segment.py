from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class Phase(Enum):
    LIQUID = auto()
    GAS = auto()


class FlowMode(Enum):
    BUBBLE = auto()
    PLUG = auto()
    SLUG = auto()
    ANNULAR = auto()
    MIST = auto()
    UNDEFINED = auto()


@dataclass
class FluidComponent:
    name: str
    phase: Phase
    molar_mass: float
    density: float
    vapor_fraction: float
    vapor_viscosity: float
    liquid_fraction: float
    liquid_viscosity: float


@dataclass
class Segment:
    name: str = 'mixture'
    phase_names: List[str] = field(default_factory=lambda: ['gas'])
    temperature: float = 300.0
    pressure: float = 101325.0
    velocity: float = 0.5
    diameter: float = 0.1
    length: float = 10
    components: List[FluidComponent] = field(default_factory=list)
    roughness: float = 0.01
    angle: int = 0

    def add_component(self, name: str, phase: Phase, molar_mass: float, density: float, vapor_fraction: float,
                      liquid_fraction: float,
                      liquid_viscosity: float, vapor_viscosity: float):
        self.components.append(
            FluidComponent(name, phase, molar_mass, density, vapor_fraction, liquid_fraction, liquid_viscosity,
                           vapor_viscosity))

    @property
    def number_of_fluids(self) -> int:
        return len(self.components)

    @property
    def overall_density(self) -> float:  # TODO: необходимо дописать учёт агрегатного состояния
        return sum(comp.density * (comp.vapor_fraction + comp.liquid_fraction) for comp in
                   self.components) / self.number_of_fluids if self.number_of_fluids > 0 else 0

    @property
    def overall_viscosity(self) -> float:
        # TODO: Пример расчета общей вязкости, требует более сложной логики в зависимости от условий
        return sum((comp.liquid_viscosity + comp.vapor_viscosity) / 2 for comp in
                   self.components) / self.number_of_fluids if self.number_of_fluids > 0 else 0

    @property
    def reynolds(self) -> float:
        """
        Calculates the Reynolds number based on the total density and total viscosity of the medium.
        :return: Reynolds number
        """
        return self.velocity * self.diameter * self.overall_density / self.overall_viscosity

    @property
    def xtt(self) -> float:
        """
        Calculates the Lockhart-Martinelli parameter to determine flow mode.
        This method dynamically identifies liquid and gas components.
        """
        liquid = next((c for c in self.components if c.phase == Phase.LIQUID), None)
        gas = next((c for c in self.components if c.phase == Phase.GAS), None)

        if not liquid or not gas:
            return None  # Не найдены компоненты жидкости или газа

        # Используя свойства жидкости и газа для вычисления xtt
        xtt = ((1.096 / liquid.density) ** 0.5) * \
              ((liquid.density / gas.density) ** 0.25) * \
              ((gas.vapor_viscosity / liquid.liquid_viscosity) ** 0.1) * \
              ((self.velocity / self.diameter) ** 0.5)
        return xtt

    @property
    def flow_mode(self) -> FlowMode | None:  # TODO: предоставить формулы
        cur_xtt = self.xtt
        if cur_xtt is None:
            return None  # Unable to compute without a valid xtt
        if cur_xtt < 10:
            return FlowMode.BUBBLE
        if 10 <= cur_xtt < 100:
            return FlowMode.PLUG
        if 100 <= cur_xtt < 1000:
            return FlowMode.SLUG
        if 1000 <= cur_xtt < 10000:
            return FlowMode.ANNULAR
        if 10000 <= cur_xtt:
            return FlowMode.MIST
        return FlowMode.UNDEFINED

    @property
    def friction_factor(self) -> None | float:  # TODO: предоставить формулу
        """
        Determines the friction factor based on the Lockhart-Martinelli parameter.
        This friction factor is used to calculate the viscosity in multiphase flow.
        """
        cur_xtt = self.xtt
        if cur_xtt is None:
            return None  # Unable to compute without a valid xtt

        if cur_xtt < 10:
            return 1.0
        elif cur_xtt < 100:
            return 0.9
        elif cur_xtt < 1000:
            return 0.8
        elif cur_xtt < 10000:
            return 0.7
        else:
            return 0.6

    @property
    def r_lambda(self) -> float:  # TODO: предоставить формулу
        reynolds_number = self.reynolds
        if reynolds_number < 2300:
            return 64 / reynolds_number
        else:
            return 0.316 / (reynolds_number ** 0.25)

    @property
    def pressure_loss(self) -> float:
        xi = self.r_lambda * self.length / self.diameter
        return (xi * self.velocity ** 2) * 0.5 * self.overall_density
