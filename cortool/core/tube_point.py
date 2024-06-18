from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


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
    molar_mass: float
    density: float
    vapor_fraction: float
    vapor_viscosity: float
    liquid_fraction: float
    liquid_viscosity: float


@dataclass
class TubePoint:
    name: str = 'mixture'
    phase_names: List[str] = field(default_factory=lambda: ['gas'])
    temperature: float = 300.0
    pressure: float = 101325.0
    velocity: float = 0.5
    diameter: float = 0.1
    length: float = 10
    components: List[FluidComponent] = field(default_factory=list)
    flow_mode: FlowMode = FlowMode.UNDEFINED
    roughness: float = 0.01
    angle: int = 0

    def add_component(self, name: str, molar_mass: float, density: float, vapor_fraction: float, liquid_fraction: float,
                      liquid_viscosity: float, vapor_viscosity: float):
        self.components.append(
            FluidComponent(name, molar_mass, density, vapor_fraction, liquid_fraction, liquid_viscosity,
                           vapor_viscosity))

    @property
    def number_of_fluids(self) -> int:
        return len(self.components)

    @property
    def overall_density(self) -> float:
        return sum(comp.density * (comp.vapor_fraction + comp.liquid_fraction) for comp in
                   self.components) / self.number_of_fluids if self.number_of_fluids > 0 else 0

    @property
    def overall_viscosity(self) -> float:
        # Пример расчета общей вязкости, требует более сложной логики в зависимости от условий
        return sum((comp.liquid_viscosity + comp.vapor_viscosity) / 2 for comp in
                   self.components) / self.number_of_fluids if self.number_of_fluids > 0 else 0
