from dataclasses import dataclass
from enum import Enum, auto
from typing import List

from cortool.models.component import Component, Phase


class FlowMode(Enum):
    BUBBLE = auto()
    PLUG = auto()
    SLUG = auto()
    ANNULAR = auto()
    MIST = auto()
    UNDEFINED = auto()


@dataclass
class PipeProperties:
    diameter: float
    length: float
    roughness: float
    angle: float  # TODO: заменить на три составляющие угла


class Segment:
    """
    Класс, представляющий один сегмент от цельной трубы.
    Используется для моделирования потока внутри трубы.
    """
    prop: PipeProperties  # Свойства трубы, сегментом которого является объект
    length: float  # Длина сегмента
    components: List[Component] = None  # Список компонентов в потоке

    def __init__(self, prop: PipeProperties, length: float, components: List[Component] = None):
        self.prop = prop
        self.length = length
        if components is None:
            self.components = []
        else:
            self.components = components

    @property
    def number_of_fluids(self) -> int:
        return len(self.components)

    @property
    def overall_density(self) -> float:  # TODO: необходимо дописать учёт агрегатного состояния
        return sum(comp.density * comp.fraction for comp in self.components) \
               / self.number_of_fluids if self.number_of_fluids > 0 else 0

    @property
    def overall_viscosity(self) -> float:
        # TODO: Пример расчета общей вязкости, требует более сложной логики в зависимости от условий
        return sum(comp.viscosity * comp.fraction for comp in self.components) \
               / self.number_of_fluids if self.number_of_fluids > 0 else 0

    @property
    def velocity(self) -> float:
        # TODO: Пример расчета общей вязкости, требует более сложной логики в зависимости от условий
        return sum(comp.velocity * comp.fraction for comp in self.components) if self.number_of_fluids > 0 else 0

    @property
    def reynolds(self) -> float:
        """
        Calculates the Reynolds number based on the total density and total viscosity of the medium.
        :return: Reynolds number
        """
        return self.velocity * self.prop.diameter * self.overall_density / self.overall_viscosity

    @property
    def xtt(self) -> float | None:
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
              ((gas.viscosity / liquid.viscosity) ** 0.1) * \
              ((self.velocity / self.prop.diameter) ** 0.5)
        return xtt

    @property
    def flow_mode(self) -> FlowMode | None:  # TODO: предоставить формулы
        """
        Determines the flow mode based on the Lockhart-Martinelli parameter.
        """
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
        """
        Calculates the lambda coefficient based on the Reynolds number.
        """
        reynolds_number = self.reynolds
        if reynolds_number < 2300:
            return 64 / reynolds_number
        else:
            return 0.316 / (reynolds_number ** 0.25)

    @property
    def pressure_loss(self) -> float:
        """
        Calculates the pressure loss in the segment based on the lambda coefficient.
        """
        xi = self.r_lambda * self.length / self.prop.diameter
        return (xi * self.velocity ** 2) * 0.5 * self.overall_density
