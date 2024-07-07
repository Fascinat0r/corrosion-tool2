from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import List

from thermopack.cubic import SoaveRedlichKwong

from cortool.models.component import Component


class FlowMode(Enum):
    """
    Enum класс для определения режима потока в трубе.
    """
    BUBBLE = auto()
    PLUG = auto()
    SLUG = auto()
    ANNULAR = auto()
    MIST = auto()
    UNDEFINED = auto()


@dataclass
class PipeProperties:
    """
    Класс, представляющий свойства трубы.
    """
    diameter: float
    length: float
    roughness: float
    angle: float  # TODO: заменить на три составляющие угла
    heat_transfer_coefficient: float  # Коэффициент теплоотдачи, Вт/(м²·К)
    ambient_temperature: float  # Температура окружающей среды, К


class Segment:
    """
    Класс, представляющий один сегмент от цельной трубы.
    Используется для моделирования потока внутри трубы.
    """

    def __init__(self, prop: PipeProperties, length: float, components: List[Component] = None):
        self.prop = prop  # Свойства трубы, сегментом которого является объект
        self.length = length  # Длина сегмента
        if components is None:
            self.components = []  # Список компонентов в потоке
        else:
            self.components = components

    def get_output_components(self):
        """
        Рассчитывает выходные параметры для каждого компонента в сегменте.
        То есть модель трубопровода преполагает, что мы пренебрегаем изменением свойств внутри сегмента.
        Поэтому мы считаем потери только на концах сегмента.
        """
        delta_T = self.temperature_loss()
        delta_P = self.pressure_loss()
        new_temperature = self.temperature - delta_T
        new_pressure = self.pressure - delta_P

        # Обновление состояния каждого компонента
        for component in self.components:
            component.temperature = new_temperature
            component.pressure = new_pressure
            # Fraction не меняется
            # Phase не меняется
            # Velocity не меняется
        return self.components

    @property
    def number_of_fluids(self) -> int:
        """
        Возвращает количество компонентов в потоке.
        """
        return len(self.components)

    @property
    def overall_density(self) -> float:  # TODO: необходимо дописать учёт агрегатного состояния
        """
        Вычисляет общую плотность потока на основе плотности каждого компонента.
        """
        return sum(comp.density * comp.composition for comp in self.components) if self.number_of_fluids > 0 else 0

    @property
    def overall_viscosity(self) -> float:
        # TODO: Пример расчета общей вязкости, требует более сложной логики в зависимости от условий
        """
        Вычисляет общую вязкость потока на основе вязкости каждого компонента.
        """
        return sum(comp.viscosity * comp.composition for comp in self.components) if self.number_of_fluids > 0 else 0

    @cached_property
    def velocity(self) -> float:
        # TODO: Пример расчета общей вязкости, требует более сложной логики в зависимости от условий
        """
        Вычисляет общую скорость потока на основе скорости каждого компонента.
        """
        return sum(comp.velocity * comp.composition for comp in self.components) if self.number_of_fluids > 0 else 0

    @cached_property
    def temperature(self) -> float:
        """
        Вычисляет общую температуру потока на основе температуры каждого компонента.
        """
        return sum(comp.temperature * comp.composition for comp in self.components) if self.number_of_fluids > 0 else 0

    @cached_property
    def pressure(self) -> float:
        """
        Вычисляет общее давление потока на основе давления каждого компонента.
        """
        return sum(comp.pressure * comp.composition for comp in self.components) if self.number_of_fluids > 0 else 0

    @property
    def reynolds(self) -> float:
        """
        Вычисляет число Рейнольдса на основе общей плотности и общей вязкости среды.
        :return: Число Рейнольдса
        """
        return self.velocity * self.prop.diameter * self.overall_density / self.overall_viscosity

    @property
    def xtt(self) -> float | None:
        """
        Вычисляет параметр Локхарта-Мартинелли для определения режима потока.
        """
        liquid_density = sum(comp.densities[0] * comp.fractions[0] for comp in self.components if
                             comp.densities[0] is not None and comp.fractions[0] is not None)
        gas_density = sum(comp.densities[1] * comp.fractions[1] for comp in self.components if
                          comp.densities[1] is not None and comp.fractions[1] is not None)
        liquid_viscosity = sum(
            comp.substance.get_viscosity(self.temperature) * comp.fractions[0] for comp in self.components if
            comp.fractions[0] is not None)
        gas_viscosity = sum(
            comp.substance.get_viscosity(self.temperature) * comp.fractions[1] for comp in self.components if
            comp.fractions[1] is not None)

        if not liquid_density or not gas_density or liquid_density <= 0 or gas_density <= 0:
            return None  # Если плотность не определена или нулевая

        if self.velocity <= 0 or self.prop.diameter <= 0:
            raise ValueError("Скорость потока и диаметр трубы должны быть больше нуля.")

        # Используя усреднённые свойства жидкости и газа для вычисления xtt
        xtt = ((1.096 / liquid_density) ** 0.5) * \
              ((liquid_density / gas_density) ** 0.25) * \
              ((gas_viscosity / liquid_viscosity) ** 0.1) * \
              ((self.velocity / self.prop.diameter) ** 0.5)
        return xtt

    @property
    def flow_mode(self) -> FlowMode | None:  # TODO: предоставить формулы
        """
        Определяет режим потока на основе параметра Локхарта-Мартинелли.
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
        Определяет коэффициент трения на основе параметра Локхарта-Мартинелли.
        Этот коэффициент трения используется для расчета вязкости в многофазном потоке.
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
        Вычисляет лямбда-коэффициент на основе числа Рейнольдса.
        """
        reynolds_number = self.reynolds
        if reynolds_number < 2300:
            return 64 / reynolds_number
        else:
            return 0.316 / (reynolds_number ** 0.25)

    def pressure_loss(self) -> float:  # TODO: предоставить формулу
        """
        Вычисляет потерю давления в сегменте на основе лямбда-коэффициента.
        """
        xi = self.r_lambda * self.length / self.prop.diameter
        return (xi * self.velocity ** 2) * 0.5 * self.overall_density

    def temperature_loss(self) -> float:
        """
        Расчет потери температуры для сегмента трубы.
        """
        # Расчет общего массового потока и теплоемкости
        total_mass_flow = sum(comp.density * comp.composition for comp in self.components)  # Общий массовый поток
        total_heat_capacity = sum(
            comp.substance.specific_heat_capacity * comp.density * comp.composition for comp in self.components)
        if total_mass_flow == 0 or total_heat_capacity == 0:
            return 0
        # Расчет теплового потока и изменения температуры
        q = self.prop.heat_transfer_coefficient * (
                self.temperature - self.prop.ambient_temperature)  # Тепловой поток
        delta_temperature = q / (total_mass_flow * total_heat_capacity)  # Изменение температуры
        return delta_temperature

    def simulate(self):
        """
        Симулирует поток через сегмент, используя ThermoPack для расчета фазового равновесия и других свойств.
        """
        # Пример использования ThermoPack для инициализации уравнения состояния (EoS)
        component_names = ','.join([comp.substance.thermopack_id for comp in self.components])
        eos = SoaveRedlichKwong(component_names)  # TODO: Обработать ошибку ненахода флюидов в базе термопака

        # Состав компонентов в мольных долях
        x = [comp.composition for comp in self.components]

        # Температура и давление в начале сегмента
        T_initial = self.temperature
        p_initial = self.pressure

        # Выполнение TP-флэш расчёта
        flsh = eos.two_phase_tpflash(T_initial, p_initial, x)

        if flsh.T != T_initial:
            raise Warning("Температура после флэш-расчёта не совпадает с начальной")

        # Обновление состояния компонентов на основе флэш-расчёта
        for idx, comp in enumerate(self.components):
            # Установка фракций для жидкости и газа
            comp.fractions = (flsh.x[idx], flsh.y[idx])

            # Расчет специфического объема и плотности для каждой фазы #TODO: Проверить правильность расчета
            v_l, = eos.specific_volume(T_initial, p_initial, flsh.x, eos.LIQPH)
            v_g, = eos.specific_volume(T_initial, p_initial, flsh.y, eos.VAPPH)
            comp.densities = (comp.substance.molar_mass / v_l, comp.substance.molar_mass / v_g)
