from typing import List

from app.models.component import Component
from app.models.segment import Segment, PipeProperties


class PipeSection:
    """
    Класс, представляющий цельную однородную секцию трубопровода.
    Для моделирования потока, секция разбивается на сегменты, в каждом из которых происходит симуляция.
    """
    prop: PipeProperties  # Свойства трубы
    segments: List[Segment] = None  # Список сегментов, на которые разбивается секция для моделирования потока

    def __init__(self, prop: PipeProperties):
        self.prop = prop
        if self.segments is None:
            self.segments = []

    def simulate_flow(self, initial_components: List[Component], segment_length: float):
        """Создает сегменты симулируя поток в секции."""
        remaining_length = self.prop.length
        current_components = initial_components
        # Пока не закончится длина секции, создаем сегменты
        try:
            while remaining_length > 0:
                # Выбираем длину текущего сегмента - минимум из длины сегмента и оставшейся длины секции
                current_segment_length = min(segment_length, remaining_length)
                # Создаем сегмент с выбранной длиной
                segment = Segment(prop=self.prop, length=current_segment_length,
                                  components=current_components)
                segment.simulate()
                self.segments.append(segment)

                # Обновляем компоненты для следующего сегмента на основе выхода из текущего
                current_components = segment.get_output_components()
                remaining_length -= current_segment_length
        except ValueError as e:
            raise ValueError(f"Error while simulating flow in the section:\n {e}")
        return current_components

    def calculate_overall_properties(self):
        """Вычисляет общие свойства секции на основе свойств сегментов."""
        total_density = sum(seg.overall_density for seg in self.segments) / len(self.segments)
        total_viscosity = sum(seg.overall_viscosity for seg in self.segments) / len(self.segments)
        print(f"Average density of the section: {total_density}")
        print(f"Average viscosity of the section: {total_viscosity}")

# Пример использования:
# properties = PipeProperties(0.5, 1000.0, 0.01, 0, 10, 300)
# section = PipeSection(properties)
# components = [Component('ethanol', 375, 16250000, 10, 0.5, Phase.LIQUID),
#               Component('nitrogen', 375, 16250000, 10, 0.5, Phase.GAS)]
#
# section.simulate_flow(components, 20)
# section.calculate_overall_properties()
