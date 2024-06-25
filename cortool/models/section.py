from typing import List

from cortool.models.component import Component
from cortool.models.segment import Segment, PipeProperties


class Section:
    """
    Класс, представляющий цельную однородную секцию трубопровода.
    Для моделирования потока, секция разбивается на сегменты, в каждом из которых происходит симуляция.
    """
    prop: PipeProperties
    segments: List[Segment] = None

    def __init__(self, prop: PipeProperties):
        self.prop = prop
        if self.segments is None:
            self.segments = []

    def simulate_flow(self, initial_components: List[Component], segment_length: float):
        """Создает сегменты симулируя поток в секции."""
        remaining_length = self.prop.length
        current_components = initial_components

        while remaining_length > 0:
            current_segment_length = min(segment_length, remaining_length)
            segment = Segment(prop=self.prop, length=current_segment_length, components=current_components.copy())
            segment.simulate()  # Метод для симуляции процессов в сегменте #TODO: реализовать
            self.segments.append(segment)

            # Обновляем компоненты для следующего сегмента на основе выхода из текущего
            current_components = segment.get_output_components()  # TODO: реализовать
            remaining_length -= current_segment_length

    def calculate_overall_properties(self):
        """Вычисляет общие свойства секции на основе свойств сегментов."""
        total_density = sum(seg.overall_density for seg in self.segments) / len(self.segments)
        total_viscosity = sum(seg.overall_viscosity for seg in self.segments) / len(self.segments)
        print(f"Average density of the section: {total_density}")
        print(f"Average viscosity of the section: {total_viscosity}")


# Пример использования:
properties = PipeProperties(0.5, 1000.0, 0.01, 0)
section = Section(properties)
section.calculate_overall_properties()
