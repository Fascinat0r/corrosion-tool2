import json

import numpy as np
import pandas as pd

from cortool.models.component import Component
from cortool.models.pipe_section import PipeSection, PipeProperties


class Pipeline:
    """
    Класс, представляющий трубопровод. Содержит секции трубопровода и начальные компоненты потока.
    Это основной класс для симуляции потока в трубопроводе.
    """

    def __init__(self, sections_config_path: str, components_config_path: str):
        # Список секций трубопровода. Предполагаем о они идут последовательно, без разветвлений
        self.sections = []  # TODO: Реализовать ветвление трубопровода
        self.load_sections(sections_config_path)
        # Начальные компоненты потока
        self.initial_components = []  # List[Component] - список начальных компонентов потока
        self.load_components(components_config_path)

    def load_sections(self, filepath: str):
        """Загружает секции трубопровода из JSON файла."""
        with open(filepath, 'r') as file:
            data = json.load(file)
            if self.sections is None:
                self.sections = []
            for section_data in data['pipeline_sections']:
                # Создаем объект PipeProperties для каждой секции, содержащий свойства трубы
                properties = PipeProperties(
                    diameter=section_data['diameter_m'],
                    length=section_data['length_m'],
                    roughness=section_data['roughness_m'],
                    angle=section_data['angle_deg'],
                    heat_transfer_coefficient=section_data['heat_transfer_coefficient'],
                    ambient_temperature=section_data['ambient_temperature']
                )
                # Создаем объект PipeSection для каждой секции и добавляем в список секций
                new_section = PipeSection(prop=properties)
                self.sections.append(new_section)

    def load_components(self, config_path: str):
        """Загрузка начальных компонентов потока из файла JSON."""
        with open(config_path, 'r') as file:
            config = json.load(file)
            for comp in config['components']:
                component = Component(comp['name'],
                                      comp['temperature'],
                                      comp['pressure'],
                                      comp['velocity'],
                                      comp['composition'])
                self.initial_components.append(component)

    def simulate(self):
        """Симулирует поток через всю систему трубопровода."""
        current_components = self.initial_components
        for idx, section in enumerate(self.sections):
            # Симулируем поток в текущей секции
            current_components = section.simulate_flow(current_components, segment_length=1000)
            # Производим адаптацию параметров потока при переходе к следующей секции
            if idx < len(self.sections) - 1:
                current_components = self.adapt_flow(idx, current_components)

            for comp in current_components:
                print(f"{comp.substance.name} - Density: {comp.density}, Viscosity: {comp.viscosity}")

    def adapt_flow(self, idx, components):
        """Адаптирует параметры потока при изменении диаметра трубы."""
        current_section = self.sections[idx]
        next_section = self.sections[idx + 1]
        # Рассчитываем площади поперечных сечений для текущей и следующей секции
        A1 = np.pi * (current_section.prop.diameter / 2) ** 2
        A2 = np.pi * (next_section.prop.diameter / 2) ** 2

        # Уравнение неразрывности для скорости
        v1 = np.mean([comp.velocity for comp in components])
        v2 = v1 * (A1 / A2)

        # Обновляем скорость компонентов для следующей секции
        for comp in components:
            comp.velocity = v2

        # Расчет средней плотности смеси #T
        total_density = sum(comp.density * comp.composition for comp in components)

        # Уравнение Бернулли для давления
        p1 = np.mean([comp.pressure for comp in components])
        p2 = p1 + 0.5 * total_density * (v1 ** 2 - v2 ** 2)

        # Обновляем давление компонентов для следующей секции
        for comp in components:
            comp.pressure = p2

        return components

    def save_sections_data_to_csv(self, file_path):
        """
        Сохраняет данные о каждом сегменте трубопровода в CSV файл.
        """
        data = []
        for section_index, section in enumerate(self.sections):
            for segment_index, segment in enumerate(section.segments):
                segment_data = {
                    "Section Index": section_index,
                    "Segment Index": segment_index,
                    "Segment Length": segment.length,
                    "Diameter": section.prop.diameter,
                    "Roughness": section.prop.roughness,
                    "Angle": section.prop.angle,
                    "Heat Transfer Coefficient": section.prop.heat_transfer_coefficient,
                    "Velocity": segment.velocity,
                    "Ambient Temperature": section.prop.ambient_temperature,
                    "Average Pressure": segment.pressure,
                    "Average Temperature": segment.temperature,
                    "Average Density": segment.overall_density,
                    "Average Viscosity": segment.overall_viscosity,
                    "Flow Mode": segment.flow_mode.name if segment.flow_mode else "Undefined",
                    "Reynolds Number": segment.reynolds,
                    "xtt": segment.xtt,
                    "Friction Factor": segment.friction_factor if segment.friction_factor else "N/A",
                    "Pressure Loss": segment.pressure_loss(),
                    "Temperature Loss": segment.temperature_loss()
                }
                data.append(segment_data)

        # Создаем DataFrame
        df = pd.DataFrame(data)
        # Сохраняем в CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

# Пример использования
# pipeline = Pipeline('../data/pipes.json', '../data/input.json')
# pipeline.simulate()
