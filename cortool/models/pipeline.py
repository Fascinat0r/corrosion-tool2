import json
from typing import List

from cortool.models.component import Component
from cortool.models.pipe_section import PipeSection, PipeProperties


class Pipeline:
    def __init__(self, sections_config_path: str, components_config_path: str):
        self.sections = []
        self.load_sections(sections_config_path)
        self.initial_components = self.load_components(components_config_path)

    def load_sections(self, filepath: str):
        """Загружает секции трубопровода из JSON файла."""
        with open(filepath, 'r') as file:
            data = json.load(file)
            if self.sections is None:
                self.sections = []
            for section_data in data['pipeline_sections']:
                properties = PipeProperties(
                    diameter=section_data['diameter_m'],
                    length=section_data['length_m'],
                    roughness=section_data['roughness_m'],
                    angle=section_data['angle_deg'],
                    heat_transfer_coefficient=section_data['heat_transfer_coefficient'],
                    ambient_temperature=section_data['ambient_temperature']
                )
                new_section = PipeSection(prop=properties)
                self.sections.append(new_section)

    def load_components(self, config_path: str) -> List[Component]:
        """Загрузка начальных компонентов потока из файла JSON."""
        with open(config_path, 'r') as file:
            config = json.load(file)
            components = []
            for comp in config['components']:
                component = Component(comp['name'],
                                      comp['temperature'],
                                      comp['pressure'],
                                      comp['velocity'],
                                      comp['fraction'],
                                      comp['phase'])
                components.append(component)
            return components

    def simulate(self):
        """Симулирует поток через всю систему трубопровода."""
        current_components = self.initial_components
        for section in self.sections:
            current_components = section.simulate_flow(current_components,
                                                       segment_length=1000)  # Предполагаемая длина сегмента
            # Вывод результатов каждой секции
            print(f"Results after section with diameter {section.prop.diameter}:")
            for comp in current_components:
                print(f"{comp.substance.name} - Density: {comp.density}, Viscosity: {comp.viscosity}")


# Пример использования
pipeline = Pipeline('../data/pipes.json', '../data/input.json')
pipeline.simulate()
