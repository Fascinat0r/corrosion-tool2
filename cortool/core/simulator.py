import copy
import json

from tube_point import TubePoint, calculate_pressure_loss


class PipelineSimulator:
    def __init__(self, data_path, segment_length=5000):
        """Инициализация симулятора с указанием пути к файлу с данными."""
        self.pipeline_segments = None
        self.tube_points = []
        self.data_path = data_path
        self.segment_length = segment_length  # Максимальная длина сегмента трубопровода
        self.load_data()

    def load_data(self):
        """Загрузка данных из JSON файла."""
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        self.pipeline_segments = data['pipeline_segments']

    def setup(self, init_data):
        """Настройка начальной точки трубопровода."""
        initial_point = TubePoint(init_data)
        # Параметры начальной точки могут быть заданы или загружены отдельно, здесь просто инициализация
        self.tube_points.append(initial_point)

    def simulate(self):
        """Моделирование течения в трубопроводе, обработка каждого участка."""
        current_point = self.tube_points[0]
        for segment in self.pipeline_segments:
            self.process_segment(current_point, segment['diameter_m'], segment['length_m'])
            current_point = self.tube_points[-1]  # Обновление текущей точки после обработки сегмента

    def process_segment(self, current_point, diameter, total_length):
        """Обработка одного участка трубопровода, разбивка на сегменты."""
        num_segments = (total_length + self.segment_length - 1) // self.segment_length
        segment_length = self.segment_length
        for _ in range(num_segments):
            actual_length = min(segment_length, total_length)
            total_length -= actual_length
            pressure_loss = calculate_pressure_loss(current_point, actual_length, diameter)
            new_pressure = current_point.pressure - pressure_loss
            new_temperature = current_point.temperature - (diameter * 0.1 * actual_length / 1000)
            next_point = copy.deepcopy(current_point)
            next_point.update_conditions(new_pressure, new_temperature)
            self.tube_points.append(next_point)

    def print_results(self):
        """Вывод результатов моделирования каждого сегмента."""
        for i, point in enumerate(self.tube_points):
            print(f'Segment {i}: Pressure = {point.pressure}, Temperature = {point.temperature}')
