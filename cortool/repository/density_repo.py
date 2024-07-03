import os

import pandas as pd
from scipy.interpolate import interp1d


class DensityModule:
    """
    Модуль для работы с данными плотности субстанций на основе эмпирических данных.
    Позволяет рассчитывать плотность субстанции по температуре с использованием интерполяции по данным из файлов.
    """

    def __init__(self, base_path):
        """Инициализация модуля с указанием пути к папке с данными."""
        self.base_path = base_path
        self.density_data = {}

    def load_density_data(self, substance_name):
        """Загрузка данных плотности из файла в зависимости от субстанции."""
        file_path = os.path.join(self.base_path, f'density_{substance_name}.csv')
        if os.path.exists(file_path):
            # Загрузка только колонок с температурой и плотностью
            data = pd.read_csv(file_path, usecols=['Temperature K', 'Density kg/m3'])
            # Переименование колонок для удобства
            data.columns = ['Temperature', 'Density']
            self.density_data[substance_name] = interp1d(data['Temperature'], data['Density'], kind='linear',
                                                         fill_value='extrapolate')
        else:
            raise FileNotFoundError(f"No density data available for {substance_name}")

    def get_density(self, substance_name, temperature):
        """Получение плотности по заданной температуре с использованием интерполяции эмпирических данных."""
        if substance_name not in self.density_data:
            self.load_density_data(substance_name)
        return self.density_data[substance_name](temperature)


density_module = DensityModule('cortool/data/')
# Пример использования
# try:
#     density_ethanol = density_module.get_density('ethanol', 300)
#     print(f"Density of Ethanol at 300K: {density_ethanol}")
# except FileNotFoundError as e:
#     print(e)
