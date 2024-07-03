import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pipeline_data(file_path):
    # Читаем данные из CSV файла
    df = pd.read_csv(file_path)

    # Добавим колонку для непрерывного индекса сегментов
    df['Cumulative Segment Index'] = np.cumsum(df['Segment Index'].groupby(df['Section Index']).cumcount() + 1)

    # Настройки отображения
    plt.figure(figsize=(14, 10))

    # Создаем цветовую карту для секций
    colors = plt.cm.viridis(np.linspace(0, 1, df['Section Index'].nunique()))

    # График изменения давления вдоль трубопровода
    plt.subplot(3, 1, 1)
    for i, section in enumerate(df['Section Index'].unique()):
        section_data = df[df['Section Index'] == section]
        plt.plot(section_data['Cumulative Segment Index'], section_data['Average Pressure'], marker='o',
                 color=colors[i], label=f'Section {section}')
    plt.title('Average Pressure Along Pipeline Segments')
    plt.xlabel('Cumulative Segment Index')
    plt.ylabel('Pressure (Pa)')
    plt.legend()

    # График изменения температуры вдоль трубопровода
    plt.subplot(3, 1, 2)
    for i, section in enumerate(df['Section Index'].unique()):
        section_data = df[df['Section Index'] == section]
        plt.plot(section_data['Cumulative Segment Index'], section_data['Average Temperature'], marker='o',
                 color=colors[i], label=f'Section {section}')
    plt.title('Average Temperature Along Pipeline Segments')
    plt.xlabel('Cumulative Segment Index')
    plt.ylabel('Temperature (K)')
    plt.legend()

    # График режимов потока
    plt.subplot(3, 1, 3)
    for i, section in enumerate(df['Section Index'].unique()):
        section_data = df[df['Section Index'] == section]
        plt.plot(section_data['Cumulative Segment Index'], section_data['Flow Mode'], marker='o', color=colors[i],
                 label=f'Section {section}')
    plt.title('Flow Mode Along Pipeline Segments')
    plt.xlabel('Cumulative Segment Index')
    plt.ylabel('Flow Mode')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Пример использования
# plot_pipeline_data('path_to_your_csv_file.csv')
