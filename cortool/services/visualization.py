import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(x, y, labels, title, x_label, y_label, color_map=None, subplot_index=1, total_plots=1):
    """
    Универсальная функция для построения графиков.
    """
    if color_map is None:
        color_map = plt.cm.viridis(np.linspace(0, 1, len(np.unique(labels))))

    plt.subplot(total_plots, 1, subplot_index)
    for i, label in enumerate(np.unique(labels)):
        section_mask = (labels == label)
        plt.plot(x[section_mask], y[section_mask], marker='o', color=color_map[i], label=f'Section {label}')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def plot_pipeline_data(file_path):
    """
    Функция для визуализации данных трубопровода из CSV файла.
    """
    df = pd.read_csv(file_path)
    df['Cumulative Segment Index'] = np.cumsum(df['Segment Index'].groupby(df['Section Index']).cumcount() + 1)

    plt.figure(figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, df['Section Index'].nunique()))

    # Давление вдоль трубопровода
    plot_data(df['Cumulative Segment Index'], df['Average Pressure'], df['Section Index'],
              'Average Pressure Along Pipeline Segments', 'Cumulative Segment Index', 'Pressure (Pa)',
              color_map=colors, subplot_index=1, total_plots=3)

    # Температура вдоль трубопровода
    plot_data(df['Cumulative Segment Index'], df['Average Temperature'], df['Section Index'],
              'Average Temperature Along Pipeline Segments', 'Cumulative Segment Index', 'Temperature (K)',
              color_map=colors, subplot_index=2, total_plots=3)

    # Режим потока вдоль трубопровода
    plot_data(df['Cumulative Segment Index'], df['Flow Mode'], df['Section Index'],
              'Flow Mode Along Pipeline Segments', 'Cumulative Segment Index', 'Flow Mode',
              color_map=colors, subplot_index=3, total_plots=3)

    plt.tight_layout()
    plt.show()


def plot_all_columns(file_path):
    """
    Функция для визуализации всех числовых данных из CSV файла.
    """
    df = pd.read_csv(file_path)

    # Генерация индекса для каждого сегмента
    df['Segment Index'] = np.arange(len(df))

    # Определение числа числовых столбцов
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    total_plots = len(numeric_columns)

    plt.figure(figsize=(14, total_plots * 5))  # Динамически настраиваем размер фигуры

    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(total_plots, 1, i)
        plt.plot(df['Segment Index'], df[column], marker='o', linestyle='-', color='blue')
        plt.title(column)
        plt.xlabel('Segment Index')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.show()

# Пример использования
# plot_all_columns('path_to_your_csv_file.csv')
# plot_pipeline_data('path_to_your_csv_file.csv')
