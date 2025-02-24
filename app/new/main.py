# main.py
from typing import List

from matplotlib import pyplot as plt

from graph_adapter_nx import GraphAdapterNX
from logger import setup_logger
from pipeline_repository import PipelineRepository


def show_graphs(graphs: List[plt.Figure]) -> None:
    """
    Отображает массив графиков в виде сетки.
    """
    n = len(graphs)
    cols = min(3, n)  # Ограничение на 3 колонки
    rows = (n + cols - 1) // cols  # Вычисление строк

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, graph in zip(axes, graphs):
        graph.canvas.draw()
        ax.imshow(graph.canvas.buffer_rgba())
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    logger = setup_logger()
    json_path = "../../data/pipeline_graph_example.json"

    # Создаем адаптер и репозиторий
    adapter = GraphAdapterNX()
    repo = PipelineRepository(adapter)
    repo.load_from_json(json_path)

    # Получаем варианты ориентаций графа
    variants = repo.get_digraph_variants()
    logger.info(f"Итоговое кол-во вариантов: {len(variants)}")
    graph_figures = []
    # Для каждого варианта:
    for i, variant_adapter in enumerate(variants):
        # Определяем типы узлов
        logger.info(f"=== Вариант графа №{i + 1} ===")
        repo_variant = PipelineRepository(variant_adapter)
        repo_variant.define_node_types()

        # Выполняем один проход расчёта Q (потерь)
        Q_values = repo_variant.solve_system()

        graph_figures.append(repo_variant.draw_graph(title=f"Вариант графа №{i + 1}, neg Q: {sum(1 for q in Q_values if q < 0)}"))

    show_graphs(graph_figures)

    # Можно выбрать те варианты, где нет отрицательных Q, и продолжить расчёты далее.


if __name__ == "__main__":
    main()
