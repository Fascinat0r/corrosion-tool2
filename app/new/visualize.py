import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from models import NodeType


def visualize_graph(graph, title: str = "Graph Visualization", edge_label: str = "dP", path=None, show=True) -> None:
    """
    Визуализировать граф:
      - Цвет узлов берётся из node.type.color.
      - На рёбрах подписываем свойство properties[edge_label], например 'dP' или 'Q'.
    Возвращает figure, чтобы можно было сохранить или отобразить.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Позиции узлов: берём из (x, y)
    pos = {}
    for n, dat in graph.nodes(data=True):
        x = dat.get("x", 0.0)
        y = dat.get("y", 0.0)
        pos[n] = (x, y)

    # Цвета узлов
    node_colors = []
    for n, dat in graph.nodes(data=True):
        node_type = dat.get("type", NodeType.UNKNOWN)
        node_colors.append(node_type.color)

    # Отрисовка узлов
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=600,
        ax=ax
    )

    # Отрисовка рёбер
    nx.draw_networkx_edges(
        graph, pos,
        arrowstyle="->", arrowsize=15,
        ax=ax
    )

    # Подписи для узлов (id)
    nx.draw_networkx_labels(
        graph, pos,
        font_size=9, font_color="white",
        ax=ax
    )

    # Подписи для рёбер: берем из properties[edge_label]
    edge_labels = {}
    for u, v, edata in graph.edges(data=True):
        val = edata.get("properties", {}).get(edge_label, None)
        if val is not None:
            edge_labels[(u, v)] = f"{edge_label}={val:.3f}"

    # Используем rotate=False, чтобы убрать кривые стрелки (может исправить ошибку)
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_size=9, ax=ax,
        rotate=False  # <-- Исправление ошибки
    )

    # Легенда по типам узлов
    unique_types = set([d.get("type", NodeType.UNKNOWN) for _, d in graph.nodes(data=True)])
    legend_patches = []
    for t in unique_types:
        if t and isinstance(t, NodeType):
            patch = mpatches.Patch(color=t.color, label=t.title)
            legend_patches.append(patch)
    if legend_patches:
        ax.legend(handles=legend_patches, loc="best")

    ax.set_title(title)
    ax.axis("equal")  # чтобы соотношение осей x/y не искажалось
    ax.axis("off")

    fig.tight_layout()

    if path:
        fig.savefig(path)

    if show:
        plt.show()
