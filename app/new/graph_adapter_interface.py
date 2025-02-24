# graph_adapter_interface.py

from typing import List, Dict, Tuple

from matplotlib import pyplot as plt

from models import PipelineNode, PipelineEdge


class IGraphAdapter:
    """
    Интерфейс-адаптер для работы с графами.
    """

    def __init__(self):
        self.digraph = None

    def add_node(self, node: PipelineNode) -> None:
        """Добавить узел в граф."""
        raise NotImplementedError

    def add_edge(self, edge: PipelineEdge) -> None:
        """Добавить ребро в граф."""
        raise NotImplementedError

    def remove_node(self, node_id: str) -> None:
        """Удалить узел по идентификатору."""
        raise NotImplementedError

    def remove_edge(self, source: str, target: str) -> None:
        """Удалить ребро (source -> target)."""
        raise NotImplementedError

    def get_nodes(self) -> List[PipelineNode]:
        """Получить список узлов как бизнес-объектов."""
        raise NotImplementedError

    def get_edges(self) -> List[PipelineEdge]:
        """Получить список рёбер как бизнес-объектов."""
        raise NotImplementedError

    def update_node(self, node: PipelineNode) -> None:
        """Обновить узел в графе."""
        raise NotImplementedError

    def all_simple_paths(self, start: str, end: str, cutoff: int = None) -> List[List[str]]:
        """
        Найти все простые пути (без повторения узлов) в ориентированном графе от start к end с ограничением глубины.
        """
        raise NotImplementedError

    def to_undirected_copy(self):
        """
        Вернуть неориентированную копию графа для упрощённых операций.
        """
        raise NotImplementedError

    def in_degree(self, node_id: str) -> int:
        """Количество входящих рёбер для узла."""
        raise NotImplementedError

    def out_degree(self, node_id: str) -> int:
        """Количество исходящих рёбер для узла."""
        raise NotImplementedError

    def simplify_graph(self, input_nodes: List[str], output_nodes: List[str]) -> Dict[Tuple[str, str], PipelineNode]:
        """
        Схлопнуть узлы со степенью 2, не входящие в input/ output.
        Возвращает словарь contraction_map.
        """
        raise NotImplementedError

    def expand_contraction_map(self, contraction_map: Dict[Tuple[str, str], PipelineNode]) -> None:
        """
        Восстановить схлопнутые узлы в графе.
        """
        raise NotImplementedError

    def force_direction_for_io_edges(self, input_nodes: List[str], output_nodes: List[str]) -> List[Tuple[str, str]]:
        """
        Фиксировать направления рёбер для узлов:
          - Для входных узлов: все рёбра исходящие (фиксируются как u->v).
          - Для выходных узлов: все рёбра входящие (фиксируются как u->v).
        Возвращает список зафиксированных ребер.
        """
        raise NotImplementedError

    def enumerate_orientations(self, input_nodes: List[str], output_nodes: List[str],
                               fixed_edges: List[Tuple[str, str]]) -> List['IGraphAdapter']:
        """
        Перебрать все допустимые варианты ориентации для свободных рёбер.
        Возвращает список адаптеров (вариантов графа).
        """
        raise NotImplementedError

    def build_divider_chains(self, max_depth: int = 10) -> None:
        """
        Для каждого узла DIVIDER найти простые пути от входных и к выходным узлам (с учетом направления),
        присвоить уникальные chain‑метки, и проставить их на рёбрах.
        """
        raise NotImplementedError

    def clear_chains(self) -> None:
        """
        Сбросить chain‑метки у всех рёбер.
        """
        raise NotImplementedError

    def visualize(self, title: str = "Pipeline graph") -> plt.Figure:
        """
        Визуализировать граф с использованием matplotlib.
        """
        raise NotImplementedError
