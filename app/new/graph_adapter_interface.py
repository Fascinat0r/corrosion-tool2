# graph_adapter_interface.py

from typing import List, Dict, Tuple

from models import PipelineNode, PipelineEdge


class IGraphAdapter:
    """
    Интерфейс-адаптер для работы с графами.
    """

    def add_node(self, node: PipelineNode) -> None:
        """Добавить узел в граф."""

    def add_edge(self, edge: PipelineEdge) -> None:
        """Добавить ребро в граф."""

    def remove_node(self, node_id: str) -> None:
        """Удалить узел по идентификатору."""

    def remove_edge(self, source: str, target: str) -> None:
        """Удалить ребро (source -> target)."""

    def get_nodes(self) -> List[PipelineNode]:
        """Получить список узлов как бизнес-объектов."""

    def get_edges(self) -> List[PipelineEdge]:
        """Получить список рёбер как бизнес-объектов."""

    def update_node(self, node: PipelineNode) -> None:
        """Обновить узел в графе."""

    def all_simple_paths(self, start: str, end: str, cutoff: int = None) -> List[List[str]]:
        """
        Найти все простые пути (без повторения узлов) в ориентированном графе
        от start к end с ограничением глубины cutoff.
        """

    def to_undirected_copy(self):
        """
        Вернуть **неориентированную** копию графа для упрощённых операций
        (например, нахождение циклов, упрощение и т.д.)
        """

    def in_degree(self, node_id: str) -> int:
        """Количество входящих рёбер для узла."""

    def out_degree(self, node_id: str) -> int:
        """Количество исходящих рёбер для узла."""

    def simplify_graph(self, input_nodes: List[str], output_nodes: List[str]) -> Dict[Tuple[str, str], PipelineNode]:
        """
        Схлопнуть узлы со степенью 2, которые не входят в список входных/выходных.
        Удалить цикл длины 3, если он есть - бросить исключение.
        Результат меняет текущий граф.
        """

    def expand_contraction_map(self, contraction_map: Dict[Tuple[str, str], PipelineNode]) -> None:
        """
        Развернуть схлопнутые узлы в графе.
        """

    def force_direction_for_io_edges(self, input_nodes: List[str], output_nodes: List[str]) -> None:
        """
        Определить рёбра, которые однозначно выходят из входных узлов
        или входят в выходные, и пометить их как "зафиксированные" (если нужно).
        """

    def enumerate_orientations(self, input_nodes: List[str], output_nodes: List[str], fixed_edges) -> List[
        'IGraphAdapter']:
        """
        Найти все допустимые способы ориентации для оставшихся незафиксированных рёбер,
        соблюдая правила in_degree/out_degree для input/outputs/внутренних узлов.

        Возвращает список **новых** адаптеров (или графов), каждый из которых -
        одна из вариантов ориентации.
        """

    def visualize(self, title: str = "Pipeline graph"):
        """
        Визуализировать граф, используя matplotlib.
        """

    def build_divider_chains(self, max_depth: int = 10):
        """
        Для каждого узла типа DIVIDER:
          - Находит все простые пути (с учетом направления) от INPUT-узлов к нему и от него к OUTPUT-узлам,
            без повторения узлов (циклов).
          - Каждому найденному пути присваивает уникальное имя (chain_inX или chain_outY).
          - Помечает все рёбра на найденном пути этой меткой.
          - Логирует цепочки.
        """

    def clear_chains(self):
        """
        Удаляет атрибут "chain" из свойств всех рёбер.
        """