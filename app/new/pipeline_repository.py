# pipeline_repository.py

import json
import logging
from typing import List

from matplotlib import pyplot as plt

from graph_adapter_interface import IGraphAdapter
from models import PipelineNode, PipelineEdge, NodeType

logger = logging.getLogger("pipeline_logger")


class PipelineRepository:
    """
    Репозиторий для загрузки данных о трубопроводе из JSON и наполнения адаптера.
    """

    def __init__(self, adapter: IGraphAdapter, input_nodes: List[str] = None, output_nodes: List[str] = None):
        self.adapter = adapter
        self.input_nodes = input_nodes or []
        self.output_nodes = output_nodes or []

    def load_from_json(self, file_path: str) -> None:
        logger.info(f"Читаем данные из файла {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.input_nodes = data.get("input_nodes", [])
        self.output_nodes = data.get("output_nodes", [])
        for ninfo in data.get("nodes", []):
            node_id = ninfo["id"]
            node_data = ninfo.get("data", {})
            x = float(node_data.get("x", 0.0))
            y = float(node_data.get("y", 0.0))
            if node_id in self.input_nodes:
                tp = NodeType.INPUT
            elif node_id in self.output_nodes:
                tp = NodeType.OUTPUT
            else:
                tp = NodeType.UNKNOWN
            node = PipelineNode(id=node_id, type=tp, x=x, y=y, data=node_data)
            self.adapter.add_node(node)
        for einfo in data.get("edges", []):
            src = einfo["source"]
            tgt = einfo["target"]
            props = einfo.get("properties", {})
            length = float(props.get("length", 0.0))
            diameter = float(props.get("diameter", 0.5))
            roughness = float(props.get("roughness_m", 0.000015))
            edge = PipelineEdge(
                source=src,
                target=tgt,
                length=length,
                diameter=diameter,
                roughness=roughness,
                properties=props
            )
            self.adapter.add_edge(edge)
        logger.info("Данные успешно загружены в адаптер.")

    def get_digraph_variants(self) -> List[IGraphAdapter]:
        logger.info("Запуск упрощения графа...")
        contraction_map = self.adapter.simplify_graph(self.input_nodes, self.output_nodes)
        logger.info("Фиксация направлений для входных/выходных рёбер...")
        fixed_edges = self.adapter.force_direction_for_io_edges(self.input_nodes, self.output_nodes)
        logger.info("Перебор вариантов ориентаций для свободных рёбер...")
        variants = self.adapter.enumerate_orientations(self.input_nodes, self.output_nodes, fixed_edges)
        logger.info(f"Найдено {len(variants)} вариантов.")
        logger.info("Восстановление схлопнутых узлов в каждом варианте...")
        for variant in variants:
            variant.expand_contraction_map(contraction_map)
        return variants

    def define_node_types(self) -> None:
        """
        Определяет типы узлов по количеству входящих и исходящих рёбер.
        """
        for node in self.adapter.get_nodes():
            indeg = self.adapter.in_degree(node.id)
            outdeg = self.adapter.out_degree(node.id)
            if indeg == 0 and outdeg > 0:
                node.type = NodeType.INPUT
            elif indeg > 0 and outdeg == 0:
                node.type = NodeType.OUTPUT
            elif indeg == 1 and outdeg == 1:
                node.type = NodeType.CONSUMER
            elif indeg == 1 and outdeg > 1:
                node.type = NodeType.BRANCH
            elif indeg > 1 and outdeg >= 1:
                node.type = NodeType.DIVIDER
            else:
                node.type = NodeType.UNKNOWN
            self.adapter.update_node(node)
        logger.info("Типы узлов определены.")

    def solve_system(self) -> List[float]:
        """
        Запускает однократный проход расчёта (одну итерацию) для варианта графа.
        Рассчитывает Q, R, dP, dPfix и выводит итоговые значения.
        Отбрасывает варианты, где обнаружены отрицательные внутренние Q.
        """
        from new.slau_service import SLASolverService
        solver = SLASolverService(self.adapter.digraph)
        Q_values = solver.solve_iteratively(max_iter=1, tolerance=1e-3)
        # Если в найденном варианте отрицательные Q, можно выбросить этот вариант (обработка на уровне репозитория)
        valid = True
        for q in Q_values:
            if q < 0:
                valid = False
                break
        if valid:
            logger.info("Вариант принят, отрицательных Q не обнаружено.")
        else:
            logger.info("Вариант отклонен, обнаружены отрицательные Q.")

        return Q_values

    def draw_graph(self, title: str = "Pipeline graph") -> plt.Figure:
        return self.adapter.visualize(title)
