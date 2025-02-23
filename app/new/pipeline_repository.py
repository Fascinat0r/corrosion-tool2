# pipeline_repository.py

import json
import logging

from graph_adapter_interface import IGraphAdapter
from models import PipelineNode, PipelineEdge, NodeType

logger = logging.getLogger("pipeline_logger")


class PipelineRepository:
    """
    Репозиторий для загрузки данных о трубопроводе из JSON и наполняет адаптер графа.
    """

    def __init__(self, adapter: IGraphAdapter, input_nodes: list = None, output_nodes: list = None):
        self.adapter = adapter
        self.output_nodes = input_nodes
        self.input_nodes = output_nodes

    def load_from_json(self, file_path: str):
        """
        Считывает JSON файл и добавляет в адаптер узлы/рёбра.
        """
        logger.info(f"Читаем данные из файла {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.input_nodes = data.get("input_nodes", [])
        self.output_nodes = data.get("output_nodes", [])

        nodes_data = data.get("nodes", [])
        edges_data = data.get("edges", [])

        # Сначала создаём узлы
        for ninfo in nodes_data:
            node_id = ninfo["id"]
            node_dict = ninfo.get("data", {})
            x = float(node_dict.get("x", 0.0))
            y = float(node_dict.get("y", 0.0))
            # Определяем тип: если в input_nodes => INPUT, в output_nodes => OUTPUT
            if node_id in self.input_nodes:
                tp = NodeType.INPUT
            elif node_id in self.output_nodes:
                tp = NodeType.OUTPUT
            else:
                tp = NodeType.UNKNOWN
            node = PipelineNode(id=node_id, type=tp, x=x, y=y, attributes=node_dict)
            self.adapter.add_node(node)

        # Создаём рёбра
        for einfo in edges_data:
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

        logger.info("Данные загружены в адаптер.")

    def get_digraph_variants(self):
        """
        Подготовить граф: схлопнуть 'промежуточные' узлы, зафиксировать рёбра и т.д.
        """
        logger.info("Запуск упрощения графа...")
        contraction_map = self.adapter.simplify_graph(self.input_nodes, self.output_nodes)
        logger.info("Устанавливаем направления рёбер, выходящих из входов/входящих в выходы...")
        fixed_edges = self.adapter.force_direction_for_io_edges(self.input_nodes, self.output_nodes)
        logger.info("Начинаем перебор всех возможных ориентаций (кроме зафиксированных)...")
        variants = self.adapter.enumerate_orientations(self.input_nodes, self.output_nodes, fixed_edges)
        logger.info(f"Количество итоговых вариантов: {len(variants)}")
        logger.info("Восстановление исходных графов...")
        for variant in variants:
            variant.expand_contraction_map(contraction_map)
        return variants

    def draw_graph(self, title: str = "Pipeline graph"):
        self.adapter.visualize(title)

    def define_node_types(self):
        """
        Определить типы узлов в зависимости от количества входящих и исходящих рёбер.
        Входные узлы - если in_degree = 0, out_degree > 0
        Выходные узлы - если in_degree > 0, out_degree = 0
        Потребители - если in_degree = 1, out_degree = 1
        Ветвление - если in_degree = 1, out_degree > 1
        Водораздел - если in_degree > 1, out_degree > 1
        """
        for node in self.adapter.get_nodes():
            in_degree = self.adapter.in_degree(node.id)
            out_degree = self.adapter.out_degree(node.id)
            if in_degree == 0 and out_degree > 0:
                node.type = NodeType.INPUT
            elif in_degree > 0 and out_degree == 0:
                node.type = NodeType.OUTPUT
            elif in_degree == 1 and out_degree == 1:
                node.type = NodeType.CONSUMER
            elif in_degree == 1 and out_degree > 1:
                node.type = NodeType.BRANCH
            elif in_degree > 1 and out_degree >= 1:
                node.type = NodeType.DIVIDER
            else:
                node.type = NodeType.UNKNOWN
            self.adapter.update_node(node)
        logger.info("Типы узлов определены.")

    # расчёт СЛАУ
    def solve_system(self):
        """
        Решить систему уравнений для трубопровода.
        """
        self.adapter.build_divider_chains()
