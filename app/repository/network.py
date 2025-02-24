# repository/network.py
import json
import logging
from typing import List

import networkx as nx

from models.node import Node, Pipe, NodeType

logger = logging.getLogger(__name__)


class PipelineGraphRepository:
    """
    Репозиторий для загрузки/сохранения графовой структуры из JSON,
    с назначением типов узлов и базовых проверок.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def load_graph(self, json_path) -> nx.DiGraph:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {json_path}: {e}")
            raise

        input_nodes = data.get("input_nodes", [])
        output_nodes = data.get("output_nodes", [])

        # Загрузка узлов
        for n_data in data.get("nodes", []):
            node_id = n_data.get("id")
            if not node_id:
                raise ValueError("Узел без id недопустим.")
            coords = n_data.get("data", {})
            x = float(coords.get("x", 0.0))
            y = float(coords.get("y", 0.0))
            # Тип по спискам
            node_type = NodeType.UNKNOWN
            if node_id in input_nodes:
                node_type = NodeType.INPUT
            elif node_id in output_nodes:
                node_type = NodeType.OUTPUT

            self.graph.add_node(
                node_id,
                data=coords,
                type=node_type,
                x=x,
                y=y
            )

        # Загрузка рёбер
        for e_data in data.get("edges", []):
            src = e_data.get("source")
            tgt = e_data.get("target")
            props = e_data.get("properties", {})
            if not (src and tgt):
                logger.warning("Обнаружено ребро без source или target")
                continue
            self.graph.add_edge(src, tgt, properties=props)

        self._validate_graph()
        self._assign_node_types()
        logger.info("Граф успешно загружен и инициализирован.")
        self.visualize_pipeline_graph()
        return self.graph

    def _validate_graph(self):
        iso = list(nx.isolates(self.graph))
        if iso:
            logger.warning(f"Изолированные узлы: {iso}")
        if not nx.is_weakly_connected(self.graph):
            logger.warning("Граф не слабо-связный. Возможны несвязные компоненты.")

    def _assign_node_types(self):
        """
        Если узел остался UNKNOWN, пытаемся определить по числу входящих/исходящих рёбер.
        """
        for n in self.graph.nodes():
            ctype = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            if ctype in [NodeType.INPUT, NodeType.OUTPUT]:
                continue
            in_deg = self.graph.in_degree(n)
            out_deg = self.graph.out_degree(n)
            logger.info(f"Узел {n}: in_deg={in_deg}, out_deg={out_deg}")
            if in_deg == 0 and out_deg > 0:
                new_type = NodeType.INPUT
            elif out_deg == 0 and in_deg > 0:
                new_type = NodeType.OUTPUT
            elif in_deg == 1 and out_deg == 1:
                new_type = NodeType.CONSUMER
            elif out_deg > in_deg:
                new_type = NodeType.BRANCH
            elif in_deg > out_deg:
                new_type = NodeType.DIVIDER
            else:
                new_type = NodeType.UNKNOWN
            self.graph.nodes[n]["type"] = new_type
            logger.info(f"Узел {n} => {new_type.name}")

    def get_nodes(self) -> List[Node]:
        result = []
        for n, attrs in self.graph.nodes(data=True):
            node_obj = Node(
                id=n,
                type=attrs.get("type", NodeType.UNKNOWN),
                data=attrs.get("data", {}),
                x=attrs.get("x", 0.0),
                y=attrs.get("y", 0.0)
            )
            result.append(node_obj)
        return result

    def get_pipes(self) -> List[Pipe]:
        result = []
        for u, v, edata in self.graph.edges(data=True):
            pipe_obj = Pipe(
                source=u,
                target=v,
                properties=edata.get("properties", {})
            )
            result.append(pipe_obj)
        return result

    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        pos = {n: (attr.get("x", 0.0), attr.get("y", 0.0)) for n, attr in self.graph.nodes(data=True)}
        node_colors = []
        for n, attr in self.graph.nodes(data=True):
            tp = attr.get("type", NodeType.UNKNOWN)
            node_colors.append(tp.color)

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=600)
        nx.draw_networkx_edges(self.graph, pos, arrowstyle="->", arrowsize=15)
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_color="white")

        edge_labels = {}
        for (u, v, eprops) in self.graph.edges(data=True):
            pr = eprops.get("properties", {})
            L = pr.get("length", None)
            if L is not None:
                edge_labels[(u, v)] = str(L)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        unique_types = set(attr.get("type", NodeType.UNKNOWN) for attr in self.graph.nodes.values())
        patches = [mpatches.Patch(color=t.color, label=t.title) for t in unique_types if t]
        plt.legend(handles=patches, loc="best")
        plt.title("Pipeline Graph Visualization")
        plt.axis("off")
        plt.show()
