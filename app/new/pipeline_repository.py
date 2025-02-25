# pipeline_repository.py
import json
import logging
from typing import List
from new.models import PipelineNode, PipelineEdge, NodeType
from new.graph_adapter_nx import GraphAdapterNX

logger = logging.getLogger("p_logger")


class PipelineRepository:
    def __init__(self, adapter: GraphAdapterNX, input_nodes: List[str] = None, output_nodes: List[str] = None):
        self.adapter = adapter
        self.input_nodes = input_nodes or []
        self.output_nodes = output_nodes or []

    def load_from_json(self, file_path: str) -> None:
        logger.info(f"Читаем данные из {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.input_nodes = data.get("input_nodes", [])
        self.output_nodes = data.get("output_nodes", [])

        for ninfo in data.get("nodes", []):
            node_id = ninfo["id"]
            node_data = ninfo.get("data", {})
            x = float(node_data.get("x", 0.0))
            y = float(node_data.get("y", 0.0))
            # Начально отметим тип как UNKNOWN, уточним позже
            node = PipelineNode(id=node_id, type=NodeType.UNKNOWN, x=x, y=y, data=node_data)
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

        logger.info("JSON загружен.")

    def define_node_types(self) -> None:
        """
        Определяем тип узлов по in_degree/out_degree:
          - INPUT:   in=0, out>=1
          - OUTPUT:  in>=1, out=0
          - MERGE:   in>1,  out=1
          - BRANCH:  in=1,  out>1
          - DIVIDER: in>1,  out>1
          - CONSUMER: (пример) in>=1, out=0 (если в JSON есть demand, но нужно?)
          - Иначе UNKNOWN
        """
        for node in self.adapter.get_nodes():
            indeg = self.adapter.in_degree(node.id)
            outdeg = self.adapter.out_degree(node.id)

            # Если узел явно находится в input_nodes:
            if node.id in self.input_nodes:
                node.type = NodeType.INPUT
            # Если узел явно находится в output_nodes:
            elif node.id in self.output_nodes:
                node.type = NodeType.OUTPUT
            else:
                # Автоматическая классификация
                if indeg == 0 and outdeg > 0:
                    node.type = NodeType.INPUT
                elif indeg > 0 and outdeg == 0:
                    node.type = NodeType.OUTPUT
                elif indeg > 1 and outdeg == 1:
                    node.type = NodeType.MERGE
                elif indeg == 1 and outdeg > 1:
                    node.type = NodeType.BRANCH
                elif indeg > 1 and outdeg > 1:
                    node.type = NodeType.DIVIDER
                elif indeg == 1 and outdeg == 1:
                    node.type = NodeType.CONSUMER
                else:
                    node.type = NodeType.UNKNOWN

            self.adapter.add_node(node)  # обновить в графе (или update_node)

        logger.info("Типы узлов определены.")
        for node in self.adapter.get_nodes():
            logger.info(f"  {node.id}: {node.type.title}")
