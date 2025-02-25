# graph_adapter_nx.py
import logging
from typing import List, Tuple

import networkx as nx

from models import PipelineNode, PipelineEdge, NodeType

logger = logging.getLogger("p_logger")


class GraphAdapterNX:
    def __init__(self):
        # храним ориентированный граф
        self.digraph = nx.DiGraph()

    def add_node(self, node: PipelineNode) -> None:
        self.digraph.add_node(
            node.id,
            type=node.type,
            x=node.x,
            y=node.y,
            data=node.data
        )

    def add_edge(self, edge: PipelineEdge) -> None:
        self.digraph.add_edge(
            edge.source,
            edge.target,
            length=edge.length,
            diameter=edge.diameter,
            roughness=edge.roughness,
            properties=edge.properties
        )

    def remove_node(self, node_id: str) -> None:
        if self.digraph.has_node(node_id):
            self.digraph.remove_node(node_id)

    def remove_edge(self, source: str, target: str) -> None:
        if self.digraph.has_edge(source, target):
            self.digraph.remove_edge(source, target)

    def get_nodes(self) -> List[PipelineNode]:
        result = []
        for n, data in self.digraph.nodes(data=True):
            result.append(PipelineNode(
                id=n,
                type=data.get("type", NodeType.UNKNOWN),
                x=data.get("x", 0.0),
                y=data.get("y", 0.0),
                data=data.get("data", {})
            ))
        return result

    def get_edges(self) -> List[PipelineEdge]:
        result = []
        for u, v, edata in self.digraph.edges(data=True):
            result.append(PipelineEdge(
                source=u,
                target=v,
                length=edata.get("length", 0.0),
                diameter=edata.get("diameter", 0.5),
                roughness=edata.get("roughness", 0.000015),
                properties=edata.get("properties", {})
            ))
        return result

    def in_degree(self, node_id: str) -> int:
        return self.digraph.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        return self.digraph.out_degree(node_id)

    def all_simple_paths(self, start: str, end: str, cutoff: int = None) -> List[List[str]]:
        return list(nx.all_simple_paths(self.digraph, start, end, cutoff))

    def to_undirected_copy(self):
        return self.digraph.to_undirected()

    def force_direction_for_io_edges(self, input_nodes: List[str], output_nodes: List[str]) -> List[Tuple[str, str]]:
        """
        Пример: для узлов input_nodes делаем все исходящие, для output_nodes все входящие.
        Возвращаем список зафиксированных рёбер, чтобы не менять их ориентацию.
        """
        fixed_edges = []
        for node in self.digraph.nodes():
            if node in input_nodes:
                for succ in list(self.digraph.neighbors(node)):
                    fixed_edges.append((node, succ))
            if node in output_nodes:
                # все ребра должны идти к node
                for pred in list(self.digraph.predecessors(node)):
                    fixed_edges.append((pred, node))
        return fixed_edges
