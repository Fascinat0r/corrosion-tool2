# graph_adapter_nx.py

import logging
from itertools import product
from typing import List, Dict, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from new.graph_adapter_interface import IGraphAdapter
from new.models import PipelineNode, PipelineEdge, NodeType

logger = logging.getLogger("pipeline_logger")


class GraphAdapterNX(IGraphAdapter):
    """
    Реализация IGraphAdapter на основе NetworkX.
    Здесь хранится неориентированный граф для большинства операций,
    а для ориентации и расчётов создаётся DiGraph.
    """

    def __init__(self):
        self.digraph = nx.DiGraph()
        self._contraction_map: Dict[Tuple[str, str], PipelineNode] = {}

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

    def update_node(self, node: PipelineNode) -> None:
        if self.digraph.has_node(node.id):
            self.digraph.nodes[node.id].update(
                type=node.type,
                x=node.x,
                y=node.y,
                data=node.data
            )
        else:
            raise ValueError(f"Узел {node.id} не найден.")

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

    def all_simple_paths(self, start: str, end: str, cutoff: int = None) -> List[List[str]]:
        """
        Возвращает все простые пути в ориентированном графе (без повторений узлов).
        networkx.all_simple_paths уже учитывает направление.
        """
        return list(nx.all_simple_paths(self.digraph, start, end, cutoff=cutoff))

    def to_undirected_copy(self):
        """Вернуть неориентированную копию."""
        return self.digraph.to_undirected()

    def in_degree(self, node_id: str) -> int:
        return self.digraph.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        return self.digraph.out_degree(node_id)

    def simplify_graph(self, input_nodes: List[str], output_nodes: List[str]) -> Dict[Tuple[str, str], PipelineNode]:
        """
        Схлопывает узлы со степенью 2 (если они не в input/ output). Если обнаружен цикл длины 3 – генерируется исключение.
        Возвращает contraction_map.
        """
        contraction_map: Dict[Tuple[str, str], PipelineNode] = {}
        G = self.digraph
        changed = True
        while changed:
            changed = False
            for node in list(G.nodes()):
                if node in input_nodes or node in output_nodes:
                    continue
                if G.degree(node) == 2:
                    neighbors = set(G.neighbors(node))
                    if len(neighbors) < 2:
                        continue
                    u, w = tuple(neighbors)[:2]
                    if G.has_edge(u, w):
                        logger.debug(f"Пропускаем схлопывание {u} -> {node} -> {w}: ребро {u}-{w} уже существует.")
                        continue
                    node_data = G.nodes[node].copy()
                    G.remove_node(node)
                    key = (u, w)
                    contraction_map[key] = PipelineNode(
                        id=node,
                        type=node_data.get("type", NodeType.UNKNOWN),
                        x=node_data.get("x", 0.0),
                        y=node_data.get("y", 0.0),
                        data=node_data.get("data", {})
                    )
                    G.add_edge(u, w, properties={"contracted": [node]})
                    logger.debug(f"Схлопнули узел {node} между {u} и {w}.")
                    changed = True
        logger.info(f"Схлопывание завершено. Узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}.")
        return contraction_map

    def _validate_graph(self):
        iso = list(nx.isolates(self.digraph))
        if iso:
            logger.warning(f"Изолированные узлы: {iso}")
        if not nx.is_weakly_connected(self.digraph):
            logger.warning("Граф не слабо-связный. Возможны несвязные компоненты.")

    def expand_contraction_map(self, contraction_map: Dict[Tuple[str, str], PipelineNode]) -> None:
        """
        Восстанавливает схлопнутые узлы в графе: для каждого ребра (u, w) из contraction_map,
        удаляет ребро и вместо него добавляет цепочку: u -> v -> w.
        """
        for (u, w), node in contraction_map.items():
            if self.digraph.has_edge(u, w):
                self.digraph.remove_edge(u, w)
                self.digraph.add_node(node.id, type=node.type, x=node.x, y=node.y, data=node.data)
                self.digraph.add_edge(u, node.id)
                self.digraph.add_edge(node.id, w)
            elif self.digraph.has_edge(w, u):
                self.digraph.remove_edge(w, u)
                self.digraph.add_node(node.id, type=node.type, x=node.x, y=node.y, data=node.data)
                self.digraph.add_edge(w, node.id)
                self.digraph.add_edge(node.id, u)
            else:
                raise ValueError(f"Не удалось восстановить узел {node.id}.")

        logger.info("Схлопнутые узлы восстановлены.")

    def force_direction_for_io_edges(self, input_nodes: List[str], output_nodes: List[str]) -> List[Tuple[str, str]]:
        """
        Фиксирует направления для рёбер, связанных с входными и выходными узлами:
         - Для каждого входного узла все рёбра должны выходить из него (u->v).
         - Для каждого выходного узла все рёбра должны входить в него (u->v).
        Возвращает список зафиксированных рёбер.
        """
        fixed_edges = []
        for n in self.digraph.nodes():
            if n in input_nodes:
                # Все рёбра, исходящие из входного узла, фиксированы
                for neighbor in list(self.digraph.neighbors(n)):
                    fixed_edges.append((n, neighbor))
                    logger.debug(f"Зафиксировано ребро {n}->{neighbor} (входной узел).")
            elif n in output_nodes:
                # Все рёбра, входящие в выходной узел, фиксированы – значит, направление должно быть from neighbor -> n
                for neighbor in list(self.digraph.neighbors(n)):
                    fixed_edges.append((neighbor, n))
                    logger.debug(f"Зафиксировано ребро {neighbor}->{n} (выходной узел).")
        return fixed_edges

    def enumerate_orientations(self, input_nodes: List[str], output_nodes: List[str],
                               fixed_edges: List[Tuple[str, str]]) -> List['GraphAdapterNX']:
        """
        Перебирает все варианты ориентации для свободных рёбер (те, что не зафиксированы).
        Создает для каждого варианта новый адаптер.
        """
        # Получаем все рёбра неориентированного графа
        edges_list = list(self.digraph.to_undirected().edges())
        # Убираем ребра, уже зафиксированные
        for edge in fixed_edges:
            if edge in edges_list:
                edges_list.remove(edge)
            elif (edge[1], edge[0]) in edges_list:
                edges_list.remove((edge[1], edge[0]))
        free_indices = list(range(len(edges_list)))
        total_variants = 2 ** len(free_indices)
        logger.info(f"Перебор {total_variants} вариантов ориентации свободных рёбер...")
        result_adapters = []
        for assignment in product([0, 1], repeat=len(free_indices)):
            H = nx.DiGraph()
            for n, d in self.digraph.nodes(data=True):
                H.add_node(n, **d)
            # Добавляем зафиксированные рёбра
            for (u, v) in fixed_edges:
                if not H.has_edge(u, v):
                    H.add_edge(u, v, **self.digraph[u][v])
            # Добавляем свободные ребра согласно назначению: 0 => u->v, 1 => v->u
            for i, (u, v) in enumerate(edges_list):
                if assignment[i] == 0:
                    H.add_edge(u, v, **self.digraph[u][v])
                else:
                    H.add_edge(v, u, **self.digraph[u][v])
            tmp_adapter = GraphAdapterNX()
            tmp_adapter.digraph = H
            tmp_adapter._digraph_initialized = True
            if tmp_adapter._check_valid_orientation(input_nodes, output_nodes):
                result_adapters.append(tmp_adapter)
        logger.info(f"Найдено {len(result_adapters)} допустимых вариантов ориентации.")
        return result_adapters

    def _check_valid_orientation(self, input_nodes: List[str], output_nodes: List[str]) -> bool:
        """
        Условие:
          - для входных узлов in_degree=0, out_degree>0
          - для выходных узлов out_degree=0, in_degree>0
          - для остальных узлов in_degree>=1, out_degree>=1
        """
        for n in self.digraph.nodes():
            indeg = self.digraph.in_degree(n)
            outdeg = self.digraph.out_degree(n)
            if n in input_nodes:
                if indeg != 0 or outdeg == 0:
                    return False
            elif n in output_nodes:
                if outdeg != 0 or indeg == 0:
                    return False
            else:
                if indeg < 1 or outdeg < 1:
                    return False
        return True

    def visualize(self, title: str = "Pipeline graph") -> plt.Figure:
        """
        Создает и возвращает объект Figure с визуализацией графа.
        """
        G = self.digraph
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = {n: (d.get("x", 0.0), d.get("y", 0.0)) for n, d in G.nodes(data=True)}
        node_colors = [d.get("type", NodeType.UNKNOWN).color for n, d in G.nodes(data=True)]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)

        edge_labels = {}
        for u, v, edata in G.edges(data=True):
            pr = edata.get("properties", {})
            Q = pr.get("Q", None)
            L = round(Q, 3) if Q is not None else None
            if L is not None:
                edge_labels[(u, v)] = str(L)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        unique = {d.get("type", NodeType.UNKNOWN) for d in G.nodes.values()}
        patches = [mpatches.Patch(color=t.color, label=t.title) for t in unique if t]
        ax.legend(handles=patches, loc="best")
        ax.set_title(title)
        ax.axis("off")

        return fig
