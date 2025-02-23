# graph_adapter_nx.py

import logging
from typing import List, Dict, Tuple

import networkx as nx

from graph_adapter_interface import IGraphAdapter
from models import PipelineNode, PipelineEdge, NodeType

logger = logging.getLogger("pipeline_logger")


class GraphAdapterNX(IGraphAdapter):
    """
    Реализация интерфейса IGraphAdapter с использованием networkx.
    """

    def __init__(self):
        self._graph = nx.Graph()
        self._digraph = nx.DiGraph()
        self._digraph_initialized = False

    def is_digraph(self) -> bool:
        return self._digraph_initialized

    def add_node(self, node: PipelineNode) -> None:
        self._graph.add_node(
            node.id,
            type=node.type,
            x=node.x,
            y=node.y,
            attributes=node.attributes
        )

    def add_edge(self, edge: PipelineEdge) -> None:
        self._graph.add_edge(
            edge.source,
            edge.target,
            length=edge.length,
            diameter=edge.diameter,
            roughness=edge.roughness,
            properties=edge.properties
        )

    def remove_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    def remove_edge(self, source: str, target: str) -> None:
        if self._graph.has_edge(source, target):
            self._graph.remove_edge(source, target)

    def get_nodes(self) -> List[PipelineNode]:
        result = []
        for n, data in self._graph.nodes(data=True):
            node_type = data.get("type", NodeType.UNKNOWN)
            x = data.get("x", 0.0)
            y = data.get("y", 0.0)
            attrs = data.get("attributes", {})
            result.append(PipelineNode(id=n, type=node_type, x=x, y=y, attributes=attrs))
        return result

    def get_edges(self) -> List[PipelineEdge]:
        result = []
        for u, v, edata in self._graph.edges(data=True):
            length = edata.get("length", 0.0)
            diameter = edata.get("diameter", 0.5)
            roughness = edata.get("roughness", 0.000015)
            props = edata.get("properties", {})
            result.append(PipelineEdge(
                source=u,
                target=v,
                length=length,
                diameter=diameter,
                roughness=roughness,
                properties=props
            ))
        return result

    def all_simple_paths(self, start: str, end: str, cutoff: int = None) -> List[List[str]]:
        """
        Возвращает все простые пути в ориентированном графе (без повторений узлов).
        networkx.all_simple_paths уже учитывает направление.
        """
        return list(nx.all_simple_paths(self._graph, start, end, cutoff=cutoff))

    def to_undirected_copy(self):
        """Вернуть неориентированную копию."""
        return self._graph.to_undirected()

    def in_degree(self, node_id: str) -> int:
        if not self.is_digraph():
            raise ValueError("Graph is not a DiGraph yet.")
        return self._digraph.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        if not self.is_digraph():
            raise ValueError("Graph is not a DiGraph yet.")
        return self._digraph.out_degree(node_id)

    def update_node(self, node: PipelineNode) -> None:
        if not self.is_digraph():
            raise ValueError("Graph is not a DiGraph yet.")
        if self._digraph.has_node(node.id):
            self._digraph.nodes[node.id].update(
                type=node.type,
                x=node.x,
                y=node.y,
                attributes=node.attributes
            )
        else:
            raise ValueError(f"Узел {node.id} не найден в графе.")

    def simplify_graph(self, input_nodes: List[str], output_nodes: List[str]) -> Dict[Tuple[str, str], PipelineNode]:
        """
        Схлопывает все узлы со степенью=2 (не входные, не выходные).
        """
        contraction_map: Dict[Tuple[str, str], PipelineNode] = {}  # {node_id: (u, w)} u и w - соседи схлопнутого узла
        G = self._graph
        changed = True
        while changed:
            changed = False
            nodes = list(G.nodes())
            for node in nodes:
                if node in input_nodes or node in output_nodes:
                    continue
                if G.degree(node) == 2:
                    neighbors = list(G.neighbors(node))
                    neighbors = set(neighbors) - {node}
                    if len(neighbors) < 2:
                        continue
                    nbs = list(neighbors)
                    u, w = nbs[0], nbs[1]
                    if G.has_edge(u, w):  # Если есть ребро u->w, то пропускаем
                        logger.debug(f"Пропускаем схлопывание {u} -> {node} -> {w}, ребро {u}->{w} уже есть.")
                        continue
                    # Удаляем узел
                    node_copy = G.nodes[node].copy()
                    G.remove_node(node)
                    # запомним схлопнутые узлы
                    key = u, w
                    contraction_map[key] = PipelineNode(id=node,
                                                        type=node_copy.get("type", NodeType.UNKNOWN),
                                                        x=node_copy.get("x", 0.0),
                                                        y=node_copy.get("y", 0.0),
                                                        attributes=node_copy.get("attributes", {}))

                    # добавим ребро u->w
                    G.add_edge(u, w, properties={"contracted": [node]})
                    logger.debug(f"Схлопнули узел {node} в ребро {u} -> {w}.")
                    changed = True
        logger.info(f"Схлопывание завершено. Убраны узлы: {contraction_map.keys()}")
        return contraction_map

    def force_direction_for_io_edges(self, input_nodes: List[str], output_nodes: List[str]):
        fixed_edges = []
        G = self._graph
        for u in G.nodes():
            if u in input_nodes:
                # Все рёбра входного узла должны быть исходящими
                for v in G.neighbors(u):
                    if v in input_nodes:
                        raise ValueError(f"Найден цикл между двумя входными узлами: {u} - {v}")
                    fixed_edges.append((u, v))
            elif u in output_nodes:
                # Все рёбра выходного узла должны быть входящими
                for v in G.neighbors(u):
                    if v in output_nodes:
                        raise ValueError(f"Найден цикл между двумя выходными узлами: {u} - {v}")
                    fixed_edges.append((v, u))
        return fixed_edges

    def _check_valid_orientation(self, input_nodes: List[str], output_nodes: List[str]) -> bool:
        """
        Условие:
          - для входных узлов in_degree=0, out_degree>0
          - для выходных узлов out_degree=0, in_degree>0
          - для остальных узлов in_degree>=1, out_degree>=1
        """
        G = self._graph
        # Копируем неориентированный граф в ориентированный
        self._digraph = G.to_directed()
        for n in G.nodes():
            indeg = G.in_degree(n)
            outdeg = G.out_degree(n)
            if n in input_nodes:
                if indeg != 0 or outdeg == 0:
                    return False
            elif n in output_nodes:
                if outdeg != 0 or indeg == 0:
                    return False
            else:
                if indeg < 1 or outdeg < 1:
                    return False
        self._digraph_initialized = True
        return True

    def enumerate_orientations(self, input_nodes: List[str], output_nodes: List[str], fixed_edges) -> List[
        'GraphAdapterNX']:
        """
        Перебираем все варианты ориентации для **нефиксированных** рёбер.
        """

        edges_list = list(self._graph.to_undirected().edges())
        # Уберём зафиксированные
        for u, v in fixed_edges:
            if (u, v) in edges_list:
                edges_list.remove((u, v))
            elif (v, u) in edges_list:
                edges_list.remove((v, u))
            else:
                raise ValueError(f"Невозможно найти ребро {u} - {v} в списке рёбер для ориентации.")
        free_indices = list(range(len(edges_list)))

        from itertools import product

        results = []
        total_vars = 2 ** len(free_indices)
        logger.info(f"Перебор {total_vars} вариантов ориентации...")

        for assignment in product([0, 1], repeat=len(free_indices)):
            # Создадим копию:
            H_copy = nx.DiGraph()
            # скопируем узлы
            for n, d in self._graph.nodes(data=True):
                H_copy.add_node(n, **d)
            # расставим рёбра
            # 1) зафиксированные
            for u, v in fixed_edges:
                if not H_copy.has_edge(u, v):
                    H_copy.add_edge(u, v, **self._graph[u][v])
            # 2) свободные
            for i, (u, v) in enumerate(edges_list):
                # 0 => u->v, 1 => v->u
                if assignment[i] == 0:
                    # у->v
                    if not H_copy.has_edge(u, v):
                        H_copy.add_edge(u, v, **self._graph[u][v])
                else:
                    if not H_copy.has_edge(v, u):
                        H_copy.add_edge(v, u, **self._graph[u][v])

            # проверим
            tmp_adapter = GraphAdapterNX()
            tmp_adapter._graph = H_copy
            if tmp_adapter._check_valid_orientation(input_nodes, output_nodes):
                results.append(tmp_adapter)

        logger.info(f"Найдено {len(results)} валидных вариантов.")
        return results

    def visualize(self, title: str = "Pipeline Graph Visualization"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        G = self._graph
        if self.is_digraph():
            G = self._digraph

        pos = {n: (attr.get("x", 0.0), attr.get("y", 0.0)) for n, attr in G.nodes(data=True)}
        node_colors = []
        for n, attr in G.nodes(data=True):
            tp = attr.get("type", NodeType.UNKNOWN)
            node_colors.append(tp.color)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=9, font_color="white")

        edge_labels = {}
        for (u, v, eprops) in G.edges(data=True):
            pr = eprops.get("properties", {})
            L = pr.get("length", None)
            if L is not None:
                edge_labels[(u, v)] = str(L)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        unique_types = set(attr.get("type", NodeType.UNKNOWN) for attr in G.nodes.values())
        patches = [mpatches.Patch(color=t.color, label=t.title) for t in unique_types if t]
        plt.legend(handles=patches, loc="best")
        plt.title(title)
        plt.axis("off")
        plt.show()

    def expand_contraction_map(self, contraction_map: Dict[Tuple[str, str], PipelineNode]) -> None:
        """
        Расширить схлопнутые узлы в ориентированном графе.
        Направление ребра посередине которого был схлопнут узел копируется на восстановленные рёбра.
        """
        if not self.is_digraph():
            raise ValueError("Graph is not a DiGraph yet.")
        for neighbors, node in contraction_map.items():
            u, w = neighbors
            if self._digraph.has_edge(u, w):
                # удалим ребро u->w
                self._digraph.remove_edge(u, w)
                # Восстановим схлопнутый узел
                self._digraph.add_node(node.id, type=node.type, x=node.x, y=node.y, attributes=node.attributes)
                # Восстановим рёбра u -> node, node -> w
                self._digraph.add_edge(u, node.id)
                self._digraph.add_edge(node.id, w)
            elif self._digraph.has_edge(w, u):
                # удалим ребро w->u
                self._digraph.remove_edge(w, u)
                # Восстановим схлопнутый узел
                self._digraph.add_node(node.id, type=node.type, x=node.x, y=node.y, attributes=node.attributes)
                # Восстановим рёбра w -> node, node -> u
                self._digraph.add_edge(w, node.id)
                self._digraph.add_edge(node.id, u)
            else:
                raise ValueError(f"Невозможно восстановить схлопнутый узел {node.id}.")
        self._graph = self._digraph.to_undirected()
        logger.info("Схлопнутые узлы восстановлены.")

    def clear_chains(self):
        """
        Удаляет атрибут "chain" из свойств всех рёбер.
        """
        for u, v in self._digraph.edges():
            props = self._digraph.edges[u, v].setdefault("properties", {})
            props["chain"] = ""
        logger.info("Все chain-метки сброшены.")

    def build_divider_chains(self, max_depth: int = 10):
        """
        Для каждого узла типа DIVIDER:
          - Находит все простые пути (с учетом направления) от INPUT-узлов к нему и от него к OUTPUT-узлам,
            без повторения узлов (циклов).
          - Каждому найденному пути присваивает уникальное имя (chain_inX или chain_outY).
          - Помечает все рёбра на найденном пути этой меткой.
          - Логирует цепочки.
        """

        # Получаем списки INPUT, OUTPUT и DIVIDER узлов
        input_nodes = [n for n, d in self._digraph.nodes(data=True) if d.get("type") == NodeType.INPUT]
        output_nodes = [n for n, d in self._digraph.nodes(data=True) if d.get("type") == NodeType.OUTPUT]
        dividers = [n for n, d in self._digraph.nodes(data=True) if d.get("type") == NodeType.DIVIDER]

        chain_in_counter = 1
        chain_out_counter = 1

        for d_node in dividers:
            logger.info(f"Обрабатываем узел-водораздел {d_node}")
            divider_chains = {"input_chains": [], "output_chains": []}

            # Поиск путей от INPUT к d_node (учитываем направление)
            for inp in input_nodes:
                try:
                    paths = list(nx.all_simple_paths(self._digraph, source=inp, target=d_node, cutoff=max_depth))
                except nx.NetworkXNoPath:
                    continue
                for path in paths:
                    cname = f"chain_in{chain_in_counter}"
                    chain_in_counter += 1
                    self._mark_path_edges(path, cname)
                    divider_chains["input_chains"].append((cname, path))

            # Поиск путей от d_node к OUTPUT
            for out in output_nodes:
                try:
                    paths = list(nx.all_simple_paths(self._digraph, source=d_node, target=out, cutoff=max_depth))
                except nx.NetworkXNoPath:
                    continue
                for path in paths:
                    cname = f"chain_out{chain_out_counter}"
                    chain_out_counter += 1
                    self._mark_path_edges(path, cname)
                    divider_chains["output_chains"].append((cname, path))

            logger.info(f"Сформированные цепочки для узла {d_node}:")
            if divider_chains["input_chains"]:
                logger.info("  Входные цепочки:")
                for cname, path in divider_chains["input_chains"]:
                    logger.info(f"    {cname}: {' -> '.join(path)}")
            if divider_chains["output_chains"]:
                logger.info("  Выходные цепочки:")
                for cname, path in divider_chains["output_chains"]:
                    logger.info(f"    {cname}: {' -> '.join(path)}")
        logger.info("Все цепочки для узлов-водоразделов успешно сформированы.")

    def _mark_path_edges(self, node_path: List[str], chain_name: str):
        """
        Добавляет chain_name к атрибуту "chain" для каждого ребра, входящего в путь.
        """
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            if self._digraph.has_edge(u, v):
                props = self._digraph.edges[u, v].setdefault("properties", {})
                old_chain = props.get("chain", "")
                existing = [c.strip() for c in old_chain.split(",") if c.strip()]
                if chain_name not in existing:
                    new_chain = old_chain + ("," if old_chain else "") + chain_name
                    props["chain"] = new_chain
