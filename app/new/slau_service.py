# slau_service.py
import json
import logging
from typing import List, Tuple

import networkx as nx
import numpy as np

from models import NodeType, PipelineNode, PipelineEdge
from visualize import visualize_graph

logger = logging.getLogger("p_logger")


class SLASolverService:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.input_nodes = []
        self.output_nodes = []
        self.edge_list = []
        # новые поля для стагнации
        self.last_residuals = []

        # коэффициент under-relaxation
        self.alpha_R = 0.5  # например 0.5
        self.alpha_dpfix = 0.5

    def init(self, path):

        self.load_from_json(path)

        self.edge_list = list(self.graph.edges())
        # Инициализация свойств на рёбрах
        for e in self.edge_list:
            props = self.graph.edges[e].setdefault("properties", {})
            props.setdefault("Q", 0.0)
            props.setdefault("Q_old", 0.0)
            props.setdefault("dP", 0.0)
            props.setdefault("dP_old", 0.0)
            props.setdefault("R", 1e-3)
            props.setdefault("dPfix", 0.0)

        for n, dat in self.graph.nodes(data=True):
            t = dat.get("type", NodeType.UNKNOWN)
            if t == NodeType.INPUT:
                self.input_nodes.append(n)
            elif t == NodeType.OUTPUT:
                self.output_nodes.append(n)

        self.define_node_types()

    def add_node(self, node: PipelineNode) -> None:
        self.graph.add_node(
            node.id,
            type=node.type,
            x=node.x,
            y=node.y,
            data=node.data
        )

    def add_edge(self, edge: PipelineEdge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            length=edge.length,
            diameter=edge.diameter,
            roughness=edge.roughness,
            properties=edge.properties
        )

    def load_from_json(self, file_path: str) -> None:
        logger.info(f"Читаем данные из {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.input_nodes = data.get("input_nodes", [])
        self.output_nodes = data.get("output_nodes", [])

        sum_supply = 0.0
        sum_demand = 0.0

        for ninfo in data.get("nodes", []):
            node_id = ninfo["id"]
            node_data = ninfo.get("data", {})
            x = float(node_data.get("x", 0.0))
            y = float(node_data.get("y", 0.0))

            # Считаем, сколько supply/demand
            s = float(node_data.get("supply", 0.0))
            d = float(node_data.get("demand", 0.0))
            sum_supply += s
            sum_demand += d

            node = PipelineNode(id=node_id, type=NodeType.UNKNOWN, x=x, y=y, data=node_data)
            self.add_node(node)

        # Предупреждение, если supply < demand
        if sum_supply < sum_demand - 1e-9:
            logger.warning(f"Обнаружено, что суммарный supply={sum_supply:.3f} < demand={sum_demand:.3f}. "
                           "Система может быть неразрешимой или даст большие невязки.")

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
            self.add_edge(edge)

        logger.info("JSON загружен.")

    def define_node_types(self) -> None:
        """
        Определяем тип узлов по in_degree/out_degree:
          - INPUT:   в списке input_nodes
          - OUTPUT:  в списке output_nodes
          - MERGE:   in>1,  out=1
          - BRANCH:  in=1,  out>1
          - DIVIDER: in>1,  out>1
          - CONSUMER: in==1 и out==1
          - Иначе UNKNOWN
        """
        for n in self.graph.nodes():
            dat = self.graph.nodes[n]
            indeg = self.graph.in_degree(n)
            outdeg = self.graph.out_degree(n)

            # Если узел явно указан как входной/выходной:
            if n in self.input_nodes:
                new_type = NodeType.INPUT
            elif n in self.output_nodes:
                new_type = NodeType.OUTPUT
            else:
                if indeg > 1 and outdeg == 1:
                    new_type = NodeType.MERGE
                elif indeg == 1 and outdeg > 1:
                    new_type = NodeType.BRANCH
                elif indeg > 1 and outdeg > 1:
                    new_type = NodeType.DIVIDER
                elif indeg == 1 and outdeg == 1:
                    new_type = NodeType.CONSUMER
                else:
                    new_type = NodeType.UNKNOWN

            dat["type"] = new_type
            logger.debug(f"Узел {n} обновлён: новый тип {new_type.title}")

        logger.info("Типы узлов определены:")
        for n in self.graph.nodes():
            node_type = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            logger.info(f"  {n}: {node_type.title}")

    def solve_iteratively(self,
                          max_iter=20,
                          tolerance=1e-6,
                          tolerance_resid=1e-3,
                          visualize_each_iter: bool = True,
                          edge_label_for_plot: str = "dP"):

        self._compute_dP_physics()

        it = 0

        while it <= max_iter:
            it += 1

            logger.info(f"=== Итерация {it} ===")
            self._compute_R_and_dPfix_underrelaxed()  # используем метод с under-relaxation
            A, b = self._build_system()

            self._print_system(A, b)

            Q_vector, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
            if len(residual) > 0:
                resid_norm = np.sqrt(residual[0])
            else:
                Ax_b = A @ Q_vector - b
                resid_norm = np.linalg.norm(Ax_b)

            logger.debug(f"residual={residual}, rank={rank}, resid_norm={resid_norm:.4g}")

            # Применяем решение
            for i, e in enumerate(self.edge_list):
                props = self.graph.edges[e]["properties"]
                oldQ = props["Q"]
                newQ = Q_vector[i]
                props["Q_old"] = oldQ
                props["Q"] = newQ

            # пересчитать dP
            self._compute_dP_physics()

            # проверяем max_diff
            max_diff = max(
                abs(self.graph.edges[e]["properties"]["Q"] - self.graph.edges[e]["properties"]["Q_old"])
                for e in self.edge_list
            )

            logger.info(f"Макс. изменение расхода: {max_diff:e}")
            logger.info(f"Невязка решения (resid_norm): {resid_norm:.4f}")

            # 1) Автоматический «разворот» рёбер
            reoriented = self._auto_reorient_edges(eps=1e-7)
            if reoriented:
                # при развороте нужно пересобрать self.edge_list и т.д.
                self.edge_list = list(self.graph.edges())
                logger.info(f"Было развёрнуто {reoriented} рёбер, пересобираем систему и начинаем итерацию заново.")
                #
                # it = 0
                # self.last_residuals.clear()
                self.define_node_types()
                # Визуализация
                # if visualize_each_iter:
                #     visualize_graph(
                #         self.graph,
                #         title=f"Iteration {it}",
                #         edge_label=edge_label_for_plot,
                #         show=True,
                #     )
                # continue

            # 2) Проверка стагнации
            self.last_residuals.append(resid_norm)
            if len(self.last_residuals) > 5:
                # смотрим разницу между 5 шагов назад
                old_r = self.last_residuals[-6]
                if old_r - resid_norm < 0.01 * old_r:
                    logger.warning("Невязка не улучшается более чем на 1% за последние 5 итераций. Останавливаемся.")
                    break

            # 3) Критерий выхода
            if max_diff < tolerance and resid_norm < tolerance_resid:
                logger.info("Расходы и невязка сошлись, завершаем итерации.")
                break

            # 4) Визуализация
            if visualize_each_iter:
                visualize_graph(
                    self.graph,
                    title=f"Iteration {it}",
                    edge_label=edge_label_for_plot,
                    show=True,
                )

        # итог
        Q_final = [self.graph.edges[e]["properties"]["Q"] for e in self.edge_list]
        return Q_final

    def _build_system(self):
        """
        Аналог _build_system, но теперь учитываем demand.
        """
        nodes = list(self.graph.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        n_edges = len(self.edge_list)
        n_nodes = len(nodes)

        A_base = np.zeros((n_nodes, n_edges))
        b_base = np.zeros(n_nodes)

        for n in nodes:
            i = node_idx[n]
            data = self.graph.nodes[n].get("data", {})
            supply = float(data.get("supply", 0.0))
            demand = float(data.get("demand", 0.0))

            # mass-balance
            # (sum_in - sum_out) = demand - supply
            b_base[i] = demand - supply

            # заполнение коэффициентов
            for pred in self.graph.predecessors(n):
                if (pred, n) in self.edge_list:
                    j = self.edge_list.index((pred, n))
                    A_base[i, j] += 1.0
            for succ in self.graph.successors(n):
                if (n, succ) in self.edge_list:
                    j = self.edge_list.index((n, succ))
                    A_base[i, j] -= 1.0

        # строим extra eqn для MERGE/DIVIDER
        extra_rows = []
        extra_b = []
        for n in nodes:
            tp = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            if tp == NodeType.MERGE:
                chain_list = self._collect_chains_for_merge(n)
                if len(chain_list) > 1:
                    base_chain = chain_list[0]
                    for other_chain in chain_list[1:]:
                        row, b_val = self._form_chain_equation(base_chain, other_chain)
                        extra_rows.append(row)
                        extra_b.append(b_val)
            elif tp == NodeType.DIVIDER:
                chain_list = self._collect_chains_for_divider(n)
                if len(chain_list) > 1:
                    base_chain = chain_list[0]
                    for other_chain in chain_list[1:]:
                        row, b_val = self._form_chain_equation(base_chain, other_chain)
                        extra_rows.append(row)
                        extra_b.append(b_val)

        if extra_rows:
            A_extra = np.array(extra_rows)
            b_extra = np.array(extra_b)
            A_ext = np.vstack((A_base, A_extra))
            b_ext = np.concatenate((b_base, b_extra))
            return A_ext, b_ext
        else:
            return A_base, b_base

    def _compute_R_and_dPfix_underrelaxed(self):
        """
        Аналог _compute_R_and_dPfix, но используем under-relaxation:
          R_new = alpha_R * R_candidate + (1-alpha_R)* R_old
          dPfix_new = alpha_dpfix*dPfix_candidate + (1-alpha_dpfix)* old_dPfix
        """
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q_old = props["Q_old"]
            Q_cur = props["Q"]
            dP_old = props["dP_old"]
            dP_cur = props["dP"]

            R_prev = props["R"]
            fix_prev = props["dPfix"]

            # Вычисляем "сырые" R_candidate, fix_candidate
            if abs(Q_cur - Q_old) > 1e-12:
                R_cand = (dP_cur - dP_old) / (Q_cur - Q_old)
                if R_cand < 1e-12:
                    R_cand = 1e-12
                fix_cand = dP_cur - R_cand * Q_cur
            else:
                if abs(Q_cur) > 1e-12:
                    r_val = dP_cur / Q_cur
                    if r_val < 1e-12:
                        r_val = 1e-12
                    R_cand = r_val
                    fix_cand = 0.0
                else:
                    R_cand = 1e5
                    fix_cand = 0.0

            # under-relaxation
            R_new = self.alpha_R * R_cand + (1 - self.alpha_R) * R_prev
            dpfix_new = self.alpha_dpfix * fix_cand + (1 - self.alpha_dpfix) * fix_prev

            props["R"] = R_new
            props["dPfix"] = dpfix_new

    def _auto_reorient_edges(self, eps=1e-7):
        """
        Проверяем все рёбра, если Q < -eps, разворачиваем ребро.
        Возвращаем число рёбер, которые были развёрнуты.
        """
        cnt = 0

        for u, v in list(self.edge_list):  # Копируем список, чтобы менять граф во время итерации
            props = self.graph.edges[u, v]["properties"]
            if props["Q"] < -eps:
                self.graph.remove_edge(u, v)
                self.graph.add_edge(v, u, properties=props)  # Копируем все атрибуты
                self.graph.edges[v, u]["properties"]["Q"] = abs(props["Q"])  # Исправляем знак Q
                self.graph.edges[v, u]["properties"]["Q_old"] = 0.0  # Или abs(props["Q"])
                cnt += 1

        if cnt:
            logger.info(f"Развернули {cnt} рёбер из-за Q < 0.")
        return cnt

    def _compute_dP_physics(self):
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q = props["Q"]
            length = self.graph.edges[e].get("length", 1.0)
            diam = self.graph.edges[e].get("diameter", 0.5)
            # простая модель
            k = length / (diam ** 4)
            props["dP_old"] = props["dP"]
            props["dP"] = k * Q * abs(Q)

    def _compute_R_and_dPfix(self):
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q_old = props["Q_old"]
            Q_cur = props["Q"]
            dP_old = props["dP_old"]
            dP_cur = props["dP"]

            if abs(Q_cur - Q_old) > 1e-12:
                R_candidate = (dP_cur - dP_old) / (Q_cur - Q_old)
                if R_candidate < 1e-12:
                    R_candidate = 1e-12
                props["R"] = R_candidate
                props["dPfix"] = dP_cur - R_candidate * Q_cur
            else:
                if abs(Q_cur) > 1e-12:
                    r_val = dP_cur / Q_cur
                    if r_val < 1e-12:
                        r_val = 1e-12
                    props["R"] = r_val
                    props["dPfix"] = 0.0
                else:
                    props["R"] = 1e5
                    props["dPfix"] = 0.0

    def _collect_chains_for_merge(self, node: str) -> List[List[Tuple[str, str]]]:
        """
        Для узла MERGE ищем пути от всех INPUT-узлов к этому узлу.
        Возвращаем список цепочек, где каждая цепочка - список (u->v) рёбер.
        """
        chains = []
        for inp in self.input_nodes:
            if inp == node:
                continue
            # Найдём все пути inp->...->node
            all_paths = list(nx.all_simple_paths(self.graph, inp, node))
            for path in all_paths:
                if len(path) < 2:
                    continue
                # Преобразуем список узлов в список рёбер
                edges = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edges.append((u, v))
                chains.append(edges)
                logger.debug(f"MERGE node {node}: found chain from {inp}: {edges}")
        return chains

    def _collect_chains_for_divider(self, node: str) -> List[List[Tuple[str, str]]]:
        """
        Для узла DIVIDER ищем пути от node к всем OUTPUT-узлам.
        Возвращаем список цепочек (список рёбер).
        """
        chains = []
        for outn in self.output_nodes:
            if outn == node:
                continue
            try:
                all_paths = list(nx.all_simple_paths(self.graph, node, outn))
                for path in all_paths:
                    if len(path) < 2:
                        continue
                    edges = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edges.append((u, v))
                    chains.append(edges)
                    logger.debug(f"DIVIDER node {node}: found chain to {outn}: {edges}")
            except nx.NetworkXNoPath:
                logger.debug(f"No path from {node} to {outn}")
                pass
        return chains

    def _form_chain_equation(self, chainA: List[Tuple[str, str]], chainB: List[Tuple[str, str]]):
        """
        Формирует вектор row, b_val для уравнения:
          sum_{e in chainA}(R_e * Q_e + dPfix_e) = sum_{e in chainB}(R_e * Q_e + dPfix_e)
        => (row[eA] += R_e, b += dPfix_e) vs (row[eB] -= R_e, b -= dPfix_e)
        """
        n_edges = len(self.edge_list)
        row = np.zeros(n_edges)
        b_val = 0.0

        # плюс для chainA
        for e in chainA:
            j = self.edge_list.index(e)
            Re = self.graph.edges[e]["properties"]["R"]
            dpf = self.graph.edges[e]["properties"]["dPfix"]
            row[j] += Re
            b_val += dpf

        # минус для chainB
        for e in chainB:
            j = self.edge_list.index(e)
            Re = self.graph.edges[e]["properties"]["R"]
            dpf = self.graph.edges[e]["properties"]["dPfix"]
            row[j] -= Re
            b_val -= dpf

        logger.debug(f"Extra eqn for chainA={chainA}, chainB={chainB}, row={row}, b={b_val}")
        return row, b_val

    def _print_system(self, A: np.ndarray, b: np.ndarray):
        n_nodes = len(self.graph.nodes())
        n_rows, n_cols = A.shape
        logger.info("----- Система уравнений (A, b) -----")
        header_cols = [f" e{i + 1}" for i in range(n_cols)]
        logger.info("           " + " ".join(header_cols) + " |  b ")
        for i in range(n_rows):
            row_vals = " ".join(f"{A[i, j]:7.2f}" for j in range(n_cols))
            if i < n_nodes:
                logger.info(f"Row{i + 1:2d}(node) : {row_vals} | {b[i]:7.2f}")
            else:
                logger.info(f"Row{i + 1:2d}(extra): {row_vals} | {b[i]:7.2f}")
        logger.info("----- Конец системы уравнений -----\n")
