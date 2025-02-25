# slau_service.py
import logging
from typing import List, Tuple

import networkx as nx
import numpy as np

from new.models import NodeType
from new.visualize import visualize_graph

logger = logging.getLogger("p_logger")


class SLASolverService:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.edge_list = list(graph.edges())
        # Инициализация свойств на рёбрах
        for e in self.edge_list:
            props = self.graph.edges[e].setdefault("properties", {})
            props.setdefault("Q", 0.0)
            props.setdefault("Q_old", 0.0)
            props.setdefault("dP", 0.0)
            props.setdefault("dP_old", 0.0)
            props.setdefault("R", 1e-3)
            props.setdefault("dPfix", 0.0)

    def solve_iteratively(self,
                          max_iter=10,
                          tolerance=1e-6,
                          visualize_each_iter: bool = True,
                          edge_label_for_plot: str = "dP") -> List[float]:
        """
        Итерационный алгоритм:
         1) На основе текущих Q, пересчитать dP (физ. формула).
         2) Пересчитать R, dPfix.
         3) Построить матрицу A и вектор b, решить.
         4) Проверить невязку (например, разницу давлений в узлах MERGE / DIVIDER).
         5) Повторять до схода или max_iter.
        """
        self._compute_dP_physics()  # На 1-й итерации можно
        for it in range(max_iter):
            logger.info(f"=== Итерация {it + 1} ===")
            # 1. Обновляем R, dPfix
            self._compute_R_and_dPfix()
            # 2. Строим систему
            A, b = self._build_system()

            # Печатаем матрицу в консоль (по желанию)
            self._print_system(A, b)

            # 3. Решаем
            Q_vector, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
            logger.debug(f"residual={residual}, rank={rank}")

            # 4. Применяем решения к графу
            for i, e in enumerate(self.edge_list):
                props = self.graph.edges[e]["properties"]
                props["Q_old"] = props["Q"]
                props["Q"] = Q_vector[i]

            # 5. Пересчитываем dP
            self._compute_dP_physics()

            # 6. Можно проверить невязку. Для упрощения — только проверка изменения Q
            max_diff = max(abs(self.graph.edges[e]["properties"]["Q"] -
                               self.graph.edges[e]["properties"]["Q_old"])
                           for e in self.edge_list)
            logger.info(f"Макс. изменение расхода: {max_diff:e}")
            # 7. Визуализация на этой итерации
            if visualize_each_iter:
                visualize_graph(
                    self.graph,
                    title=f"Iteration {it + 1}",
                    edge_label=edge_label_for_plot,
                    path=f"graph_iter_{it + 1}.png",
                    show=True
                )
                logger.info(f"Сохранили рисунок: graph_iter_{it + 1}.png")

            # 8. Критерий останова
            if max_diff < tolerance:
                logger.info("Расходы сошлись, завершаем итерации")
                break

        return [self.graph.edges[e]["properties"]["Q"] for e in self.edge_list]

    def _compute_dP_physics(self):
        """
        Упрощённая формула dP = k * Q * |Q|.
        k может быть = length / (diam^4) или любая иная форма, имитирующая потери.
        """
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q = props["Q"]
            length = self.graph.edges[e].get("length", 1.0)
            diam = self.graph.edges[e].get("diameter", 0.5)
            k = length / (diam ** 4)
            props["dP_old"] = props["dP"]
            props["dP"] = k * Q * abs(Q)

    def _compute_R_and_dPfix(self):
        """
        Формулы для R, dPfix:
          если |Q - Q_old| > epsilon => R = (dP - dP_old)/(Q-Q_old), dPfix = dP - R*Q
          иначе R = dP/Q (если Q!=0), dPfix=0
        """
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
                    # Q=0, ставим большое R
                    props["R"] = 1e5
                    props["dPfix"] = 0.0

    def _build_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Формируем систему уравнений:
          - первые N узлов -> mass balance
          - добавляем extra eqs для MERGE / DIVIDER (равенство сумм RQ + dPfix)
        """
        nodes = list(self.graph.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        n_edges = len(self.edge_list)
        n_nodes = len(nodes)

        # БАЗОВАЯ ЧАСТЬ: mass balance
        A_base = np.zeros((n_nodes, n_edges))
        b_base = np.zeros(n_nodes)

        for n in nodes:
            i = node_idx[n]
            # определяем тип
            tp = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            data = self.graph.nodes[n].get("data", {})
            supply = data.get("supply", 0.0)
            demand = data.get("demand", 0.0)

            # В простом варианте:
            # INPUT -> b_base[i] = - supply
            # consumer -> b_base[i] = demand (или -demand)
            # Смотрим знак "приход - расход" = b
            # Для INPUT: +supply приходит, значит (in - out) = supply => out - in = -supply
            # Здесь договоримся: (sum in) - (sum out) = -supply + demand
            # (так как supply>0 => b<0 => расход пойдет "наружу" и т.п.)
            b_base[i] = -float(supply) + float(demand)

            # заполнение матрицы: входящие рёбра +1, исходящие -1
            for pred in self.graph.predecessors(n):
                if (pred, n) in self.edge_list:
                    j = self.edge_list.index((pred, n))
                    A_base[i, j] += 1.0
            for succ in self.graph.successors(n):
                if (n, succ) in self.edge_list:
                    j = self.edge_list.index((n, succ))
                    A_base[i, j] -= 1.0

        # ДОПОЛНИТЕЛЬНЫЕ УРАВНЕНИЯ для MERGE / DIVIDER
        # Идея: для узла, у которого in>1 (или out>1), ищем все сочетания входящих (или исходящих) ветвей,
        # говорим: sum(RQ + dPfix) по ветви1 = sum(...) по ветви2.
        # Упрощённо реализуем: если узел MERGE, сравниваем все входные ветви попарно.
        # Если узел DIVIDER, тоже.  Можно искать пути chain, но для простоты ограничимся "короткими цепочками":
        extra_rows = []
        extra_b = []

        for n in nodes:
            tp = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            if tp not in [NodeType.MERGE, NodeType.DIVIDER]:
                continue
            in_edges = [(u, v) for (u, v) in self.edge_list if v == n]
            out_edges = [(u, v) for (u, v) in self.edge_list if u == n]
            # Простейший случай: MERGE => in>1, out=1
            # Будем сравнивать in_1 с in_2, in_3..., (на самом деле n-1 уравнений)
            if tp == NodeType.MERGE and len(in_edges) > 1:
                # Выбираем за "базовую" in_edges[0], все остальные сравниваем
                base = in_edges[0]
                for other in in_edges[1:]:
                    row = np.zeros(n_edges)
                    b_val = 0.0
                    # base: +R, +dPfix
                    Rb = self.graph.edges[base]["properties"]["R"]
                    fixb = self.graph.edges[base]["properties"]["dPfix"]
                    jbase = self.edge_list.index(base)
                    row[jbase] += Rb
                    b_val += fixb
                    # other: -R, -dPfix
                    Ro = self.graph.edges[other]["properties"]["R"]
                    fox = self.graph.edges[other]["properties"]["dPfix"]
                    joth = self.edge_list.index(other)
                    row[joth] -= Ro
                    b_val -= fox
                    extra_rows.append(row)
                    extra_b.append(b_val)
            # DIVIDER => (in>1 && out>1) или (in=1 && out>1)?
            # Для учебного примера: если out>1, сравниваем out_1... out_k
            if tp == NodeType.DIVIDER:
                # Сравниваем все входы между собой + все выходы между собой
                # (или более сложный "цепочный" метод)
                # Для примера возьмем: если in>1, сравниваем in-edges попарно
                if len(in_edges) > 1:
                    base_in = in_edges[0]
                    for other in in_edges[1:]:
                        row = np.zeros(n_edges)
                        b_val = 0.0
                        Rb = self.graph.edges[base_in]["properties"]["R"]
                        fixb = self.graph.edges[base_in]["properties"]["dPfix"]
                        jbase = self.edge_list.index(base_in)
                        row[jbase] += Rb
                        b_val += fixb

                        Ro = self.graph.edges[other]["properties"]["R"]
                        fox = self.graph.edges[other]["properties"]["dPfix"]
                        joth = self.edge_list.index(other)
                        row[joth] -= Ro
                        b_val -= fox

                        extra_rows.append(row)
                        extra_b.append(b_val)

                # Аналогично для out-edges
                if len(out_edges) > 1:
                    base_out = out_edges[0]
                    for other in out_edges[1:]:
                        row = np.zeros(n_edges)
                        b_val = 0.0

                        Rb = self.graph.edges[base_out]["properties"]["R"]
                        fixb = self.graph.edges[base_out]["properties"]["dPfix"]
                        jbase = self.edge_list.index(base_out)
                        row[jbase] += Rb
                        b_val += fixb

                        Ro = self.graph.edges[other]["properties"]["R"]
                        fox = self.graph.edges[other]["properties"]["dPfix"]
                        joth = self.edge_list.index(other)
                        row[joth] -= Ro
                        b_val -= fox

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

    def _print_system(self, A: np.ndarray, b: np.ndarray):
        """
        Печать матрицы A и вектора b в консоль.
        Первая часть строк — соответствуют узлам,
        оставшиеся строки (если есть) — дополнительным уравнениям (MERGE/DIVIDER).
        """
        n_nodes = len(self.graph.nodes())
        n_rows, n_cols = A.shape
        logger.info("----- Система уравнений (A, b) -----")
        # Печатаем шапку для столбцов: edge1, edge2, ...
        header_cols = [f" e{i + 1}" for i in range(n_cols)]
        logger.info("           " + " ".join(header_cols) + " |  b ")

        # Основные уравнения (первая часть)
        for i in range(n_nodes):
            row_vals = " ".join(f"{A[i, j]:6.2f}" for j in range(n_cols))
            logger.info(f"Row{i + 1:2d}(node): {row_vals} | {b[i]:6.2f}")

        # Если есть доп. уравнения
        if n_rows > n_nodes:
            for i in range(n_nodes, n_rows):
                row_vals = " ".join(f"{A[i, j]:6.2f}" for j in range(n_cols))
                logger.info(f"Row{i + 1:2d}(extra): {row_vals} | {b[i]:6.2f}")

        logger.info("----- Конец системы уравнений -----\n")
