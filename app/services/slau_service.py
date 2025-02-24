# services/slau_service.py
import logging
from typing import List, Tuple

import networkx as nx
import numpy as np

from new.models import NodeType

logger = logging.getLogger(__name__)


class SLASolverServiceOld:
    """
    Итерационный расчет гидравлической сети с учетом дополнительных уравнений для узлов-водоразделов.

    Алгоритм:
      1. Инициализация начальных значений (Q, R, dPfix).
      2. На каждой итерации:
         a) Пересчет R(n) и dPfix(n) по формулам (2.1)/(3.1) и (2.2)/(3.2).
         b) Построение расширенной системы уравнений A_ext * Q = b_ext:
            – базовые уравнения массового баланса для всех узлов,
            – дополнительные уравнения для узлов DIVIDER на основе chain-меток.
         c) Решение системы и обновление Q.
         d) Пересчет физических потерь dP по заданной формуле.
         e) Поиск внутренних ребер с Q < –tolerance.
         f) Если найдены такие ребра, производится переворот:
            если ребро касается потребителя, то собирается вся цепочка потребительских узлов и переворачивается целиком.
         g) Если были перевороты, сбрасываются chain-метки и пересчитываются типы узлов и цепочки (сброс всей информации о направлениях).
         h) Если невязка давлений в узлах-водоразделах (разница dP среди входящих ребер) меньше tolerance, итерации завершаются.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.edge_list: List[Tuple[str, str]] = []
        self._init_edges()

    def _init_edges(self):
        """Инициализируем список рёбер и задаем начальные свойства для каждого ребра."""
        self.edge_list = list(self.graph.edges())
        for e in self.edge_list:
            props = self.graph.edges[e].setdefault("properties", {})
            props.setdefault("Q", 0.0)
            props.setdefault("Q_old", 0.0)
            props.setdefault("dP", 0.0)
            props.setdefault("dP_old", 0.0)
            props.setdefault("R", 1e-3)  # Начальное сопротивление (эвристика)
            props.setdefault("dPfix", 0.0)  # Начальное фиксированное падение давления
        logger.info(f"Инициализировано {len(self.edge_list)} рёбер.")

    def _compute_dP_physics(self):
        """
        Пересчитывает dP для каждого ребра по упрощенной формуле (например, Дарси–Вейсбаха).
        """
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q = props["Q"]
            length = props.get("length", 100.0)
            diam = props.get("diameter", 0.5)
            # Простая формула: dP = k * Q * |Q|, где k = length/(diameter^4)
            dp_val = Q * abs(Q) * (length / (diam ** 4))
            props["dP_old"] = props["dP"]
            props["dP"] = dp_val

    def _compute_R_and_dPfix(self):
        """
        Вычисляет R и dPfix для каждого ребра по формулам:
          Если |Q - Q_old| > epsilon:
              R = (dP - dP_old) / (Q - Q_old)
              dPfix = dP - R * Q
          Иначе:
              Если Q != 0: R = dP / Q, dPfix = 0
              Если Q = 0: R = большое число, dPfix = 0
        """
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q_old = props["Q_old"]
            Q_curr = props["Q"]
            dP_old = props["dP_old"]
            dP_curr = props["dP"]
            if abs(Q_curr - Q_old) > 1e-12:
                R_candidate = (dP_curr - dP_old) / (Q_curr - Q_old)
                R_candidate = min(max(R_candidate, 1e-12), 1e5)
                props["R"] = R_candidate
                props["dPfix"] = dP_curr - R_candidate * Q_curr
            else:
                if abs(Q_curr) > 1e-12:
                    r_val = dP_curr / Q_curr
                    r_val = min(max(r_val, 1e-12), 1e5)
                    props["R"] = r_val
                    props["dPfix"] = 0.0
                else:
                    props["R"] = 1e5
                    props["dPfix"] = 0.0

    def _build_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Формирует систему уравнений A_ext * Q = b_ext.
          - Первые n строк – базовые уравнения (массовый баланс) для всех узлов.
          - Дополнительные строки – для узлов DIVIDER (учитывая chain-метки, уравнения типа 3.2).
        """
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        n_edges = len(self.edge_list)
        A_base = np.zeros((n_nodes, n_edges))
        b_base = np.zeros(n_nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Базовые уравнения
        for n in nodes:
            i = node_to_idx[n]
            tp = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            data = self.graph.nodes[n].get("data", {})
            if tp == NodeType.INPUT:
                supply = float(data.get("supply", 0.0))
                b_base[i] = -supply
            elif tp == NodeType.CONSUMER:
                demand = float(data.get("demand", 0.0))
                b_base[i] = demand
            else:
                b_base[i] = 0.0
            for pred in self.graph.predecessors(n):
                if (pred, n) in self.edge_list:
                    j = self.edge_list.index((pred, n))
                    A_base[i, j] += 1
            for succ in self.graph.successors(n):
                if (n, succ) in self.edge_list:
                    j = self.edge_list.index((n, succ))
                    A_base[i, j] -= 1

        extra_rows = []
        extra_b = []
        # Дополнительные уравнения для узлов DIVIDER
        for n in nodes:
            if self.graph.nodes[n].get("type", NodeType.UNKNOWN) != NodeType.DIVIDER:
                continue
            chains = self._collect_chains_for_node(n)
            if len(chains) <= 1:
                continue
            ref = chains[0]
            for cur in chains[1:]:
                row = np.zeros(n_edges)
                b_val = 0.0
                for idx, e in enumerate(self.edge_list):
                    ch = self.graph.edges[e]["properties"].get("chain", "")
                    R_val = self.graph.edges[e]["properties"].get("R", 1e-3)
                    dPfix = self.graph.edges[e]["properties"].get("dPfix", 0.0)
                    has_ref = (ref in ch)
                    has_cur = (cur in ch)
                    if has_ref and not has_cur:
                        row[idx] += R_val
                        b_val += dPfix
                    elif has_cur and not has_ref:
                        row[idx] -= R_val
                        b_val -= dPfix
                    elif has_ref and has_cur:
                        logger.warning(f"Ребро {e} принадлежит двум цепочкам: ref={ref}, cur={cur}")
                extra_rows.append(row)
                extra_b.append(b_val)
                logger.info(f"Узел {n}, цепочки '{ref}' vs '{cur}', b={b_val:.2f}")

        if extra_rows:
            A_extra = np.array(extra_rows)
            b_extra = np.array(extra_b)
            A_ext = np.vstack((A_base, A_extra))
            b_ext = np.concatenate((b_base, b_extra))
        else:
            A_ext = A_base
            b_ext = b_base

        logger.info("Сформированная матрица A_ext:")
        print(A_ext)
        logger.info("Сформированный вектор b_ext:")
        print(b_ext)
        return A_ext, b_ext

    def _collect_chains_for_node(self, node: str):
        """
        Собирает все уникальные chain-метки у рёбер, инцидентных узлу.
        """
        cset = set()
        for e in self.edge_list:
            if node in e:
                ch = self.graph.edges[e]["properties"].get("chain", "")
                if ch:
                    for token in ch.split(","):
                        token = token.strip()
                        if token:
                            cset.add(token)
        return sorted(list(cset))

    def _collect_consumer_chain(self, edge: Tuple[str, str]) -> List[Tuple[str, str]]:
        """
        По заданному ребру, если один из концов – потребитель (NodeType.CONSUMER), собирает
        весь сегмент цепочки, состоящий из последовательных потребительских узлов (без ветвлений).
        Возвращает список ребер, принадлежащих цепочке (в порядке следования).
        Алгоритм:
          - Если один из концов ребра – потребитель, то, начиная с этого узла, двигаемся назад,
            пока следующий узел является потребителем и имеет ровно один вход, и аналогично вперед.
          - Возвращаем все ребра, составляющие эту цепочку.
        """
        u, v = edge
        chain_edges = [edge]

        # Функция для расширения цепочки в указанном направлении
        def extend_chain(start_node: str, direction: str) -> List[Tuple[str, str]]:
            edges = []
            current = start_node
            while True:
                if direction == "backward":
                    # Ищем предшественника: должен быть ровно один и его тип должен быть CONSUMER
                    preds = list(self.graph.predecessors(current))
                    # Исключаем ветвления: если больше одного – останавливаемся
                    if len(preds) != 1:
                        break
                    prev = preds[0]
                    # Добавляем ребро (prev, current)
                    edges.insert(0, (prev, current))
                    current = prev
                    if self.graph.nodes[prev].get("type") != NodeType.CONSUMER:
                        break
                elif direction == "forward":
                    succs = list(self.graph.successors(current))
                    if len(succs) != 1:
                        break
                    nxt = succs[0]
                    if self.graph.nodes[nxt].get("type") != NodeType.CONSUMER:
                        break
                    edges.append((current, nxt))
                    current = nxt
                else:
                    break
            return edges

        # Если u является потребителем, расширяем цепочку назад от u
        if self.graph.nodes[u].get("type") == NodeType.CONSUMER:
            backward = extend_chain(u, "backward")
            chain_edges = backward + chain_edges

        # Если v является потребителем, расширяем цепочку вперед от v
        if self.graph.nodes[v].get("type") == NodeType.CONSUMER:
            forward = extend_chain(v, "forward")
            chain_edges = chain_edges + forward

        logger.info(f"Собрана цепочка для ребра {edge}: {chain_edges}")
        return chain_edges

    def _check_divider_pressures(self, tolerance: float) -> bool:
        """
        Для узлов-водоразделов (DIVIDER) проверяет невязку давлений – разница между
        максимальным и минимальным абсолютным значением dP входящих ребер.
        Если разница меньше tolerance, возвращает True.
        """
        for n in self.graph.nodes():
            if self.graph.nodes[n].get("type") == NodeType.DIVIDER:
                in_edges = [(u, v) for (u, v) in self.edge_list if v == n]
                dp_list = [abs(self.graph.edges[e]["properties"].get("dP", 0.0)) for e in in_edges]
                if dp_list:
                    diff = max(dp_list) - min(dp_list)
                    if diff > tolerance:
                        logger.info(f"Узел {n}: разница давлений = {diff:.4f} > tol={tolerance}")
                        return False
        return True

    def solve_iteratively(self, max_iter=20, tolerance=1e-6):
        """
        Итерационный процесс:
          - На каждой итерации вычисляем R, dPfix, формируем систему, решаем, обновляем Q, dP.
          - Если невязка в узлах DIVIDER (разница dP входящих ребер) меньше tolerance, завершаем итерации.
        """
        for it in range(1, max_iter + 1):
            logger.debug(f"=== Итерация {it} ===")
            self._compute_R_and_dPfix()
            A_ext, b_ext = self._build_system()
            Q, residuals, rank, s = np.linalg.lstsq(A_ext, b_ext, rcond=None)
            logger.debug(f"Итерация {it}: rank={rank}, residuals={residuals}")
            for idx, e in enumerate(self.edge_list):
                props = self.graph.edges[e]["properties"]
                props["Q_old"] = props["Q"]
                props["Q"] = Q[idx]
            self._compute_dP_physics()

            # Ищем внутренние ребра с отрицательным Q
            neg_edges = []
            for e in self.edge_list:
                q = self.graph.edges[e]["properties"]["Q"]
                utp = self.graph.nodes[e[0]].get("type", NodeType.UNKNOWN)
                vtp = self.graph.nodes[e[1]].get("type", NodeType.UNKNOWN)
                if q < -tolerance and (utp not in [NodeType.INPUT] and vtp not in [NodeType.OUTPUT]):
                    neg_edges.append(e)
            if neg_edges:
                logger.debug(f"Найдено {len(neg_edges)} рёбер с отрицательным Q..")
            else:
                logger.info("Нет отрицательных внутренних расходов. Проверяем невязку в узлах-водоразделах.")
                if self._check_divider_pressures(tolerance):
                    logger.info("Невязка удовлетворяет tolerance. Завершаем итерации.")
                    break
                else:
                    logger.info("Невязка превышает tolerance, продолжаем итерации.")
        else:
            logger.warning("Достигнут лимит итераций, решение может быть неточным.")

        for e in self.edge_list:
            q = self.graph.edges[e]["properties"]["Q"]
            logger.info(f"{e[0]} -> {e[1]}: Q={q:.4f}")
        return [self.graph.edges[e]["properties"]["Q"] for e in self.edge_list]

    def validate_graph(self):
        self