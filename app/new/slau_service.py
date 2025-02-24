import logging
from typing import List, Tuple

import networkx as nx
import numpy as np

from new.models import NodeType

logger = logging.getLogger("pipeline_logger")


class SLASolverService:
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
         h) Если невязка давлений в узлах-водоразделах (разница dP среди входящих ребер) меньше tolerance,
            итерации завершаются.
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
        logger.debug(f"Инициализировано {len(self.edge_list)} рёбер.")

    def _compute_dP_physics(self):
        """
        Пересчитывает dP для каждого ребра по упрощенной формуле (например, Дарси–Вейсбаха).
        """
        for e in self.edge_list:
            props = self.graph.edges[e]["properties"]
            Q = props["Q"]
            length = props.get("length", 0.0)
            diam = props.get("diameter", 0.0)
            # Простая формула: dP = k * Q * |Q|, где k = length/(diameter^4)
            try:
                dp_val = Q * abs(Q) * (length / (diam ** 4))
            except Exception as ex:
                logger.error(f"Ошибка при расчёте dP для ребра {e}: {ex}")
                dp_val = 0.0
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
        Возвращает матрицу A_ext и вектор b_ext.
        """
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        n_edges = len(self.edge_list)
        A_base = np.zeros((n_nodes, n_edges))
        b_base = np.zeros(n_nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Базовые уравнения (массовый баланс)
        for n in nodes:
            i = node_to_idx[n]
            tp = self.graph.nodes[n].get("type", NodeType.UNKNOWN)
            data = self.graph.nodes[n].get("data", {})
            if tp.value == NodeType.INPUT.value:
                supply = float(data.get("supply", 0.0))
                b_base[i] = -supply
            elif tp.value == NodeType.CONSUMER.value:
                demand = float(data.get("demand", 0.0))
                b_base[i] = demand
            else:
                b_base[i] = 0.0
            # Заполнение коэффициентов: входящие ребра +1, исходящие -1
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
        # Каждая такая строка соответствует дополнительному уравнению (сравнение сумм потерь по двум цепочкам)
        for n in nodes:
            if self.graph.nodes[n].get("type", NodeType.UNKNOWN).value != NodeType.DIVIDER.value:
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
                    # Если ребро содержит обе метки, пропускаем
                extra_rows.append(row)
                extra_b.append(b_val)
                logger.info(f"Узел {n}, сравнение цепочек '{ref}' vs '{cur}', b = {b_val:.2f}")

        if extra_rows:
            A_extra = np.array(extra_rows)
            b_extra = np.array(extra_b)
            A_ext = np.vstack((A_base, A_extra))
            b_ext = np.concatenate((b_base, b_extra))
        else:
            A_ext = A_base
            b_ext = b_base

        return A_ext, b_ext

    def _collect_chains_for_node(self, node: str) -> List[str]:
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

    def print_pretty_system(self, A_ext, b_ext):
        """
        Выводит ее в удобном табличном виде.
        Первые n строк соответствуют базовым уравнениям (массовый баланс) для узлов,
        а следующие строки – дополнительным уравнениям для узлов DIVIDER.
        Каждая строка подписывается идентификатором узла и его типом (или помечается как extra eqn).
        """
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        n_total = A_ext.shape[0]
        # Выводим базовую часть
        header = "Base equations (массовый баланс):"
        logger.info(header)
        logger.info("-" * len(header))
        for i in range(n_nodes):
            node = nodes[i]
            node_type = self.graph.nodes[node].get("type", NodeType.UNKNOWN)
            row_str = " | ".join(f"{A_ext[i, j]:10.3f}" for j in range(A_ext.shape[1]))
            print(f"Узел {node:4s}({node_type.title+')':15s}:{row_str} | b = {b_ext[i]:10.3f}")

        # Если есть дополнительные строки – для узлов DIVIDER
        if n_total > n_nodes:
            for i in range(n_nodes, n_total):
                # Здесь мы не знаем точно, к какому узлу относится дополнительное уравнение,
                # но можем пометить его как extra eqn # (i - n_nodes + 1)
                row_str = " | ".join(f"{A_ext[i, j]:10.3f}" for j in range(A_ext.shape[1]))
                print(f"Extra eqn #{str(i - n_nodes + 1):14s}:{row_str} | b = {b_ext[i]:10.3f}")

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
                    if self.graph.nodes[prev].get("type").value != NodeType.CONSUMER.value:
                        break
                elif direction == "forward":
                    succs = list(self.graph.successors(current))
                    if len(succs) != 1:
                        break
                    nxt = succs[0]
                    if self.graph.nodes[nxt].get("type").value != NodeType.CONSUMER.value:
                        break
                    edges.append((current, nxt))
                    current = nxt
                else:
                    break
            return edges

        # Если u является потребителем, расширяем цепочку назад от u
        if self.graph.nodes[u].get("type").value == NodeType.CONSUMER.value:
            backward = extend_chain(u, "backward")
            chain_edges = backward + chain_edges

        # Если v является потребителем, расширяем цепочку вперед от v
        if self.graph.nodes[v].get("type").value == NodeType.CONSUMER.value:
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
            if self.graph.nodes[n].get("type").value == NodeType.DIVIDER.value:
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
        if not self._validate_graph():
            raise ValueError("Граф не прошел валидацию.")

        for it in range(1, max_iter + 1):
            logger.info(f"=== Итерация {it} ===")
            self._build_divider_chains()
            self._compute_R_and_dPfix()
            A_ext, b_ext = self._build_system()

            # Выводим систему уравнений в удобном виде
            self.print_pretty_system(A_ext, b_ext)

            Q, residuals, rank, s = np.linalg.lstsq(A_ext, b_ext, rcond=None)
            logger.info(f"Итерация {it}: rank={rank}, residuals={residuals}")
            for idx, e in enumerate(self.edge_list):
                props = self.graph.edges[e]["properties"]
                props["Q_old"] = props["Q"]
                props["Q"] = Q[idx]
            self._compute_dP_physics()

            # Ищем внутренние ребра с отрицательным Q
            neg_edges = []
            for e in self.edge_list:
                q = self.graph.edges[e]["properties"]["Q"]
                utp = self.graph.nodes[e[0]].get("type", NodeType.UNKNOWN).value
                vtp = self.graph.nodes[e[1]].get("type", NodeType.UNKNOWN).value
                if q < -tolerance and (utp not in [NodeType.INPUT.value] and vtp not in [NodeType.OUTPUT.value]):
                    neg_edges.append(e)
            if neg_edges:
                logger.info(f"Найдено {len(neg_edges)} рёбер с отрицательным Q")
        return [self.graph.edges[e]["properties"]["Q"] for e in self.edge_list]

    def _build_divider_chains(self, max_depth: int = 10):
        """
        Для каждого узла типа DIVIDER:
          - Находит все простые пути (с учетом направления) от INPUT-узлов к нему и от него к OUTPUT-узлам,
            без повторения узлов (циклов).
          - Каждому найденному пути присваивает уникальное имя (chain_inX или chain_outY).
          - Помечает все рёбра на найденном пути этой меткой.
          - Логирует цепочки.
        """

        # Получаем списки INPUT, OUTPUT и DIVIDER узлов
        input_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type").value == NodeType.INPUT.value]
        output_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type").value == NodeType.OUTPUT.value]
        dividers = [n for n, d in self.graph.nodes(data=True) if d.get("type").value == NodeType.DIVIDER.value]

        chain_in_counter = 1
        chain_out_counter = 1

        for d_node in dividers:
            logger.info(f"Обрабатываем узел-водораздел {d_node}")
            divider_chains = {"input_chains": [], "output_chains": []}

            # Поиск путей от INPUT к d_node (учитываем направление)
            for inp in input_nodes:
                try:
                    paths = list(nx.all_simple_paths(self.graph, source=inp, target=d_node, cutoff=max_depth))
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
                    paths = list(nx.all_simple_paths(self.graph, source=d_node, target=out, cutoff=max_depth))
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
            if self.graph.has_edge(u, v):
                props = self.graph.edges[u, v].setdefault("properties", {})
                old_chain = props.get("chain", "")
                existing = [c.strip() for c in old_chain.split(",") if c.strip()]
                if chain_name not in existing:
                    new_chain = old_chain + ("," if old_chain else "") + chain_name
                    props["chain"] = new_chain

    def _validate_graph(self) -> bool:
        """
        Проверяет, что все узлы и рёбра имеют необходимые данные для корректного расчёта.

        Для узлов:
          - Проверяется, что задан параметр 'type' (NodeType).
          - Если узел имеет тип INPUT, то в data должен присутствовать параметр 'supply' (и он должен быть положительным).
          - Если узел имеет тип CONSUMER, то в data должен присутствовать параметр 'demand'.

        Для рёбер:
          - В properties должны быть заданы параметры 'length' и 'diameter'.
          - Диаметр должен быть положительным (иначе деление на 0 при расчёте dP).

        Если хотя бы одна проверка не пройдена, метод возвращает False, при этом все ошибки логируются.
        """
        iso = list(nx.isolates(self.graph))
        if iso:
            logger.warning(f"Изолированные узлы: {iso}")
            raise
        if not nx.is_weakly_connected(self.graph):
            logger.warning("Граф не слабо-связный. Возможны несвязные компоненты.")

        valid = True

        # Проверка узлов
        for node, data in self.graph.nodes(data=True):
            # Проверяем наличие типа
            if "type" not in data:
                logger.error(f"Узел {node} не имеет атрибута 'type'.")
                valid = False
            else:
                node_type = data["type"]
                if node_type.value == NodeType.INPUT.value:
                    supply = data.get("data", {}).get("supply", None)
                    if supply is None:
                        logger.error(f"Входной узел {node} не содержит параметр 'supply'.")
                        valid = False
                    else:
                        try:
                            if float(supply) <= 0:
                                logger.error(f"Входной узел {node} имеет некорректное значение supply: {supply}.")
                                valid = False
                        except Exception as ex:
                            logger.error(f"Невозможно преобразовать supply узла {node}: {supply} ({ex}).")
                            valid = False
                else:
                    demand = data.get("data", {}).get("demand", None)
                    if demand is None:
                        logger.error(f"Потребитель {node} не содержит параметр 'demand'.")
                        valid = False
                    else:
                        try:
                            if float(demand) <= 0:
                                logger.error(f"Потребитель {node} имеет некорректное значение demand: {demand}.")
                                valid = False
                        except Exception as ex:
                            logger.error(f"Невозможно преобразовать demand узла {node}: {demand} ({ex}).")
                            valid = False

        # Проверка рёбер
        for e in self.edge_list:
            props = self.graph.edges[e].get("properties", {})
            # Проверяем наличие длины
            if "length" not in props:
                logger.error(f"Ребро {e} не содержит параметр 'length'.")
                valid = False
            else:
                try:
                    length = float(props["length"])
                    if length <= 0:
                        logger.error(f"Ребро {e} имеет некорректную длину: {length}.")
                        valid = False
                except Exception as ex:
                    logger.error(f"Невозможно преобразовать параметр 'length' ребра {e}: {props['length']} ({ex}).")
                    valid = False

            # Проверяем наличие диаметра
            if "diameter" not in props:
                logger.error(f"Ребро {e} не содержит параметр 'diameter'.")
                valid = False
            else:
                try:
                    diam = float(props["diameter"])
                    if diam <= 0:
                        logger.error(f"Ребро {e} имеет некорректный диаметр: {diam}.")
                        valid = False
                except Exception as ex:
                    logger.error(
                        f"Невозможно преобразовать параметр 'diameter' ребра {e}: {props['diameter']} ({ex}).")
                    valid = False

            # Можно добавить проверки для других параметров (например, roughness), если они критичны

        if valid:
            logger.info("Проверка данных завершена: все необходимые параметры присутствуют и корректны.")
        else:
            logger.error("Проверка данных завершена: обнаружены ошибки в входных данных.")
        return valid
