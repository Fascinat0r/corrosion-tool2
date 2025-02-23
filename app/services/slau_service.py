import logging
from typing import Tuple

import numpy as np

from models.node import NodeType
from services.graph_service import PipelineGraphService

logger = logging.getLogger(__name__)


class CalculateSLAEUseCase:
    """
    Юзкейс для расчёта системы линейных алгебраических уравнений (СЛАУ)
    для трубопроводной системы.

    Входные параметры:
      - Для узлов типа CONSUMER (потребителей): в поле data узла должен быть задан параметр "demand",
        равный требуемому расходу.
      - Для узлов типа BRANCH (разветвителей): используется базовое уравнение баланса потока:
        сумма расходов на входе минус сумма расходов на выходе равна 0.
      - Для узлов типа DIVIDER (водоразделов):
          * Если в data узла заданы параметры "deltaP_in" и "deltaP_out", а в свойствах ребер присутствует
            "resistance", то используется расширенная модель:
              sum(R_i * Q_in_i) - sum(R_j * Q_out_j) = deltaP_in - deltaP_out.
          * Иначе применяется базовый баланс: сумма расходов на входе минус сумма расходов на выходе равна 0.
    Необязательные узлы INPUT и OUTPUT не участвуют в составлении СЛАУ.
    """

    def __init__(self, graph_service: PipelineGraphService):
        self.graph_service = graph_service

    def execute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисляет матрицу коэффициентов A, вектор свободных членов b и решает систему A * Q = b,
        где Q – вектор расходов по трубам.

        Возвращает:
            A: numpy.ndarray – матрица коэффициентов,
            b: numpy.ndarray – вектор свободных членов,
            Q: numpy.ndarray – найденный вектор расходов.
        """
        # Загружаем граф трубопроводной системы
        graph = self.graph_service.load_pipeline()

        # Составляем упорядоченный список труб (ребер) для неизвестных расходов Q
        pipes = list(graph.edges(data=True))
        num_pipes = len(pipes)
        pipe_index = {(u, v): idx for idx, (u, v, _) in enumerate(pipes)}

        # Формируем список узлов, для которых составляем уравнения:
        # рассматриваем узлы типов: CONSUMER, BRANCH и DIVIDER.
        equation_nodes = [
            n for n, attr in graph.nodes(data=True)
            if attr.get("type") in [NodeType.CONSUMER, NodeType.BRANCH, NodeType.DIVIDER]
        ]
        num_eq = len(equation_nodes)

        # Инициализируем матрицу A (num_eq x num_pipes) и вектор b (num_eq)
        A = np.zeros((num_eq, num_pipes))
        b = np.zeros(num_eq)

        # Проходим по каждому узлу и составляем соответствующее уравнение
        for eq_idx, node in enumerate(equation_nodes):
            node_attr = graph.nodes[node]
            node_type = node_attr.get("type")
            # Выбираем все входящие и исходящие трубы (ребра) для текущего узла
            in_edges = list(graph.in_edges(node, data=True))
            out_edges = list(graph.out_edges(node, data=True))

            if node_type == NodeType.CONSUMER:
                # Узел-потребитель: предположим, что имеется одна входящая труба
                if len(in_edges) != 1:
                    logger.warning(
                        f"Узел {node} типа CONSUMER имеет {len(in_edges)} входящих ребер, ожидается 1."
                    )
                for u, v, data in in_edges:
                    idx = pipe_index[(u, v)]
                    A[eq_idx, idx] = 1
                # Из data узла получаем требуемый расход (demand)
                demand = node_attr.get("data", {}).get("demand")
                if demand is None:
                    logger.error(f"Узел {node} типа CONSUMER не имеет заданного расхода 'demand'.")
                    raise ValueError(f"Узел {node} типа CONSUMER должен иметь параметр 'demand' в data.")
                b[eq_idx] = demand

            elif node_type == NodeType.BRANCH:
                # Узел-разветвитель: баланс потока – сумма входящих расходов минус сумма исходящих равна 0
                for u, v, data in in_edges:
                    idx = pipe_index[(u, v)]
                    A[eq_idx, idx] = 1
                for u, v, data in out_edges:
                    idx = pipe_index[(u, v)]
                    A[eq_idx, idx] = -1
                b[eq_idx] = 0

            elif node_type == NodeType.DIVIDER:
                # Узел-водораздел: два варианта уравнения
                node_data = node_attr.get("data", {})
                if "deltaP_in" in node_data and "deltaP_out" in node_data:
                    # Расширенная модель с учетом сопротивления труб
                    deltaP_in = node_data.get("deltaP_in")
                    deltaP_out = node_data.get("deltaP_out")
                    for u, v, edge_attr in in_edges:
                        resistance = edge_attr.get("properties", {}).get("resistance")
                        if resistance is None:
                            logger.error(f"Ребро ({u} -> {v}) не имеет параметра 'resistance'.")
                            raise ValueError(
                                f"Ребро ({u} -> {v}) должно иметь параметр 'resistance' для узла DIVIDER."
                            )
                        idx = pipe_index[(u, v)]
                        A[eq_idx, idx] = resistance
                    for u, v, edge_attr in out_edges:
                        resistance = edge_attr.get("properties", {}).get("resistance")
                        if resistance is None:
                            logger.error(f"Ребро ({u} -> {v}) не имеет параметра 'resistance'.")
                            raise ValueError(
                                f"Ребро ({u} -> {v}) должно иметь параметр 'resistance' для узла DIVIDER."
                            )
                        idx = pipe_index[(u, v)]
                        A[eq_idx, idx] = -resistance
                    b[eq_idx] = deltaP_in - deltaP_out
                else:
                    # Базовый баланс потока: сумма входящих расходов минус сумма исходящих равна 0
                    for u, v, data in in_edges:
                        idx = pipe_index[(u, v)]
                        A[eq_idx, idx] = 1
                    for u, v, data in out_edges:
                        idx = pipe_index[(u, v)]
                        A[eq_idx, idx] = -1
                    b[eq_idx] = 0

            else:
                logger.info(f"Узел {node} типа {node_type} не участвует в расчёте SLAE.")
                continue

            logger.info(
                f"Узел {node}: составлено уравнение A[{eq_idx}] = {A[eq_idx]}, b[{eq_idx}] = {b[eq_idx]}"
            )

        # Решаем систему уравнений: если система квадратная, используем прямое решение, иначе - метод наименьших квадратов.
        try:
            if A.shape[0] == A.shape[1]:
                Q = np.linalg.solve(A, b)
            else:
                Q, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                logger.info(f"Решение методом наименьших квадратов, остатки: {residuals}")
        except np.linalg.LinAlgError as e:
            logger.error(f"Ошибка при решении системы уравнений: {e}")
            raise

        logger.info(f"Найденные значения расходов (Q): {Q}")
        return A, b, Q


# Пример использования юзкейса:
if __name__ == "__main__":
    # Предполагается, что PipelineGraphService инициализируется с нужным репозиторием
    from repository.network import PipelineGraphRepository

    # Задайте путь к JSON файлу с описанием графа
    json_file_path = "data/pipeline_graph.json"
    repository = PipelineGraphRepository(json_file=json_file_path)
    graph_service = PipelineGraphService(repository=repository)
    slae_usecase = CalculateSLAEUseCase(graph_service=graph_service)

    A, b, Q = slae_usecase.execute()

    print("Матрица коэффициентов A:")
    print(A)
    print("Вектор свободных членов b:")
    print(b)
    print("Решенные расходы Q:")
    print(Q)
