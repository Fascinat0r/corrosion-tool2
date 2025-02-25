# main.py
import sys

from graph_adapter_nx import GraphAdapterNX
from logger import setup_logger
from pipeline_repository import PipelineRepository
from slau_service import SLASolverService


def main():
    logger = setup_logger()
    if len(sys.argv) < 2:
        print("Usage: python main.py path_to_json")
        sys.exit(1)

    json_path = sys.argv[1]

    # 1. Создаём адаптер
    adapter = GraphAdapterNX()
    # 2. Репозиторий
    repo = PipelineRepository(adapter)
    repo.load_from_json(json_path)

    # 3. Определяем типы узлов
    repo.define_node_types()

    # 4. Запускаем Solver
    digraph = adapter.digraph
    solver = SLASolverService(digraph)

    # 5. Считаем
    Q_values = solver.solve_iteratively(max_iter=10, tolerance=1e-5)
    logger.info("Итоговые расходы по рёбрам:")
    edge_list = list(digraph.edges())
    for i, e in enumerate(edge_list):
        logger.info(f"  {e}: Q = {Q_values[i]}")

    # При желании: вывести dP, R и т.д.
    for e in edge_list:
        props = digraph.edges[e]["properties"]
        logger.info(f"{e}: R={props['R']:.4g}, dPfix={props['dPfix']:.4g}, dP={props['dP']:.4g}")


if __name__ == "__main__":
    main()
