# main_slau.py
import sys

from app.new.logger import setup_logger
from app.new.slau_service import SLASolverService


def main():
    logger = setup_logger()
    if len(sys.argv) < 2:
        print("Usage: python main_slau.py path_to_json")
        sys.exit(1)

    json_path = sys.argv[1]

    solver = SLASolverService()

    solver.init(json_path)

    # Пример: tolerance_q=1e-5 (по расходу), tolerance_res=1e-2 (по невязке)
    Q_values = solver.solve_iteratively(max_iter=50, tolerance=1e-5, tolerance_resid=1e-2, visualize_each_iter=True)

    digraph = solver.graph
    logger.info("Итоговые расходы по рёбрам:")
    edge_list = list(digraph.edges())
    for i, e in enumerate(edge_list):
        logger.info(f"  {e}: Q = {Q_values[i]}")

    for e in edge_list:
        props = digraph.edges[e]["properties"]
        logger.info(f"{e}: R={props['R']:.4g}, dPfix={props['dPfix']:.4g}, dP={props['dP']:.4g}")


if __name__ == "__main__":
    main()
