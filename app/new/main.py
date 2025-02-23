# main.py

from graph_adapter_nx import GraphAdapterNX
from logger import setup_logger
from pipeline_repository import PipelineRepository


def main():
    logger = setup_logger()

    # Путь к JSON
    json_path = "../../data/pipeline_graph_example.json"

    # 1) Создаем адаптер
    adapter = GraphAdapterNX()

    # 2) Загружаем данные из JSON в адаптер (через репозиторий)
    repo = PipelineRepository(adapter)
    repo.load_from_json(json_path)

    # 3) Перебираем все варианты
    variants = repo.get_digraph_variants()
    logger.info(f"Итоговое кол-во вариантов: {len(variants)}")

    # 5) Определим типы узлов в каждом варианте
    for i, var_adapter in enumerate(variants):
        var_repo = PipelineRepository(var_adapter, input_nodes=repo.input_nodes, output_nodes=repo.output_nodes)
        var_repo.define_node_types()
        var_repo.draw_graph(title=f"Pipeline graph variant {i + 1}")

        # 6) Решаем систему уравнений для каждого варианта
        var_repo.solve_system()


if __name__ == "__main__":
    main()
