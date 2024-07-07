from models.pipeline import Pipeline
from services.visualization import plot_pipeline_data, plot_all_columns


def main():
    pipeline = Pipeline('cortool/data/pipes.json', 'cortool/data/input.json')
    pipeline.simulate()
    out_path = 'cortool/data/output.csv'
    pipeline.save_sections_data_to_csv(out_path)
    plot_pipeline_data(out_path)
    plot_all_columns(out_path)


if __name__ == "__main__":
    main()
