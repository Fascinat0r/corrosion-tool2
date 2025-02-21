from models.pipeline import Pipeline
from services.visualization import plot_pipeline_data, plot_all_columns


def main():
    pipeline = Pipeline('data/pipes.json', 'data/input.json')
    try:
        pipeline.simulate()
    except ValueError as e:
        print(f"Error while simulating the pipeline:\n {e}")
    out_path = 'data/output/output.csv'
    pipeline.save_sections_data_to_csv(out_path)
    plot_pipeline_data(out_path)
    plot_all_columns(out_path)


if __name__ == "__main__":
    main()
