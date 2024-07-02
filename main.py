from models.pipeline import Pipeline


def main():
    pipeline = Pipeline('cortool/data/pipes.json', 'cortool/data/input.json')
    pipeline.simulate()
    pipeline.save_sections_data_to_csv('cortool/data/output.csv')


if __name__ == "__main__":
    main()
