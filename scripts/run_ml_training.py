from core.pipelines import TrainPipeline
import argparse

#Add arguments for cli
def main():
    parser = argparse.ArgumentParser(description="Run ML training pipeline")
    parser.add_argument('-f', '--filename', type=str, required=True, help='Path to the dataset file')
    args = parser.parse_args()

    # Create an instance of TrainPipeline
    train_pipeline = TrainPipeline()

    # Run the training pipeline with the provided filename
    train_pipeline.run(args.filename)


if __name__ == "__main__":
    main()
