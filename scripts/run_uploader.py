from core.pipelines import UploadPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the upload pipeline for .md files.")
    parser.add_argument("-d","--directory", type=str, help="Directory containing .md files to upload")
    args = parser.parse_args()

    upload_pipeline = UploadPipeline()
    upload_pipeline.run(args.directory)

if __name__ == "__main__":
    main()

