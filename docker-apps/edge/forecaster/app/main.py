import argparse
import json
import os

from edge.minio_client import setup_minio_client
from edge.prepare_forecasting import setup_model


def setup(args: argparse.Namespace):
    with open(args.settings_file, "rt") as f:
        settings = json.load(f)

    minio_client = setup_minio_client(settings["minio_settings"])

    for model_configuration in settings["forecast_models"]:
        forecast_publisher, other_threads = setup_model(model_configuration, minio_client)
        for thread in other_threads:
            thread.start()
        forecast_publisher.run()
        for thread in other_threads:
            thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings-file",
        type=str,
        default=os.path.join("forecaster-configuration", "settings.json"),
        help="Path to the settings file.",
    )
    setup(parser.parse_args())
