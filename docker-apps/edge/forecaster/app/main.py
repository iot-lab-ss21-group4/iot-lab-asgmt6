import argparse
import json
import logging
import os
import queue
import sys
import threading
from typing import Any, Dict, List

# suppress ssl warning for IoT platform
import urllib3

from edge.minio_client import setup_minio_client
from edge.prepare_forecasting import setup_model
from edge.thread.forecast_evaluator_thread import ForecastEvaluatorThread
from edge.thread.timer_thread import TimerThread
from edge.util.accuracy import AccuracyCalculator
from edge.util.data_fetcher import DataFetcher
from edge.util.edge_broker_publisher import EdgeBrokerPublisher
from edge.util.forecast_combiner import ForecastCombiner
from edge.util.platform_sensor_publisher import PlatformSensorPublisher

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def setup(args: argparse.Namespace):
    logging.info("Reading the settings file '{}'".format(args.settings_file))
    with open(args.settings_file, "rt") as f:
        settings: Dict[str, Any] = json.load(f)

    # create singleton mqtt publisher
    # under the assumption that one device (topic: username_deviceId) is sufficient
    platform_sensor_publisher = PlatformSensorPublisher(settings["iot_platform_mqtt_settings"])
    edge_broker_publisher = EdgeBrokerPublisher(settings["edge_broker_mqtt_settings"])

    all_threads: List[threading.Thread] = []

    forecast_evaluator_in_q = queue.Queue()
    data_fetcher = DataFetcher(settings["iot_platform_consumer_settings"])
    forecast_evaluator_thread = ForecastEvaluatorThread(
        event_in_q=forecast_evaluator_in_q,
        platform_sensor_publisher=platform_sensor_publisher,
        edge_broker_publisher=edge_broker_publisher,
        data_fetcher=data_fetcher,
        number_of_models=len(settings["forecast_models"]),
        accuracy_calculator=AccuracyCalculator(),
        forecast_combiner=ForecastCombiner.build_from_config(settings["forecast_combine_strategy"]),
    )
    all_threads.append(forecast_evaluator_thread)

    forecaster_in_qs = []
    minio_client = setup_minio_client(settings["minio_settings"])
    for model_configuration in settings["forecast_models"]:
        logging.info("Setup model of type {}".format(model_configuration["type"]))
        forecaster_in_q = queue.Queue()
        forecaster_in_qs.append(forecaster_in_q)
        forecaster_thread = setup_model(
            config=model_configuration,
            minio_client=minio_client,
            platform_sensor_publisher=platform_sensor_publisher,
            data_fetcher=data_fetcher,
            forecaster_in_q=forecaster_in_q,
            forecast_evaluator_in_q=forecast_evaluator_in_q,
        )
        all_threads.append(forecaster_thread)

    timer_thread = TimerThread(event_out_qs=forecaster_in_qs)
    all_threads.append(timer_thread)

    for thread in all_threads:
        thread.start()

    for thread in all_threads:
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
