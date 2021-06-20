import argparse
import json
import os
import queue
import threading
from typing import Any, Dict, List

from edge.minio_client import setup_minio_client
from edge.prepare_forecasting import setup_model
from edge.thread.forecast_evaluator_thread import ForecastEvaluatorThread
from edge.thread.timer_thread import TimerThread
from edge.util.kafka_count_publisher import KafkaCountPublisher
from edge.util.platform_sensor_publisher import PlatformSensorPublisher


def setup(args: argparse.Namespace):
    with open(args.settings_file, "rt") as f:
        settings: Dict[str, Any] = json.load(f)

    # TODO: waiting for answer on moodle
    # create singleton mqtt publisher
    # under the assumption that one device (topic: username_deviceId) is sufficient
    platform_sensor_publisher = PlatformSensorPublisher(settings["iot_platform_mqtt_settings"])

    all_threads: List[threading.Thread] = []

    forecast_evaluator_in_q = queue.Queue()
    kafka_count_publisher = KafkaCountPublisher(settings["message_broker_settings"])
    forecast_evaluator_thread = ForecastEvaluatorThread(
        event_in_q=forecast_evaluator_in_q,
        platform_sensor_publisher=platform_sensor_publisher,
        kafka_count_publisher=kafka_count_publisher,
        number_of_models=len(settings["forecast_models"]),
        accuracy_calculator=settings["accuracy_calculator_settings"],
    )
    all_threads.append(forecast_evaluator_thread)

    forecaster_in_qs = []
    minio_client = setup_minio_client(settings["minio_settings"])
    for model_configuration in settings["forecast_models"]:
        forecaster_in_q = queue.Queue()
        forecaster_in_qs.append(forecaster_in_q)
        forecaster_thread = setup_model(
            model_configuration,
            minio_client,
            platform_sensor_publisher,
            forecaster_in_q,
            forecast_evaluator_in_q,
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
