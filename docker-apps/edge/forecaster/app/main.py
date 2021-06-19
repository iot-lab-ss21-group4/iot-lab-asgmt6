import argparse
import json
import os
import queue
import threading
from typing import Any, Dict, List

from edge.minio_client import setup_minio_client
from edge.prepare_forecasting import setup_model
from edge.thread.best_online_forecaster_thread import BestOnlineForecasterThread
from edge.thread.timer_thread import TimerThread
from edge.util.kafka_count_publisher import KafkaCountPublisher
from edge.util.room_count_publisher import PlatformSensorPublisher


def setup(args: argparse.Namespace):
    with open(args.settings_file, "rt") as f:
        settings: Dict[str, Any] = json.load(f)

    # TODO: waiting for answer on moodle
    # create singleton mqtt publisher
    # under the assumption that one device (topic: username_deviceId) is sufficient
    platform_mqtt_publisher = PlatformSensorPublisher(settings["iot_platform_mqtt_settings"])

    all_threads: List[threading.Thread] = []

    accuracy_results_out_q = queue.Queue()
    kafka_count_publisher = KafkaCountPublisher(settings["message_broker_settings"])
    best_online_forecaster_thread = BestOnlineForecasterThread(
        event_in_q=accuracy_results_out_q,
        publisher=platform_mqtt_publisher,
        kafka_count_publisher=kafka_count_publisher,
        number_of_models=len(settings["forecast_models"]),
    )
    all_threads.append(best_online_forecaster_thread)

    periodic_forecaster_in_qs = []
    minio_client = setup_minio_client(settings["minio_settings"])
    for model_configuration in settings["forecast_models"]:
        periodic_forecaster_in_q = queue.Queue()
        periodic_forecaster_in_qs.append(periodic_forecaster_in_q)
        model_threads = setup_model(
            model_configuration,
            minio_client,
            platform_mqtt_publisher,
            periodic_forecaster_in_q,
            accuracy_results_out_q,
        )
        all_threads.extend(model_threads)

    timer_thread = TimerThread(event_out_qs=periodic_forecaster_in_qs)
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
