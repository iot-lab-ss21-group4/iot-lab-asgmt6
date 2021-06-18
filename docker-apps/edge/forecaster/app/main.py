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
from edge.util.forecast_message_producer import ForecastMessageProducer
from edge.util.room_count_publisher import setup_publisher


def setup(args: argparse.Namespace):
    with open(args.settings_file, "rt") as f:
        settings: Dict[str, Any] = json.load(f)

    minio_client = setup_minio_client(settings["minio_settings"])
    forecast_messages_producer = ForecastMessageProducer(settings["message_broker_settings"])

    all_threads: List[threading.Thread] = []

    # TODO: waiting for answer on moodle
    # create singleton mqtt publisher
    # under the assumption that one device (topic: username_deviceId) is sufficient
    mqtt_publisher, mqtt_thread = setup_publisher(settings["iot_platform_mqtt_settings"])
    all_threads.append(mqtt_thread)

    acuraccy_results_out_q = queue.Queue()
    best_online_forecaster_thread = BestOnlineForecasterThread(
        event_in_q=acuraccy_results_out_q,
        publisher=mqtt_publisher,
        message_producer=forecast_messages_producer,
        number_of_models=len(settings["forecast_models"]),
    )
    all_threads.append(best_online_forecaster_thread)

    periodic_forecaster_in_qs = []
    for model_configuration in settings["forecast_models"]:
        periodic_forecaster_in_q = queue.Queue()
        periodic_forecaster_in_qs.append(periodic_forecaster_in_q)
        model_threads = setup_model(
            model_configuration,
            minio_client,
            mqtt_publisher,
            periodic_forecaster_in_q,
            acuraccy_results_out_q,
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
