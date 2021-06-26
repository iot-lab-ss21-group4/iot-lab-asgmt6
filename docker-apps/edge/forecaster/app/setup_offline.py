import argparse
import json
import logging
import os
import queue
import threading
from typing import Any, Dict, List

from offline.thread.interval_timer_thread import IntervalTimerThread
from offline.thread.offline_evaluator_thread import OfflineEvaluatorThread
from util.accuracy import AccuracyCalculator
from util.forecast_model_selector import ForecastModelSelector
from util.platform_sensor_publisher import PlatformSensorPublisher


def setup(args: argparse.Namespace):
    logging.info("Setup OFFLINE")
    logging.info("Reading the settings file '{}'".format(args.settings_file))
    with open(args.settings_file, "rt") as f:
        settings: Dict[str, Any] = json.load(f)

    platform_sensor_publisher = PlatformSensorPublisher(settings["iot_platform_mqtt_settings"])

    all_threads: List[threading.Thread] = []

    offline_evaluator_in_q = queue.Queue()
    offline_evaluator_thread = OfflineEvaluatorThread(
        event_in_q=offline_evaluator_in_q,
        config_consumers=settings["iot_platform_consumers"],
        platform_sensor_publisher=platform_sensor_publisher,
        accuracy_calculator=AccuracyCalculator(),
        forecast_model_selector=ForecastModelSelector.build_from_config(settings["forecast_model_select_strategy"]),
    )
    all_threads.append(offline_evaluator_thread)

    interval_timer_thread = IntervalTimerThread(event_out_q=offline_evaluator_in_q, interval=settings["interval"])
    all_threads.append(interval_timer_thread)

    for thread in all_threads:
        thread.start()

    for thread in all_threads:
        thread.join()


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--settings-file",
        type=str,
        default=os.path.join("offline-configuration", "settings.json"),
        help="Path to the settings file.",
    )
    parser.set_defaults(func=setup)
