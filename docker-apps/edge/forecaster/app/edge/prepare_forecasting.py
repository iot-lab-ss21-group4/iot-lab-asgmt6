import queue
import threading
from typing import List

from edge.room_count_publisher import setup_publisher
from edge.threads import PeriodicForecasterThread, ForecastPublisherThread


def setup_model(data_json, minio_client):
    periodic_forecaster_out_q = queue.Queue()
    periodic_forecaster = PeriodicForecasterThread(
        event_out_q=periodic_forecaster_out_q,
        minio_client=minio_client,
        model_bucket=data_json["model_bucket"],
        model_blob_name=data_json["model_blob_name"],
    )
    publisher, mqtt_client = setup_publisher(data_json["iot_platform_mqtt_settings"])
    forecast_publisher = ForecastPublisherThread(event_in_q=periodic_forecaster_out_q, publisher=publisher)
    other_threads: List[threading.Thread] = [periodic_forecaster, mqtt_client]
    return forecast_publisher, other_threads
