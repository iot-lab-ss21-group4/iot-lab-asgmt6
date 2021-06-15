import queue
import threading
from typing import Any, Dict, List

from minio import Minio

from .room_count_publisher import setup_publisher
from .threads import ForecastPublisherThread, PeriodicForecasterThread
from .data_initializer import DataInitializer


def setup_model(config: Dict[str, Any], minio_client: Minio) -> List[threading.Thread]:
    periodic_forecaster_out_q = queue.Queue()
    data_initializer = DataInitializer(config["iot_platform_consumer_settings"])
    periodic_forecaster = PeriodicForecasterThread(
        event_out_q=periodic_forecaster_out_q,
        data_initializer=data_initializer,
        minio_client=minio_client,
        model_bucket=config["model_bucket"],
        model_blob_name=config["model_blob_name"],
    )
    publisher, mqtt_client = setup_publisher(config["iot_platform_mqtt_settings"])
    forecast_publisher = ForecastPublisherThread(event_in_q=periodic_forecaster_out_q, publisher=publisher)
    return [forecast_publisher, periodic_forecaster, mqtt_client]
