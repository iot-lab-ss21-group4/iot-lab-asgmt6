import queue
import threading
from typing import Any, Dict, List

from minio import Minio

from edge.thread.forecast_publisher_thread import ForecastPublisherThread
from edge.thread.forecaster_thread import ForecasterThread
from edge.util.data_initializer import DataInitializer
from edge.util.room_count_publisher import PlatformSensorPublisher


def setup_model(
    config: Dict[str, Any],
    minio_client: Minio,
    publisher: PlatformSensorPublisher,
    forecaster_in_q: queue.Queue,
    accuracy_results_out_q: queue.Queue,
) -> List[threading.Thread]:
    forecaster_out_q = queue.Queue()
    data_initializer = DataInitializer(config["iot_platform_consumer_settings"])
    forecaster_thread = ForecasterThread(
        event_in_q=forecaster_in_q,
        event_out_q=forecaster_out_q,
        accuracy_results_out_q=accuracy_results_out_q,
        data_initializer=data_initializer,
        minio_client=minio_client,
        model_bucket=config["model_bucket"],
        model_blob_name=config["model_blob_name"],
    )
    forecast_publisher = ForecastPublisherThread(event_in_q=forecaster_out_q, publisher=publisher)
    return [forecaster_thread, forecast_publisher]
