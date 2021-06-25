import queue
import threading
from typing import Any, Dict

from minio import Minio

from edge.thread.forecaster_thread import ForecasterThread
from util.data_fetcher import DataFetcher
from util.platform_sensor_publisher import PlatformSensorPublisher


def setup_model(
    config: Dict[str, Any],
    minio_client: Minio,
    platform_sensor_publisher: PlatformSensorPublisher,
    data_fetcher: DataFetcher,
    forecaster_in_q: queue.Queue,
    forecast_evaluator_in_q: queue.Queue,
) -> threading.Thread:
    forecaster_thread = ForecasterThread(
        model_type=config["type"],
        event_in_q=forecaster_in_q,
        event_out_q=forecast_evaluator_in_q,
        platform_sensor_publisher=platform_sensor_publisher,
        data_fetcher=data_fetcher,
        minio_client=minio_client,
        model_bucket=config["model_bucket"],
        model_blob_name=config["model_blob_name"],
    )
    return forecaster_thread
