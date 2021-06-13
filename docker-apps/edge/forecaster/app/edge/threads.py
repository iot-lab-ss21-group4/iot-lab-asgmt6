import queue
import threading
import time

from minio import Minio

from .room_count_publisher import Publisher


class PeriodicForecasterThread(threading.Thread):
    def __init__(
        self,
        event_out_q: queue.Queue,
        minio_client: Minio,
        model_bucket: str,
        model_blob_name: str,
        forecast_period: int = 300,
        forecast_dt: int = 900,
    ):
        super().__init__()
        self.event_out_q = event_out_q
        self.minio_client = minio_client
        self.model_bucket = model_bucket
        self.model_blob_name = model_blob_name
        self.forecast_period = forecast_period
        self.forecast_dt = forecast_dt

        self._next_forecast_time = time.time() + self.forecast_period

    def run(self):
        while True:
            model_fit = self.minio_client.get_object(self.model_bucket, self.model_blob_name)
            # Calculate the next prediction time.
            pred_time = int(time.time()) + self.forecast_dt
            # TODO call model and insert predicted number
            self.event_out_q.put((pred_time, 0))
            time.sleep(self.forecast_period)


class ForecastPublisherThread(threading.Thread):
    def __init__(self, event_in_q: queue.Queue, publisher: Publisher):
        super().__init__()
        self.event_in_q = event_in_q
        self.publisher = publisher

    def run(self):
        while True:
            t, y = self.event_in_q.get()
            # msec timestamps
            t = t * 1000
            self.publisher.publish(t, y)
