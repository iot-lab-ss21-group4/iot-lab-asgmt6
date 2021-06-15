import pickle
import queue
import threading
import time

import urllib3
from iotlab_utils.data_manager import TIME_COLUMN
from minio import Minio

from .data_initializer import DataInitializer
from .room_count_publisher import Publisher


class PeriodicForecasterThread(threading.Thread):
    def __init__(
        self,
        event_out_q: queue.Queue,
        data_initializer: DataInitializer,
        minio_client: Minio,
        model_bucket: str,
        model_blob_name: str,
        forecast_period: int = 300,
        forecast_dt: int = 900,
    ):
        super().__init__()
        self.event_out_q = event_out_q
        self.data_initializer = data_initializer
        self.minio_client = minio_client
        self.model_bucket = model_bucket
        self.model_blob_name = model_blob_name
        self.forecast_period = forecast_period
        self.forecast_dt = forecast_dt

        self._next_forecast_time = time.time() + self.forecast_period

    def run(self):
        sequence_length = 0
        latest_data = None
        while True:

            try:
                response: urllib3.HTTPResponse = self.minio_client.get_object(self.model_bucket, self.model_blob_name)
                model_fit = pickle.loads(response.data)
            finally:
                response.close()
                response.release_conn()

            # Initialize data
            if latest_data is None or model_fit.look_back_length != sequence_length:
                sequence_length = model_fit.look_back_length
                latest_data = self.data_initializer.initialize_data(sequence_length)

            # Calculate the next prediction time.
            pred_time = int(time.time()) + self.forecast_dt
            latest_data = latest_data.append({TIME_COLUMN: pred_time}, ignore_index=True)
            latest_data = model_fit.forecast(latest_data)
            latest_data_row = latest_data.iloc[-1:]
            forecast_value = int(latest_data_row.iloc[0]["count"])

            self.event_out_q.put((pred_time, forecast_value))

            # Remove oldest element
            latest_data = latest_data.iloc[1:]

            time.sleep(max(0.0, self._next_forecast_time - time.time()))
            self._next_forecast_time += self.forecast_period


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
