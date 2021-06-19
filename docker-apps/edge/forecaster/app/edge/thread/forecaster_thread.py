import pickle
import queue
import threading
from collections import deque

import urllib3
from edge.util.data_initializer import DataInitializer
from edge.util.platform_sensor_publisher import PlatformSensorPublisher
from iotlab_utils.data_manager import TIME_COLUMN
from minio import Minio


class ForecasterThread(threading.Thread):

    SENSOR_PREFIX = "forecast"

    def __init__(
        self,
        model_type: str,
        event_in_q: queue.Queue,
        event_out_q: queue.Queue,
        platform_sensor_publisher: PlatformSensorPublisher,
        data_initializer: DataInitializer,
        minio_client: Minio,
        model_bucket: str,
        model_blob_name: str,
    ):
        super().__init__()
        self.model_type = model_type
        self.sensor_name = ForecasterThread.SENSOR_PREFIX + model_type.upper()
        self.forecaster_in_q = event_in_q
        self.forecast_evaluator_in_q = event_out_q
        self.platform_sensor_publisher = platform_sensor_publisher
        self.data_initializer = data_initializer
        self.minio_client = minio_client
        self.model_bucket = model_bucket
        self.model_blob_name = model_blob_name

    def run(self):
        sequence_length = 0
        latest_data, y_column = None, None
        latest_forecasts = deque([])
        while True:
            pred_time = self.forecaster_in_q.get()
            try:
                response: urllib3.HTTPResponse = self.minio_client.get_object(self.model_bucket, self.model_blob_name)
                # Pickle protocol version is the latest available for ibmfunctions/action-python-v3.7:master
                model_fit = pickle.loads(response.data)
            finally:
                response.close()
                response.release_conn()

            # Initialize data
            if latest_data is None or model_fit.look_back_length != sequence_length:
                sequence_length = model_fit.look_back_length
                latest_data, y_column = self.data_initializer.initialize_data(sequence_length)

            # Calculate the next prediction time.
            latest_data = latest_data.append({TIME_COLUMN: pred_time}, ignore_index=True)
            latest_data = model_fit.forecast(latest_data)
            latest_data_row = latest_data.iloc[-1:]
            forecast_value = int(latest_data_row.iloc[0][y_column])

            latest_forecasts.append((pred_time, forecast_value))

            # publish forecast result
            self.platform_sensor_publisher.publish(self.sensor_name, pred_time * 1000, forecast_value)
            self.forecast_evaluator_in_q.put((self.model_type, pred_time, forecast_value))

            latest_forecasts.popleft()

            # Remove oldest sequence element
            latest_data = latest_data.iloc[1:]
