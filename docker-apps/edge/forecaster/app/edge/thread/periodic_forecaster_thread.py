import pickle
import queue
import threading
from collections import deque

import urllib3
from iotlab_utils.data_manager import TIME_COLUMN
from minio import Minio

from edge.util.data_initializer import DataInitializer


class PeriodicForecasterThread(threading.Thread):

    forecast_sensor = "forecast"
    accuracy_metrics_sensors = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "IAS"]

    def __init__(
        self,
        event_in_q: queue.Queue,
        event_out_q: queue.Queue,
        accuracy_results_out_q: queue.Queue,
        data_initializer: DataInitializer,
        minio_client: Minio,
        model_bucket: str,
        model_blob_name: str,
    ):
        super().__init__()
        self.event_in_q = event_in_q
        self.event_out_q = event_out_q
        self.accuracy_results_out_q = accuracy_results_out_q
        self.data_initializer = data_initializer
        self.minio_client = minio_client
        self.model_bucket = model_bucket
        self.model_blob_name = model_blob_name

    def run(self):
        sequence_length = 0
        latest_data = None
        latest_forecasts = deque([])
        while True:

            pred_time = self.event_in_q.get()

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
            latest_data = latest_data.append({TIME_COLUMN: pred_time}, ignore_index=True)
            latest_data = model_fit.forecast(latest_data)
            latest_data_row = latest_data.iloc[-1:]
            forecast_value = int(latest_data_row.iloc[0]["count"])

            latest_forecasts.append((pred_time, forecast_value))

            # publish forecast result
            self.event_out_q.put((pred_time, forecast_value, self.forecast_sensor))

            # TODO: check if accuracy computation (online evaluation) is possible
            # TODO: how to get newest observation? continuous polling?
            # TODO: compute accuracies and publish results
            # TODO: remove oldest forecast element if evaluation was possible
            for acc_metric in self.accuracy_metrics_sensors:
                pass
            latest_forecasts.popleft()

            # Remove oldest sequence element
            latest_data = latest_data.iloc[1:]
