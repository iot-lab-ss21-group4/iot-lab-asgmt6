import logging
import pickle
import queue
import threading

import pandas as pd
import urllib3
from util.data_fetcher import DataFetcher
from util.platform_sensor_publisher import PlatformSensorPublisher
from iotlab_utils.data_manager import DEFAULT_FLOAT_TYPE, TIME_COLUMN
from minio import Minio


class ForecasterThread(threading.Thread):

    SENSOR_PREFIX = "forecast"

    def __init__(
        self,
        model_type: str,
        event_in_q: queue.Queue,
        event_out_q: queue.Queue,
        platform_sensor_publisher: PlatformSensorPublisher,
        data_fetcher: DataFetcher,
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
        self.data_fetcher = data_fetcher
        self.minio_client = minio_client
        self.model_bucket = model_bucket
        self.model_blob_name = model_blob_name

    def run(self):
        while True:
            pred_time = self.forecaster_in_q.get()
            forecast_value = 0
            try:
                response: urllib3.HTTPResponse = self.minio_client.get_object(self.model_bucket, self.model_blob_name)
                # Pickle protocol version is the latest available for ibmfunctions/action-python-v3.7:master
                model_fit = pickle.loads(response.data)
            finally:
                response.close()
                response.release_conn()

            try:
                # Get the latest look back data for the forecast model.
                logging.info("{}: Get latest look back data. In total: {}".format(self.model_type, model_fit.look_back_length))
                latest_data, y_column = self.data_fetcher.fetch_latest(model_fit.look_back_length)
                model_fit.update_look_back_buffer(latest_data)

                # Calculate the next prediction time.
                forecast_data = model_fit.forecast(pd.DataFrame({TIME_COLUMN: [pred_time], y_column: [DEFAULT_FLOAT_TYPE()]}))
                if len(forecast_data.index) <= 0:
                    # if forecast model fails to forecast use 0 by default
                    forecast_data.append({TIME_COLUMN: pred_time, y_column: 0})
                forecast_value = int(forecast_data.loc[forecast_data.index[-1], y_column])
                forecast_value = min(45, max(0, forecast_value))
            except Exception as e:
                # catch exception to avoid breaking logic in forecast evaluator which always expects forecast value
                logging.error("{}: Forecasting failed! Reason: {}".format(self.model_type, str(e)))

            # publish forecast result
            self.platform_sensor_publisher.publish(self.sensor_name, pred_time, forecast_value)
            self.forecast_evaluator_in_q.put((self.model_type, pred_time, forecast_value))
