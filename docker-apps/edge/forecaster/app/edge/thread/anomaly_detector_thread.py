import logging
import pickle
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import urllib3
from common.iotlab_utils.iotlab_utils.data_manager import TIME_COLUMN
from iotlab_utils.data_manager import prepare_data_with_features
from minio import Minio
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from util.data_fetcher import DataFetcher
from util.platform_sensor_publisher import PlatformSensorPublisher


class AnomalyDetectorThread(threading.Thread):

    SENSOR_NAME = "is_anomaly"

    def __init__(
        self,
        platform_sensor_publisher: PlatformSensorPublisher,
        data_fetcher: DataFetcher,
        minio_client: Minio,
        model_bucket: str,
        model_blob_name: str,
        detection_period: int = 3600 * 24,
    ):
        super().__init__()
        self.platform_sensor_publisher = platform_sensor_publisher
        self.data_fetcher = data_fetcher
        self.minio_client = minio_client
        self.model_bucket = model_bucket
        self.model_blob_name = model_blob_name
        self.detection_period = detection_period

        self._now = int(time.time())
        self._next_detection_time = self._now + self.detection_period

    def run(self):
        while True:
            detection_upper_bound = self._now
            detection_lower_bound = detection_upper_bound - self.detection_period
            try:
                response: urllib3.HTTPResponse = self.minio_client.get_object(self.model_bucket, self.model_blob_name)
                # Pickle protocol version is the latest available for ibmfunctions/action-python-v3.7:master
                anomaly_detector_model: IsolationForest
                scaler: StandardScaler
                anomaly_detector_model, scaler = pickle.loads(response.data)
            finally:
                response.close()
                response.release_conn()

            logging.info(
                "Detecting anomalies for the time window between '{}' and '{}'".format(
                    datetime.utcfromtimestamp(detection_lower_bound), datetime.utcfromtimestamp(detection_upper_bound)
                )
            )
            ts, y_column = self.data_fetcher.fetch(detection_lower_bound, detection_upper_bound)
            _, x_columns, ts, _ = prepare_data_with_features(ts, detailed_seasonality=False, extra_features=False)
            data = ts.loc[:, x_columns + [y_column]]
            scaled_data = pd.DataFrame(scaler.transform(data))
            ts["is_anomaly"] = pd.Series(anomaly_detector_model.predict(scaled_data), index=ts.index, dtype=np.int64)
            ts["is_anomaly"].replace({-1: 1, 1: 0}, inplace=True)
            for t, is_anomaly in ts[[TIME_COLUMN, "is_anomaly"]].itertuples(index=False):
                self.platform_sensor_publisher.publish(AnomalyDetectorThread.SENSOR_NAME, t, is_anomaly)
            del ts, data, scaled_data

            # Wait until the next anomaly detection time
            self._now = self._next_detection_time
            time.sleep(max(0.0, self._next_detection_time - time.time()))
            self._next_detection_time += self.detection_period
