import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from iotlab_utils.data_manager import TIME_COLUMN, time_series_interpolate
from util.accuracy import AccuracyCalculator
from util.data_fetcher import DataFetcher
from util.forecast_model_selector import ForecastModelSelector
from util.platform_sensor_publisher import PlatformSensorPublisher


class OfflineEvaluatorThread(threading.Thread):

    SENSOR_NAME = "bestOffline"
    REAL_COUNT_SENSOR_TYPE = "real"

    def __init__(
        self,
        event_in_q: queue.Queue,
        config_consumers: List[Dict[str, Any]],
        platform_sensor_publisher: PlatformSensorPublisher,
        accuracy_calculator: AccuracyCalculator,
        forecast_model_selector: ForecastModelSelector,
    ):
        super().__init__()
        self.evaluator_in_q = event_in_q
        self.platform_sensor_publisher = platform_sensor_publisher
        self.model_data_fetchers = self.create_data_fetchers(config_consumers)
        self.accuracy_calculator = accuracy_calculator
        self.forecast_model_selector = forecast_model_selector
        self.y_column: Optional[str] = None

    def create_data_fetchers(self, config_consumers: List[Dict[str, Any]]) -> Dict[str, DataFetcher]:
        data_fetchers = {}
        for config_consumer in config_consumers:
            model_name, consumer_id = config_consumer["sensor_type"], config_consumer["sensor_id"]
            data_fetcher = DataFetcher.from_inputs(
                config_consumer["iot_platform_consumer_host"], consumer_id, config_consumer["iot_platform_consumer_key"]
            )
            data_fetchers[model_name] = data_fetcher
        assert self.REAL_COUNT_SENSOR_TYPE in data_fetchers.keys()
        return data_fetchers

    def interpolate_real_counts_to_forecasts(self, real_counts_df: pd.DataFrame, forecast_df: pd.DataFrame) -> List[int]:
        real_counts = (
            time_series_interpolate(
                real_counts_df[TIME_COLUMN].to_numpy(),
                real_counts_df[self.y_column].to_numpy(),
                np.array(forecast_df[TIME_COLUMN]),
            )
            .round()
            .astype(np.int64)
        )
        return real_counts.tolist()

    def run(self):
        while True:
            lower_bound, upper_bound = self.evaluator_in_q.get()

            # fetch data
            model_forecast_frames: Dict[str, pd.DataFrame] = {}
            for model, data_fetcher in self.model_data_fetchers.items():
                if model == self.REAL_COUNT_SENSOR_TYPE:
                    continue
                targets, _ = data_fetcher.fetch(lower_bound=lower_bound, upper_bound=upper_bound)
                targets.drop_duplicates(subset=TIME_COLUMN, inplace=True)
                model_forecast_frames[model] = targets
            real_counts_df, self.y_column = self.model_data_fetchers[self.REAL_COUNT_SENSOR_TYPE].fetch(
                lower_bound=lower_bound, upper_bound=upper_bound
            )
            real_counts_df.drop_duplicates(subset=TIME_COLUMN, inplace=True)

            # only respect timestamp all forecast models have
            # take first model as reference
            model_1, forecasts_df_1 = list(model_forecast_frames.items())[0]
            for model, forecasts_df in model_forecast_frames.items():
                if model == model_1:
                    continue
                forecasts_df_1 = forecasts_df_1[forecasts_df_1[TIME_COLUMN].isin(forecasts_df[TIME_COLUMN])]
            model_forecast_frames[model_1] = forecasts_df_1
            # now preprocess the others as well
            for model, forecasts_df in model_forecast_frames.items():
                if model == model_1:
                    continue
                forecasts_df = forecasts_df[forecasts_df[TIME_COLUMN].isin(forecasts_df_1[TIME_COLUMN])]
                model_forecast_frames[model] = forecasts_df

            # interpolate the real counts to one of the forecast lists. Use first one
            real_counts = self.interpolate_real_counts_to_forecasts(real_counts_df, forecasts_df_1)

            if len(real_counts) > 0:
                # perform accuracy calculation and strategy to find bestOffline
                model_accuracies = {}
                for model, forecast_ts in model_forecast_frames.items():
                    forecasts = forecast_ts[self.y_column]
                    assert len(forecasts) == len(real_counts)
                    accuracy = self.accuracy_calculator.compute_accuracy_metrics(real_counts=real_counts, forecasts=forecasts)
                    model_accuracies[model] = accuracy
                model_winner = self.forecast_model_selector.select(model_accuracies)

                logging.info("Winner is {}. Send forecasts from evaluation interval to platform".format(model_winner))

                # push the forecasts of the winner to the sensor
                for _, row in model_forecast_frames[model_winner].iterrows():
                    t, y = row[TIME_COLUMN], row[self.y_column]
                    self.platform_sensor_publisher.publish(self.SENSOR_NAME, t, y)
