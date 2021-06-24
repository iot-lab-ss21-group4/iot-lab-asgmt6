import logging
import threading
from collections import defaultdict, deque
from queue import Queue
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.iotlab_utils.iotlab_utils.data_manager import TIME_COLUMN
from edge.util.accuracy import Accuracy, AccuracyCalculator
from edge.util.data_fetcher import DataFetcher
from edge.util.forecast_combiner import ForecastCombiner
from edge.util.edge_broker_publisher import EdgeBrokerPublisher
from edge.util.platform_sensor_publisher import PlatformSensorPublisher
from iotlab_utils.data_manager import time_series_interpolate


class ForecastEvaluatorThread(threading.Thread):

    ACCURACY_SENSOR_PREFIX = "accuracy"
    BEST_ONLINE_SENSOR_NAME = "bestOnline"
    ACCURACY_METRIC_NAMES = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "IAS"]
    ACCURACY_SENSOR_PATTERN = "accuracy{}-{}"

    def __init__(
        self,
        event_in_q: Queue,
        platform_sensor_publisher: PlatformSensorPublisher,
        edge_broker_publisher: EdgeBrokerPublisher,
        data_fetcher: DataFetcher,
        number_of_models: int,
        accuracy_calculator: AccuracyCalculator,
        forecast_combiner: ForecastCombiner,
        max_overlap_cnt: int = 5,
    ):
        super().__init__()
        self.forecast_evaluator_in_q = event_in_q
        self.platform_sensor_publisher = platform_sensor_publisher
        self.edge_broker_publisher = edge_broker_publisher
        self.data_fetcher = data_fetcher
        self.number_of_models = number_of_models
        self.accuracy_calculator = accuracy_calculator
        self.forecast_combiner = forecast_combiner
        self.max_overlap_cnt = max_overlap_cnt

        self.target_buffer: Dict[str, pd.DataFrame] = {}
        self.forecast_buffer: Dict[str, Deque[Tuple[int, int]]] = defaultdict(deque)
        self.overlap_cnt: int = 0
        self.y_column: Optional[str] = None

    def track_model_forecasts(self, model_type: str, t: int, y: int):
        self.forecast_buffer[model_type].append((t, y))
        earliest_forecast_t = self.forecast_buffer[model_type][0][0]
        biggest_earlier_target, self.y_column = self.data_fetcher.fetch(
            upper_bound=earliest_forecast_t, query_size=1, query_time_order="desc"
        )
        relevant_targets, _ = self.data_fetcher.fetch(
            lower_bound=biggest_earlier_target.loc[biggest_earlier_target.index[0], TIME_COLUMN]
        )

        # Make sure that there are at most self.max_acc_samples overlapping forecast values
        self.overlap_cnt = 0
        latest_relevant_target_t = relevant_targets.loc[relevant_targets.index[-1], TIME_COLUMN]
        for t, _ in self.forecast_buffer[model_type]:
            if t > latest_relevant_target_t:
                break
            self.overlap_cnt += 1
        excess_forecast_cnt = max(0, self.overlap_cnt - self.max_overlap_cnt)
        # Get rid of excessive overlapping forecast values
        for _ in range(excess_forecast_cnt):
            self.forecast_buffer[model_type].popleft()
        earliest_forecast_t = self.forecast_buffer[model_type][0][0]
        # Adjust and reduce the number of targets accordingly
        biggest_earlier_index = 0
        while biggest_earlier_index < relevant_targets.shape[0]:
            cur_t = relevant_targets.loc[relevant_targets.index[biggest_earlier_index], TIME_COLUMN]
            if cur_t > earliest_forecast_t:
                break
            biggest_earlier_index += 1
        biggest_earlier_index = max(0, biggest_earlier_index - 1)
        relevant_targets = relevant_targets.iloc[biggest_earlier_index:].copy()

        self.target_buffer[model_type] = relevant_targets

    def get_target_and_forecast_pairs(self, model_type: str) -> Tuple[List[int], List[int]]:
        # Assuming the target and forecast buffers are in a valid state, return the longest list of target
        # and forecast values such that targets[0] <= forecasts[0] AND forecasts[-1] <= targets[-1].
        if self.overlap_cnt <= 0:
            return [], []
        forecast_ts, forecast_ys = tuple(zip(*[t_y for t_y in self.forecast_buffer[model_type]]))
        target_values = (
            time_series_interpolate(
                self.target_buffer[model_type][TIME_COLUMN].to_numpy(),
                self.target_buffer[model_type][self.y_column].to_numpy(),
                np.array(forecast_ts),
            )
            .round()
            .astype(np.int64)
        )

        return target_values.tolist(), list(forecast_ys)

    def run(self):
        # TODO: consider evaluation_round as class -> difficult to understand what is what in the Dict
        evaluation_rounds: List[Dict[str, Tuple[int, int, Accuracy]]] = []
        while True:
            model_type: str
            t: int
            y: int
            model_type, t, y = self.forecast_evaluator_in_q.get()
            round_index = 0
            while round_index < len(evaluation_rounds) and model_type in evaluation_rounds[round_index]:
                round_index += 1
            if round_index == len(evaluation_rounds):
                evaluation_rounds.append(dict())

            # Get the target and forecast values ready for accuracy metric calculations
            self.track_model_forecasts(model_type, t, y)
            logging.info(
                "{}: Current forecasts [{}]".format(
                    model_type, " ".join(["( " + str(t) + " " + str(y) + " )" for (t, y) in self.forecast_buffer[model_type]])
                )
            )
            # Get the target and forecast value pairs
            real_counts, forecasts = self.get_target_and_forecast_pairs(model_type)
            assert len(real_counts) == len(forecasts), "Target and forecast value pairs must be of equal size!"
            logging.info(
                "{}: Pairs of real counts and forecasts [{}]".format(
                    model_type, " ".join(["( " + str(c) + " " + str(f) + " )" for (c, f) in zip(real_counts, forecasts)])
                )
            )
            if len(real_counts) > 0:
                # Perform accuracy metric calculations, since we have some target and forecast values
                accuracy = self.accuracy_calculator.compute_accuracy_metrics(real_counts=real_counts, forecasts=forecasts)
                for accuracy_metric_name in self.ACCURACY_METRIC_NAMES:
                    accuracy_sensor = self.ACCURACY_SENSOR_PATTERN.format(model_type.upper(), accuracy_metric_name)
                    self.platform_sensor_publisher.publish(accuracy_sensor, t, getattr(accuracy, accuracy_metric_name.lower()))
                evaluation_rounds[round_index][model_type] = (t, y, accuracy)

            # Check if all models submitted their forecasts for this evaluation round.
            if len(evaluation_rounds[round_index]) >= self.number_of_models:
                # Note that 't' for all models at this evaluation round must be the same!
                best_y = self.forecast_combiner.combine(evaluation_rounds[round_index])
                self.platform_sensor_publisher.publish(ForecastEvaluatorThread.BEST_ONLINE_SENSOR_NAME, t, best_y)
                self.edge_broker_publisher.publish(best_y)
                del evaluation_rounds[round_index]
