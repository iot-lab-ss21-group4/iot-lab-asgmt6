import threading
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Tuple

from edge.util.accuracy import Accuracy, AccuracyCalculator
from edge.util.data_fetcher import DataFetcher
from edge.util.forecast_combiner import ForecastCombiner
from edge.util.kafka_count_publisher import KafkaCountPublisher
from edge.util.platform_sensor_publisher import PlatformSensorPublisher
from iotlab_utils.data_manager import TIME_COLUMN


class ForecastEvaluatorThread(threading.Thread):

    ACCURACY_SENSOR_PREFIX = "accuracy"
    BEST_ONLINE_SENSOR_NAME = "bestOnline"
    ACCURACY_METRIC_NAMES = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "IAS"]
    MODEL_COUNT = 3

    def __init__(
        self,
        event_in_q: Queue,
        platform_sensor_publisher: PlatformSensorPublisher,
        kafka_count_publisher: KafkaCountPublisher,
        data_fetcher: DataFetcher,
        number_of_models: int,
        accuracy_calculator: AccuracyCalculator,
        forecast_combiner: ForecastCombiner,
        max_acc_samples: int = 5,
    ):
        super().__init__()
        self.forecast_evaluator_in_q = event_in_q
        self.platform_sensor_publisher = platform_sensor_publisher
        self.kafka_count_publisher = kafka_count_publisher
        self.data_fetcher = data_fetcher
        self.number_of_accuracy_values = number_of_models * len(ForecastEvaluatorThread.ACCURACY_METRIC_NAMES)
        self.accuracy_calculator = accuracy_calculator
        self.forecast_combiner = forecast_combiner
        self.max_acc_samples = max_acc_samples

        self.target_buffer: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.forecast_buffer: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def track_model_forecasts(self, model_type: str, t: int, y: int):
        # TODO: update the target and forecast buffers and leave them in a valid state!
        self.forecast_buffer[model_type].append((t, y))
        earliest_forecast_t = self.forecast_buffer[model_type][0][0]
        relevant_targets, y_column = self.data_fetcher.fetch_time_window(lower_bound=earliest_forecast_t)
        earliest_target_t = relevant_targets.loc[relevant_targets.index[0], TIME_COLUMN]
        if relevant_targets.shape[0] == 0 or earliest_target_t > earliest_forecast_t:
            # TODO: load relevant_targets.shape[0] + 1 many target datapoints this time.
            pass

    def get_target_and_forecast_pairs(self) -> Tuple[List[int], List[int]]:
        # TODO: assuming the target and forecast buffers are in a valid state, return the longest list of target
        # and forecast values such that targets[0] <= forecasts[0] AND forecasts[-1] <= targets[-1].
        targets: List[int] = []
        forecasts: List[int] = []
        return targets, forecasts

    def run(self):
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
            # Get the target and forecast value pairs
            real_counts, forecasts = self.get_target_and_forecast_pairs()
            assert len(real_counts) == len(forecasts), "Target and forecast value pairs must be of equal size!"
            if len(real_counts) > 0:
                # Perform accuracy metric calculations, since we have some target and forecast values
                accuracy = self.accuracy_calculator.compute_accuracy_metrics(real_counts=real_counts, forecasts=forecasts)
                self.platform_sensor_publisher.publish(
                    ForecastEvaluatorThread.ACCURACY_SENSOR_PREFIX + model_type.upper(), t, accuracy._asdict()
                )
                evaluation_rounds[round_index][model_type] = (t, y, accuracy)

            # Check if all models submitted their forecasts for this evaluation round.
            if len(evaluation_rounds[round_index]) >= ForecastEvaluatorThread.MODEL_COUNT:
                # Note that 't' for all models at this evaluation round must be the same!
                best_y = self.forecast_combiner.combine(evaluation_rounds[round_index])
                self.platform_sensor_publisher.publish(ForecastEvaluatorThread.BEST_ONLINE_SENSOR_NAME, t, best_y)
                self.kafka_count_publisher.publish(best_y)
                del evaluation_rounds[round_index]
