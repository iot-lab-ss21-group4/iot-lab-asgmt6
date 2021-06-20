import threading
from queue import Queue
from typing import Dict, List, Tuple

from edge.util.accuracy import AccuracyCalculator
from edge.util.kafka_count_publisher import KafkaCountPublisher
from edge.util.platform_sensor_publisher import PlatformSensorPublisher


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
        number_of_models: int,
        accuracy_calculator: AccuracyCalculator,
    ):
        super().__init__()
        self.forecast_evaluator_in_q = event_in_q
        self.platform_sensor_publisher = platform_sensor_publisher
        self.kafka_count_publisher = kafka_count_publisher
        self.number_of_accuracy_values = number_of_models * len(ForecastEvaluatorThread.ACCURACY_METRIC_NAMES)
        self.accuracy_calculator = accuracy_calculator

    def run(self):
        evaluation_rounds: List[Dict[str, Tuple[int, float, List[float]]]] = []
        while True:
            model_type, t, y = self.forecast_evaluator_in_q.get()
            round_index = 0
            while round_index < len(evaluation_rounds) and model_type in evaluation_rounds[round_index]:
                round_index += 1
            if round_index == len(evaluation_rounds):
                evaluation_rounds.append(dict())

            # TODO: check if accuracy computation (online evaluation) is possible
            # TODO: how to get newest observation? continuous polling?
            # TODO: compute accuracies and publish results
            # TODO: remove oldest forecast element if evaluation was possible
            acc_values = []
            for acc_metric in ForecastEvaluatorThread.ACCURACY_METRIC_NAMES:
                acc_value = None
                acc_values.append(acc_value)
            evaluation_rounds[round_index][model_type] = (t, y, acc_values)
            self.platform_sensor_publisher.publish(ForecastEvaluatorThread.ACCURACY_SENSOR_PREFIX + model_type, t, acc_values)

            if round_index >= ForecastEvaluatorThread.MODEL_COUNT:
                # Note that 't' for all models at this evaluation round must be the same!
                # TODO: Find best y using strategy - model independent
                best_y = None
                self.platform_sensor_publisher.publish(ForecastEvaluatorThread.BEST_ONLINE_SENSOR_NAME, t, best_y)
                self.kafka_count_publisher.publish(best_y)
