import threading
from queue import Queue

from edge.util.kafka_count_publisher import KafkaCountPublisher
from edge.util.room_count_publisher import PlatformSensorPublisher

from .forecaster_thread import ForecasterThread


class BestOnlineForecasterThread(threading.Thread):

    SENSOR_NAME = "bestOnline"

    def __init__(
        self,
        event_in_q: Queue,
        publisher: PlatformSensorPublisher,
        kafka_count_publisher: KafkaCountPublisher,
        number_of_models: int,
    ):
        super().__init__()
        self.event_in_q = event_in_q
        self.publisher = publisher
        self.kafka_count_publisher = kafka_count_publisher
        self.number_of_accuracy_values = number_of_models * len(ForecasterThread.ACCURACY_METRIC_NAMES)

    def run(self):
        results = []
        while True:
            t, y, accuracy = self.event_in_q.get()
            results.append((t, y, accuracy))
            if self.number_of_accuracy_values <= len(results):
                # TODO: Find best y using strategy - model independent
                t, best_y, _ = results[0]
                self.publisher.publish(BestOnlineForecasterThread.SENSOR_NAME, t, best_y)
                self.kafka_count_publisher.publish(best_y)
                results.clear()
