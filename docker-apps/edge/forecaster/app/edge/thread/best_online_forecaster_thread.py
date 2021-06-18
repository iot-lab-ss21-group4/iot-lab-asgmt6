import threading
from queue import Queue

from edge.util.forecast_message_producer import ForecastMessageProducer
from edge.util.room_count_publisher import IotPlatformPublisher

from .periodic_forecaster_thread import PeriodicForecasterThread


class BestOnlineForecasterThread(threading.Thread):

    sensor_name = "bestOnline"

    def __init__(
        self,
        event_in_q: Queue,
        publisher: IotPlatformPublisher,
        message_producer: ForecastMessageProducer,
        number_of_models: int,
    ):
        super().__init__()
        self.event_in_q = event_in_q
        self.publisher = publisher
        self.message_producer = message_producer
        self.number_of_accuracy_values = number_of_models * len(PeriodicForecasterThread.accuracy_metrics_sensors)

    def run(self):
        results = []
        while True:
            t, y, accuracy = self.event_in_q.get()
            results.append((t, y, accuracy))
            if self.number_of_accuracy_values <= len(results):
                # TODO: Find best y using strategy - model independent
                t, best_y, _ = results[0]
                self.publisher.publish(self.sensor_name, t, best_y)
                self.message_producer.produce(best_y)
                results.clear()
