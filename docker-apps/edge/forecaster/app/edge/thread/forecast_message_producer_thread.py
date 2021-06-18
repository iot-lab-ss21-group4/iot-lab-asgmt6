import queue
import threading

from edge.util.forecast_message_producer import ForecastMessageProducer


class ForecastMessageProducerThread(threading.Thread):
    def __init__(
        self,
        event_in_q: queue.Queue,
        forecast_message_producer: ForecastMessageProducer,
    ):
        super().__init__()
        self.event_in_q = event_in_q
        self.forecast_message_producer = forecast_message_producer

    def run(self):
        while True:
            _, y = self.event_in_q.get()
            self.forecast_message_producer.produce(y)
