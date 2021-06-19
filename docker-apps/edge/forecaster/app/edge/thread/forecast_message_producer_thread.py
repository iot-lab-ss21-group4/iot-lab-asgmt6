import queue
import threading

from edge.util.kafka_count_publisher import KafkaCountPublisher


class KafkaCountPublisherThread(threading.Thread):
    def __init__(
        self,
        event_in_q: queue.Queue,
        kafka_count_publisher: KafkaCountPublisher,
    ):
        super().__init__()
        self.event_in_q = event_in_q
        self.kafka_count_publisher = kafka_count_publisher

    def run(self):
        while True:
            _, y = self.event_in_q.get()
            self.kafka_count_publisher.publish(y)
