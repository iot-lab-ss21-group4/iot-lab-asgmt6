import queue
import threading

from edge.util.edge_broker_publisher import EdgeBrokerPublisher


class EdgeBrokerPublisherThread(threading.Thread):
    def __init__(
        self,
        event_in_q: queue.Queue,
        edge_broker_publisher: EdgeBrokerPublisher,
    ):
        super().__init__()
        self.event_in_q = event_in_q
        self.edge_broker_publisher = edge_broker_publisher

    def run(self):
        while True:
            _, y = self.event_in_q.get()
            self.edge_broker_publisher.publish(y)
