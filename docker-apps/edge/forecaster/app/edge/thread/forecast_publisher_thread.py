import queue
import threading

from edge.util.room_count_publisher import PlatformSensorPublisher


class ForecastPublisherThread(threading.Thread):
    def __init__(self, event_in_q: queue.Queue, publisher: PlatformSensorPublisher):
        super().__init__()
        self.event_in_q = event_in_q
        self.publisher = publisher

    def run(self):
        while True:
            t, y, sensor_name = self.event_in_q.get()
            # msec timestamps
            t = t * 1000
            self.publisher.publish(t, y, sensor_name)
