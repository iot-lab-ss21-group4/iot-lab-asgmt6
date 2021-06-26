import logging
import queue
import threading
import time

UNIT_DAY = "day"
UNIT_WEEK = "week"
INTERVAL_PATTERN = "{}-{}"  # e.g. day-3 -> after every three days we evaluate the forecasts from last three days


class IntervalTimerThread(threading.Thread):
    def __init__(
        self,
        event_out_q: queue.Queue,
        interval: str = "day-3",
    ):
        super().__init__()
        self.event_out_q = event_out_q
        self.interval_length = self.compute_interval_length(interval)
        self.evaluation_time_period = 3600 * 24

        self._next_evaluation_time = time.time() + self.evaluation_time_period

    def compute_interval_length(self, interval: str) -> int:
        assert "-" in interval
        unit, number = interval.split("-")[0], interval.split("-")[1]
        number = int(number)
        if unit.lower() == UNIT_DAY:
            return 3600 * 24 * number
        if unit.lower() == UNIT_WEEK:
            return 3600 * 24 * 7 * number
        raise Exception("Unit unknown")

    def run(self):
        while True:
            now = int(time.time())
            lower_time_bound, upper_time_bound = now - self.interval_length, now
            logging.info("Send interval [{}, {}] to offline evaluation.".format(str(lower_time_bound), str(upper_time_bound)))
            self.event_out_q.put((lower_time_bound, upper_time_bound))
            time.sleep(max(0.0, self._next_evaluation_time - time.time()))
            self._next_evaluation_time += self.evaluation_time_period
