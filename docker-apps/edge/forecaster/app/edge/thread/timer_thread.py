import queue
import threading
import time
from typing import List


class TimerThread(threading.Thread):
    def __init__(
        self,
        event_out_qs: List[queue.Queue],
        forecast_period: int = 900,
        forecast_dt: int = 900,
    ):
        super().__init__()
        self.event_out_qs = event_out_qs
        self.forecast_period = forecast_period
        self.forecast_dt = forecast_dt

        self._next_forecast_time = time.time() + self.forecast_period

    def run(self):
        while True:
            pred_time = int(time.time()) + self.forecast_dt
            for q in self.event_out_qs:
                q.put(pred_time)
            time.sleep(max(0.0, self._next_forecast_time - time.time()))
            self._next_forecast_time += self.forecast_period
