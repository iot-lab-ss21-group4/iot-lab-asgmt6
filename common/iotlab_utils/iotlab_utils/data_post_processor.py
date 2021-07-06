from typing import List, Dict, Any


class ProcessInterval:
    def __init__(self, lower_bound_ms: int, upper_bound_ms: int, delta: int):
        self.lower_bound_ms = lower_bound_ms
        self.upper_bound_ms = upper_bound_ms
        self.delta = delta


class DataPostProcessor:
    def __init__(self, process_commands: List[Dict[str, Any]]):
        self.intervals = []
        for pc in process_commands:
            process_interval = ProcessInterval(pc["lower_bound"], pc["upper_bound"], pc["delta"])
            self.intervals.append(process_interval)

    def apply(self, timestamp_ms: int, value: int) -> int:
        for i in self.intervals:
            if i.lower_bound_ms <= timestamp_ms <= i.upper_bound_ms:
                return value + i.delta
        return value
