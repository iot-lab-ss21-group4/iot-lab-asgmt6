from time import sleep
from typing import Dict, Any

from kafka import KafkaProducer

FORECAST_MSG = "Forecast:{}"


class ForecastMessageProducer:
    def __init__(self, config: Dict[str, Any]):
        server = "{}:{}".format(config["message_broker_host"], config["message_broker_port"])
        print("Connect to: " + server)
        self.producer = KafkaProducer(bootstrap_servers=server)
        self.topic_name = config["message_broker_topic_name"]

    def produce(self, count: int):
        self.producer.send(self.topic_name, FORECAST_MSG.format(count).encode("UTF-8"))
