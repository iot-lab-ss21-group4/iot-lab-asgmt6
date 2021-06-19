from typing import Any, Dict

from kafka import KafkaProducer

FORECAST_MSG = "Forecast:{}"


class KafkaCountPublisher:
    def __init__(self, config: Dict[str, Any]):
        server = "{}:{}".format(config["message_broker_host"], config["message_broker_port"])
        print("Connect to: " + server)
        self.producer = KafkaProducer(bootstrap_servers=server)
        self.topic_name: str = config["message_broker_topic_name"]

    def publish(self, count: int):
        print("Send count {} to topic {}.".format(count, self.topic_name))
        self.producer.send(self.topic_name, FORECAST_MSG.format(count).encode("UTF-8"))
