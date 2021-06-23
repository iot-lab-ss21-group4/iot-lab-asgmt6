from typing import Any, Dict

import paho.mqtt.client as mqtt

FORECAST_MSG = "Forecast:{}"

delivered_records = 0


class EdgeBrokerSettings:
    def __init__(self, config: Dict[str, Any]):
        self.mqtt_broker_ip = config["edge_broker_host"]
        self.mqtt_broker_port = config["edge_broker_port"]
        self.mqtt_topic = config["edge_topic_name"]


class EdgeBrokerPublisher:
    def __init__(self, config: Dict[str, Any]):
        settings = EdgeBrokerSettings(config)
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        self.client.on_publish = on_publish
        self.client.connect(settings.mqtt_broker_ip, settings.mqtt_broker_port)
        self.topic = settings.mqtt_topic
        self.client.loop_start()

    def publish(self, forecast: int):
        print("Send count {} to topic {}.".format(forecast, self.topic))
        message = FORECAST_MSG.format(forecast)
        self.client.publish(self.topic, message, qos=2)


def on_connect(client: mqtt.Client, userdata: None, flags: Dict[str, int], rc: int):
    print("Connected. Result code " + str(rc))


def on_disconnect(client: mqtt.Client, userdata: None, rc: int):
    print("Disconnected. Result code " + str(rc))


def on_publish(client: mqtt.Client, userdata: None, rc: int):
    print("MQTT event published. Result code: {}.".format(rc))
