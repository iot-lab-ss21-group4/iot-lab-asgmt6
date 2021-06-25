import threading
import time
import unittest
from typing import Dict

import paho.mqtt.client as mqtt
from util.edge_broker_publisher import FORECAST_MSG, EdgeBrokerPublisher

test_forecast_message = ""


class TestMqttBrokerPublisher(unittest.TestCase):
    """
    Requires running mqtt broker on host OS
    """

    def setUp(self):
        global test_forecast_message
        test_forecast_message = ""
        self.topic = "TEST_TOPIC"
        self.forecast_result = 10
        self.expected_test_forecast_message = FORECAST_MSG.format(self.forecast_result)
        self.publisher = EdgeBrokerPublisher(
            config={
                "edge_broker_host": "localhost",
                "edge_broker_port": 1883,
                "edge_topic_name": self.topic,
            }
        )

    def test_receive_message(self):
        # Client to fetch forecast
        mqtt_client = mqtt.Client(client_id="test-client")
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.on_message = on_message
        mqtt_client.connect(host="localhost", port=1883)
        mqtt_client.subscribe(self.topic)
        # Start publisher
        forecast_publisher = threading.Thread(target=self.publisher.publish, args=(self.forecast_result,))
        forecast_publisher.start()
        # Fetch result
        for i in range(10):
            mqtt_client.loop()
            if test_forecast_message != "":
                break
            time.sleep(1)
        self.assertEqual(self.expected_test_forecast_message, test_forecast_message)


def on_connect(client: mqtt.Client, userdata: None, flags: Dict[str, int], rc: int):
    pass


def on_disconnect(client: mqtt.Client, userdata: None, rc: int):
    pass


def on_message(client: mqtt.Client, userdata: None, message):
    global test_forecast_message
    print("Subscriber: Message received")
    test_forecast_message = str(message.payload.decode("utf-8"))
