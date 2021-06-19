import json
from typing import Any, Dict

import paho.mqtt.client as mqtt


class IotPlatformSettings:
    def __init__(self, data: Dict[str, Any]):
        self.iot_platform_gateway_username = data["iot_platform_gateway_username"]
        self.iot_platform_gateway_ip = data["iot_platform_gateway_ip"]
        self.iot_platform_gateway_port = data["iot_platform_gateway_port"]
        self.iot_platform_group_name = data["iot_platform_group_name"]
        self.iot_platform_sensor_name = data["iot_platform_sensor_name"]
        self.iot_platform_user_id = data["iot_platform_user_id"]
        self.iot_platform_device_id = data["iot_platform_device_id"]
        self.iot_platform_gateway_password = data["iot_platform_gateway_password"]


class PlatformSensorPublisher:
    def __init__(self, config: Dict[str, Any]):
        settings = IotPlatformSettings(config)
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        self.client.on_publish = on_publish
        self.client.username_pw_set(settings.iot_platform_gateway_username, settings.iot_platform_gateway_password)
        self.client.connect(settings.iot_platform_gateway_ip, port=settings.iot_platform_gateway_port)
        self.topic = str(settings.iot_platform_user_id) + "_" + str(settings.iot_platform_device_id)
        self.username = settings.iot_platform_group_name
        self.device_id = settings.iot_platform_device_id
        self.client.loop_start()

    def publish(self, sensor_name: str, timestamp: int, value: Any):
        message = (
            "{"
            + '"username":"{}","{}":{},"device_id":{},"timestamp":{}'.format(
                self.username,
                sensor_name,
                json.dumps(value),
                str(self.device_id),
                str(timestamp),
            )
            + "}"
        )
        print("Publishing '{}' on topic '{}'".format(message, self.topic))
        self.client.publish(self.topic, message, qos=2)


def on_connect(client: mqtt.Client, userdata: None, flags: Dict[str, int], rc: int):
    print("Connected. Result code " + str(rc))


def on_disconnect(client: mqtt.Client, userdata: None, rc: int):
    print("Disconnected. Result code " + str(rc))


def on_publish(client: mqtt.Client, userdata: None, rc: int):
    print("MQTT event published. Result code: {}.".format(rc))
