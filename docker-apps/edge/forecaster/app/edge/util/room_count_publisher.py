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
    def __init__(self, client: mqtt.Client, topic: str, username: str, device_id: int):
        self.client = client
        self.topic = topic
        self.username = username
        self.device_id = device_id
        self.client.loop_start()

    def publish(self, sensor_name: str, timestamp: int, count: int):
        message = (
            "{"
            + '"username":"{}","{}":{},"device_id":{},"timestamp":{}'.format(
                self.username,
                sensor_name,
                str(count),
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


def setup_publisher(config: Dict[str, Any]) -> PlatformSensorPublisher:
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    settings = IotPlatformSettings(config)
    mqtt_topic = str(settings.iot_platform_user_id) + "_" + str(settings.iot_platform_device_id)
    client.username_pw_set(settings.iot_platform_gateway_username, settings.iot_platform_gateway_password)
    client.connect(settings.iot_platform_gateway_ip, port=settings.iot_platform_gateway_port)
    publisher = PlatformSensorPublisher(
        client,
        mqtt_topic,
        settings.iot_platform_group_name,
        settings.iot_platform_device_id,
    )
    return publisher
