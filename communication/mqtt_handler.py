import asyncio
import paho.mqtt.client as mqtt
import socket
from .mqtt_helpers import AIOSocket
import json
from cfg.communication_config import mqtt_config
import time
from cfg.labels import StatusType
from .helpers import cvt2base64, cvt2img
import logging
logger = logging.getLogger('mqtt_subcriber')

class MqttHandler:
    def __init__(self, apis, loop):
        self._loop = loop or asyncio.get_event_loop()
        self.client_id = f'nano_mqttclient_{str(time.time())}'
        self.apis = apis
        self.mqtt_client = mqtt.Client(self.client_id, clean_session=True)
        self.connected_flag = False
        self.MQTT_IP = mqtt_config.get("MQTT_IP")
        self.MQTT_PORT = mqtt_config.get("MQTT_PORT") or 1883
        self.MQTT_USER = mqtt_config.get("MQTT_USER")
        self.MQTT_PASS = mqtt_config.get("MQTT_PASS")

    def set_api_reciever(self, api_reciever):
        self.api_reciever = api_reciever
    
    
    def start(self):
        if mqtt_config.get("IS_TLS"):
            ca_certs = mqtt_config.get("CA_CERTS")
            certfile = mqtt_config.get("CERTFILE")
            keyfile = mqtt_config.get("KEYFILE")
            ciphers = mqtt_config.get("CIPHERS")
            self.mqtt_client.tls_set(ca_certs=ca_certs, certfile=certfile, keyfile=keyfile, tls_version=mqtt.ssl.PROTOCOL_TLS, ciphers=ciphers)

        self.mqtt_client.username_pw_set(self.MQTT_USER, password=self.MQTT_PASS)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.socket_aio = AIOSocket(self._loop, self.mqtt_client, self.on_disconnect)

        while not self.connected_flag:
            try:
                self.mqtt_client.connect(self.MQTT_IP, self.MQTT_PORT, 60)
                time.sleep(5)  # Wait for 5 seconds before checking again
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)  # Wait for 5 seconds before checking again
        self.mqtt_client.socket().setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected_flag = True
            client.subscribe("avi/request/nano/api", qos=1)
            print(f"Mqtt client subcriber is connected to the mqtt broker at {self.MQTT_IP}")
        else:
            self.connected_flag = False
            print(f"Connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        if rc == 0:
            self.connected_flag = True
            print(f"Mqtt client subcriber is re-connected to the mqtt broker at {self.MQTT_IP}")
        else:
            self.connected_flag = False
            print(f"Disconnected with result code {rc}. Attempting to reconnect...")
            self.socket_aio.last_rc = rc
            # Your reconnect logic goes here
            asyncio.ensure_future(self.reconnect(client))
    
    async def reconnect(self, client):
        while not self.socket_aio.is_connected:
            try:
                print("Reconnecting...")
                client.reconnect()
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Reconnect failed: {str(e)}")
                await asyncio.sleep(5)

    def handle_msg(self, command, message):
        if command == "FR_DETECT_FACE":
            images = message.get("images")
            images = [cvt2img(image) for image in images]
            faces = self.apis.detect_face(images)
            results = [[]] * len(images)
            for face in faces:
                source_id = face.source_id
                results[source_id].append({"portrait": cvt2base64(face.portrait), "align": cvt2base64(face.align)})
            del message["images"]
            message["results"] = results
        elif command == "FR_ADD_UPDATE_USER":
            images = message.get("images")
            images = [cvt2img(image) for image in images]
            user_label = message.get("name") + "|" + message.get("uuid") + "|" + message.get("organization") + "|" + str(message.get("time_start")) + "|" + str(message.get("time_end"))
            status, content = self.apis.add_update_user(images, user_label)
            message["status"] = status
            message["content"] = content
        else:
            pass
        return message

    def on_message(self, client, userdata, msg):
        try:
            in_message = json.loads(msg.payload)
            command = in_message["command"]
            out_message = self.handle_msg(command, in_message)
            self.mqtt_client.publish("avi/response/nano/api", json.dumps(out_message), qos=1)
        except Exception as e:
            # logger.error(f"Error {str(e)}")
            print(f"Error {str(e)}")
            in_message["status"] = False
            in_message["content"] = StatusType.UNKNOWN.value
            self.mqtt_client.publish("avi/response/nano/api", json.dumps(in_message), qos=1)

