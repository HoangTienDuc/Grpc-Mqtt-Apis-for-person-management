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
        """
        Initialize the MqttHandler class.

        Parameters:
        apis (object): An instance of the APIs class that provides the necessary functionalities.
        loop (asyncio.AbstractEventLoop): The event loop to run the asynchronous operations. If not provided, the default event loop will be used.

        Attributes:
        _loop (asyncio.AbstractEventLoop): The event loop to run the asynchronous operations.
        client_id (str): The unique identifier for the MQTT client.
        apis (object): An instance of the APIs class.
        mqtt_client (mqtt.Client): The MQTT client instance.
        connected_flag (bool): A flag indicating whether the MQTT client is connected to the broker.
        MQTT_IP (str): The IP address of the MQTT broker.
        MQTT_PORT (int): The port number of the MQTT broker.
        MQTT_USER (str): The username for authentication with the MQTT broker.
        MQTT_PASS (str): The password for authentication with the MQTT broker.
        """
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
        """
        This function initializes and starts the MQTT client to connect to the MQTT broker.
        It sets up the necessary configurations, such as TLS settings, authentication, and event handlers.
        It also handles the reconnection logic in case of disconnection.

        Parameters:
        None

        Returns:
        None
        """
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
        """
        This function handles the MQTT client's connection event.

        Parameters:
        client (mqtt.Client): The MQTT client instance.
        userdata (object): User data provided when creating the client instance.
        flags (dict): Dictionary containing response flags sent by the broker.
        rc (int): The result code of the connection attempt. 0 indicates a successful connection.

        Returns:
        None
        """
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
        """
        This function processes incoming messages based on the specified command.
        If the command is "FR_DETECT_FACE", it detects faces in the images provided in the message,
        and appends the detected faces' portraits and aligned images to the message.
        If the command is "FR_ADD_UPDATE_USER", it adds or updates a user in the system using the images and user information provided in the message.

        Parameters:
        command (str): The command to be processed.
        message (dict): The incoming message containing the necessary data for processing.

        Returns:
        dict: The processed message with the results or status of the operation.
        """
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
        """
        This function handles incoming MQTT messages and processes them based on the command.
        If the command is "FR_DETECT_FACE", it detects faces in the images provided in the message,
        and appends the detected faces' portraits and aligned images to the message.
        If the command is "FR_ADD_UPDATE_USER", it adds or updates a user in the system using the images and user information provided in the message.
        In case of any exceptions, it logs the error and sends a response message with an unknown status.

        Parameters:
        client (mqtt.Client): The MQTT client instance.
        userdata (object): User data provided when creating the client instance.
        msg (mqtt.MQTTMessage): The incoming MQTT message.

        Returns:
        None
        """
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

