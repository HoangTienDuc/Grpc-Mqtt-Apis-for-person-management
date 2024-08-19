import json
import base64
import paho.mqtt.client as mqtt
import cv2
# MQTT Broker settings



def cvt2base64(img):
    img = base64.encodebytes(cv2.imencode('.jpeg',  img)[1].tostring())
    img = img.decode('ascii')
    return img

def create_detect_face_message():
    topic = "avi/request/nano/api"

    # Input message
    message = {
        "id": "123456",
        "command": "CREATE_ORGANIZATION",
        "organization_id": "1627",
        "organization": "organization",
    }
    return topic, message


client = mqtt.Client(client_id="client_id", userdata=None)
# enable TLS for secure connection
client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
# set username and password
client.username_pw_set("iamev", "!@mM@ver!ck8")
# connect to HiveMQ Cloud on port 8883 (default for MQTT)
client.connect("6f5bc4afbaa54440b7bb1155e3c6f6de.s1.eu.hivemq.cloud", 8883)


topic, message = create_detect_face_message()
# Convert input message to JSON string
json_message = json.dumps(message)

# Publish message
client.publish(topic, json_message)
client.loop_forever()
