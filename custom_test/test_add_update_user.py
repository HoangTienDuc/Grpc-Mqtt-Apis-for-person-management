import json
import base64
import paho.mqtt.client as mqtt
import cv2

def cvt2base64(img):
    img = base64.encodebytes(cv2.imencode('.jpeg',  img)[1].tostring())
    img = img.decode('ascii')
    return img

def create_detect_face_message(images):
    topic = "avi/request/nano/api"

    # Input message
    message = {
        "id": "123456",
        "command": "FR_ADD_UPDATE_USER",
        "name": "Tiến Đức",
        "uuid": "123456",
        "organization_id": "1627",
        "organization": "organization",
        "images": images,
        "time_start": 0, # Timestamp
        "time_end": 0
    }
    return topic, message


client = mqtt.Client(client_id="client_id", userdata=None, protocol=mqtt.MQTTv5)
# enable TLS for secure connection
client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
# set username and password
client.username_pw_set("iamev", "!@mM@ver!ck8")
# connect to HiveMQ Cloud on port 8883 (default for MQTT)
client.connect("6f5bc4afbaa54440b7bb1155e3c6f6de.s1.eu.hivemq.cloud", 8883)


image = cv2.imread("/develop/apis/user_data/registry/Tiến Đức|123456|location|0|0/17135372560.jpg")
base64_image = cvt2base64(image)
images = [base64_image]
topic, message = create_detect_face_message(images)
# Convert input message to JSON string
json_message = json.dumps(message)

# Publish message
client.loop_start()
client.publish(topic, json_message)
# client.loop_forever()
client.loop_stop()