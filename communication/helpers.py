import base64
import cv2
import numpy as np


def cvt2img(base64_img):
    imgdata = base64.b64decode(base64_img)
    nparr = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cvt2base64(img):
    # img = base64.b64encode(cv2.imencode('.jpeg',  img)[1].tobytes())
    img = base64.b64encode(cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes())
    img = img.decode('ascii')
    return img