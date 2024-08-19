from itertools import chain, islice
import base64
import cv2
import numpy as np
import logging
from app.modules.log_utils import logger_init
logger = logging.getLogger('helpers')

def get_portrait(body_image):
    height, width = body_image.shape[:2]
    if height > width:
        portrait = body_image[0: width, 0: width]
    else:
        portrait = body_image[0: height, 0: height]
    portrait = cv2.resize(portrait, (200, 200))
    return portrait

def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def cvt2base64(img):
    img = base64.encodebytes(cv2.imencode('.jpeg',  img)[1].tostring())
    img = img.decode('ascii')
    return img


def drawer(faces, frame):
    for face in faces:
        name = face.name
        bbox = face.bbox
        detection_score = face.det_score
        landmark = face.landmark
        mask_prob = face.mask_prob
        embedding_norm = face.embedding_norm

        if detection_score > 0.4:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        pt1 = tuple(map(int, bbox[0:2]))
        pt2 = tuple(map(int, bbox[2:4]))
        cv2.rectangle(frame, pt1, pt2, color, 1)
        # if mask_prob < 0.1:
        #     cv2.putText(frame, str(mask_prob), pt2, 0, 5e-3 * 100, (0,255,0), 1)
        # else:
        #     cv2.putText(frame, str(mask_prob), pt2, 0, 5e-3 * 100, (0,0,255), 1)
        cv2.putText(frame, name, pt1, 0, 5e-3 * 100, (0, 0, 255), 1)
        # cv2.putText(frame, str(track_id), pt1, 0, 5e-3 * 100, (0,255,0), 1)
        for point in landmark:
            point = list(map(int, point))
            cv2.circle(frame, (point[0], point[1]), 4, (255, 0, 255), -4)
    return frame

def cvt2img(base64_img):
    imgdata = base64.b64decode(base64_img)
    nparr = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def split_array(arr, sub_array_size):
    num_sub_arrays = len(arr) // sub_array_size
    sub_arrays = [arr[i * sub_array_size: (i + 1) * sub_array_size] for i in range(num_sub_arrays)]

    if len(arr) % sub_array_size != 0:
        sub_arrays.append(arr[num_sub_arrays * sub_array_size:])

    return sub_arrays