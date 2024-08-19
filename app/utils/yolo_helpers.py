import cv2
import numpy as np


_LINE_THICKNESS_SCALING = 500.0

np.random.seed(0)
RAND_COLORS = np.random.randint(50, 255, (64, 3), "int")  # used for class visu
RAND_COLORS[0] = [220, 220, 220]

def render_box(img, box, color=(200, 200, 200)):
    """
    Render a box. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color
    :return: updated image
    """
    x1, y1, x2, y2 = box
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_LINE_THICKNESS_SCALING * _LINE_THICKNESS_SCALING)
        )
    )
    thickness = max(1, thickness)
    img = cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness=thickness
    )
    return img

def render_filled_box(img, box, color=(200, 200, 200)):
    """
    Render a box. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color
    :return: updated image
    """
    x1, y1, x2, y2 = box
    img = cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness=cv2.FILLED
    )
    return img

_TEXT_THICKNESS_SCALING = 700.0
_TEXT_SCALING = 520.0


def get_text_size(img, text, normalised_scaling=1.0):
    """
    Get calculated text size (as box width and height)
    :param img: image reference, used to determine appropriate text scaling
    :param text: text to display
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: (width, height) - width and height of text box
    """
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)
        )
        * normalised_scaling
    )
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scaling, thickness)[0]


def render_text(img, text, pos, color=(200, 200, 200), normalised_scaling=1.0):
    """
    Render a text into the image. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param text: text to display
    :param pos: (x, y) - upper left coordinates of render position
    :param color: (b, g, r) - text color
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: updated image
    """
    x, y = pos
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)
        )
        * normalised_scaling
    )
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    size = get_text_size(img, text, normalised_scaling)
    cv2.putText(
        img,
        text,
        (int(x), int(y + size[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scaling,
        color,
        thickness=thickness,
    )
    return img

class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))

    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)

def preprocess(img, input_shape, letter_box=True):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

def postprocess(num_dets, det_boxes, det_scores, det_classes, img_w, img_h, input_shape, letter_box=True):
    boxes = det_boxes[:num_dets[0]] / np.array([input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype=np.float32)
    scores = det_scores[:num_dets[0]]
    classes = det_classes[:num_dets[0]].astype(int)

    old_h, old_w = img_h, img_w
    offset_h, offset_w = 0, 0
    if letter_box:
        if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
            old_h = int(input_shape[0] * img_w / input_shape[1])
            offset_h = (old_h - img_h) // 2
        else:
            old_w = int(input_shape[1] * img_h / input_shape[0])
            offset_w = (old_w - img_w) // 2

    boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
    if letter_box:
        boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
    boxes = boxes.astype(int)

    detected_objects = []
    for box, score, label in zip(boxes, scores, classes):
        detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img_w, img_h))
    return detected_objects