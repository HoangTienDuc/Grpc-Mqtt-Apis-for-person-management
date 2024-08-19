# import os
# from cfg.model_configs import yolo_config
from .triton_backend import TritonIS
from ..utils.yolo_helpers import preprocess, postprocess
from ..utils.yolo_helpers import *
from cfg.labels import COCOLabels
import cv2
import queue
import time

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

class InputInfo:
    id: int
    image: np.ndarray

class YoloInference(TritonIS):
    def __init__(self, yolo_config):
        self.yolo_config = yolo_config
        super().__init__(yolo_config)

    def run(self, list_img, user_data):
        """
        This function performs inference on a list of images using a YOLO model.
        It divides the images into batches, preprocesses each batch, executes the model,
        and processes the results to extract detected objects.
        If a detected object is a person, it crops the person from the image.

        Parameters:
        - list_img (List[np.ndarray]): A list of images to perform inference on.
        - user_data (UserData): User-defined data for tracking completed requests.

        Returns:
        - List[List[np.ndarray]]: A list of cropped person images from the input images.
        """
        crops = []
        max_batch_size = self.max_batch_size
        input_size = self.yolo_config.get("input_size")
        
        if len(list_img) > max_batch_size:
            list_batches = [list_img[i:i + max_batch_size] for i in range(0, len(list_img), max_batch_size)]
        else:
            list_batches = [list_img]
        
        list_batches_preproces_imgs = [self.preprocess_batch(batch_images, input_size) for batch_images in list_batches]
        list_batch_results = self.execute(list_batches_preproces_imgs, user_data)
        input_id = 0
        for batch_image, result in zip(list_batches, list_batch_results):
            if result is None:
                continue
            batch_num_dets, batch_det_boxes, batch_det_scores, batch_det_classes = [result.as_numpy(output_name) for output_name in self.output_names]
            

            for img, num_dets, det_boxes, det_scores, det_classes in zip(batch_image, batch_num_dets, batch_det_boxes, batch_det_scores, batch_det_classes):
                input_id += 1
                origin_img_shape = img.shape[:2]
                detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, origin_img_shape[1], origin_img_shape[0], [self.yolo_config.get("input_size")[1], self.yolo_config.get("input_size")[0]])
                # self.draw_on_image(detected_objects, img)
                crops.append(self.get_crops(detected_objects, img, origin_img_shape))
        return crops

    def preprocess_batch(self, batch_images, input_size):
        """
        Preprocess a batch of images for YOLO model inference.

        This function takes a list of images and an input size, and applies preprocessing steps
        to each image to prepare it for inference. The preprocessing steps include resizing the image
        to the specified input size and normalizing the pixel values.

        Parameters:
        - batch_images (List[np.ndarray]): A list of images to preprocess. Each image is a NumPy array.
        - input_size (Tuple[int, int]): The input size for the YOLO model. The images will be resized to this size.

        Returns:
        - List[np.ndarray]: A list of preprocessed images. Each preprocessed image is a NumPy array.
        """
        return [preprocess(img, input_size) for img in batch_images]

    def get_crops(self, detected_objects, img, image_shape):
        """
        Extracts and crops person images from the input image based on detected objects.

        This function iterates through the list of detected objects, checks if the object is a person,
        and then crops the person from the input image. The cropped person images are stored in a list.

        Parameters:
        - detected_objects (List[ObjectInfo]): A list of detected objects. Each object contains information about the detected object, such as classID, bounding box coordinates, and confidence score.
        - img (np.ndarray): The input image from which to extract the person images.
        - image_shape (Tuple[int, int]): The shape of the input image (height, width).

        Returns:
        - List[np.ndarray]: A list of cropped person images. Each cropped image is a NumPy array.
        """
        crops = []
        for object_info in detected_objects:
            if COCOLabels(object_info.classID).name == "PERSON":
                crop = img[max(object_info.y1, 0): min(object_info.y2, image_shape[0]), max(object_info.x1, 0): min(object_info.x2, image_shape[1])]
                crops.append(crop)
        return crops

    def draw_on_image(self, detected_objects, img):
        """
        Draws bounding boxes, class labels, and confidence scores on an input image.

        This function iterates through a list of detected objects, retrieves the class name, confidence score,
        and bounding box coordinates for each object. It then renders the bounding box, class label, and confidence
        score on the input image using the provided rendering functions. The resulting image is saved as a JPEG file.

        Parameters:
        - detected_objects (List[ObjectInfo]): A list of detected objects. Each object contains information about the detected object, such as classID, bounding box coordinates, and confidence score.
        - img (np.ndarray): The input image on which to draw the bounding boxes, class labels, and confidence scores.

        Returns:
        None. The function saves the resulting image as a JPEG file.
        """
        for object_info in detected_objects:
            class_name = COCOLabels(object_info.classID).name
            confidence = object_info.confidence
            color = tuple(RAND_COLORS[object_info.classID % 64].tolist())
            input_image = render_box(img, object_info.box(), color=color)
            size = get_text_size(input_image, f"{class_name}: {confidence:.2f}", normalised_scaling=0.6)
            input_image = render_filled_box(input_image, (object_info.x1 - 3, object_info.y1 - 3, object_info.x1 + size[0], object_info.y1 + size[1]), color=(220, 220, 220))
            input_image = render_text(input_image, f"{class_name}: {confidence:.2f}", (object_info.x1, object_info.y1), color=(30, 30, 30), normalised_scaling=0.5)
        cv2.imwrite(f"data/output_{str(int(time.time()))}.jpg", input_image)
