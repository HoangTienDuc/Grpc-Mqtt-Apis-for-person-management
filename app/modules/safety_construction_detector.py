# import os
# from cfg.model_configs import safety_construction_config
from app.modules.triton_backend import TritonIS
from app.utils.safety_construction_helpers import preprocess, postprocess
from app.utils.safety_construction_helpers import *
from cfg.labels import SafetyConstructionLabels
import cv2
import queue
import time

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

class InputInfo:
    id: int
    image: np.ndarray

class SafetyConstructionInference(TritonIS):
    def __init__(self, yolo_config):
        self.yolo_config = yolo_config
        super().__init__(yolo_config)

    def run(self, list_img, user_data):
        """
        This function performs object detection on a list of images using a pre-trained model.
        It divides the list of images into batches based on the maximum batch size, preprocesses each batch,
        executes the model on the preprocessed batches, and postprocesses the results to obtain detected objects.

        Parameters:
        - list_img (List[np.ndarray]): A list of images on which object detection needs to be performed.
        - user_data (UserData): An object containing user-specific data.

        Returns:
        - list_detected_objects (List[List[ObjectInfo]]): A list of detected objects for each image in the input list.
        """
        list_detected_objects = []
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
                list_detected_objects.append(detected_objects)
        return list_detected_objects

    def preprocess_batch(self, batch_images, input_size):
        """
        Preprocess a batch of images for object detection.

        This function takes a list of images and an input size, and applies preprocessing steps to each image
        to prepare it for inference with a pre-trained model. The preprocessing steps include resizing the image
        to match the input size and normalizing the pixel values.

        Parameters:
        - batch_images (List[np.ndarray]): A list of images to be preprocessed. Each image is represented as a NumPy array.
        - input_size (Tuple[int, int]): The target size (width, height) for the preprocessed images.

        Returns:
        - List[np.ndarray]: A list of preprocessed images, each represented as a NumPy array.
        """
        return [preprocess(img, input_size) for img in batch_images]

    def draw_on_image(self, detected_objects, img):
        """
        Draws bounding boxes, class labels, and confidence scores on an input image.

        This function takes a list of detected objects and an input image, and draws bounding boxes, class labels,
        and confidence scores on the image. The function uses the provided rendering functions to achieve this.

        Parameters:
        - detected_objects (List[ObjectInfo]): A list of detected objects, where each object contains information such as
        class ID, confidence score, and bounding box coordinates.
        - img (np.ndarray): The input image on which the bounding boxes, class labels, and confidence scores will be drawn.

        Returns:
        - np.ndarray: The input image with bounding boxes, class labels, and confidence scores drawn on it. The image is saved
        as a JPEG file in the "data" directory with a unique timestamp in the filename.
        """
        for object_info in detected_objects:
            class_name = SafetyConstructionLabels(object_info.classID).name
            confidence = object_info.confidence
            color = tuple(RAND_COLORS[object_info.classID % 64].tolist())
            input_image = render_box(img, object_info.box(), color=color)
            size = get_text_size(input_image, f"{class_name}: {confidence:.2f}", normalised_scaling=0.6)
            input_image = render_filled_box(input_image, (object_info.x1 - 3, object_info.y1 - 3, object_info.x1 + size[0], object_info.y1 + size[1]), color=(220, 220, 220))
            input_image = render_text(input_image, f"{class_name}: {confidence:.2f}", (object_info.x1, object_info.y1), color=(30, 30, 30), normalised_scaling=0.5)
        cv2.imwrite(f"data/output_{str(int(time.time()))}.jpg", input_image)