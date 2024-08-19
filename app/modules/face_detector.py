from .triton_backend import TritonIS
import cv2
import numpy as np
from ..utils.retinaface_helpers import *

class RetinafaceMnetCov2(TritonIS):
    def __init__(self, retinaface_config):
        self.retinaface_config = retinaface_config
        super().__init__(self.retinaface_config)
        # self.max_batch_size = 1

        self.masks = self.retinaface_config['is_masks']
        self.nms_threshold = self.retinaface_config['nms_threshold']

        _ratio = (1.,)
        fmc = 3
        
        self.rac = self.retinaface_config['rac']
        if self.rac == 'net3':
            _ratio = (1.,)
        elif self.rac == 'net3l':
            _ratio = (1.,)
            self.landmark_std = 0.2
        else:
            assert False, 'rac setting error %s' % self.rac

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }

        self.use_landmarks = True
        self.fpn_keys = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        self.anchor_plane_cache = {}

        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

    def preprocess(self, img):
        """
        Preprocess an input image for RetinaFace model.

        This function resizes the image, converts it to RGB format, and transposes the dimensions
        to match the expected input format for the RetinaFace model.

        Parameters:
        - img (Image): An Image object representing the input image.

        Returns:
        - np.ndarray: A NumPy array of shape (3, height, width) and dtype float32, representing the preprocessed image.
        """
        img.resize_image(mode='pad')
        im = cv2.cvtColor(img.transformed_image, cv2.COLOR_BGR2RGB)
        im = np.transpose(im, (2, 0, 1))
        # input_blob = np.expand_dims(im, axis=0).astype(np.float32)
        # return input_blob
        return im.astype(np.float32)
    
    def postprocess(self, net_out, threshold):
        """
        Postprocess the output of the RetinaFace model to extract bounding boxes, scores, and optionally landmarks.

        Parameters:
        - net_out (list): A list of NumPy arrays representing the output of the RetinaFace model.
        - threshold (float): A threshold value for filtering bounding boxes based on their scores.

        Returns:
        - det (numpy.ndarray): A NumPy array containing the detected bounding boxes.
        - landmarks (numpy.ndarray): A NumPy array containing the facial landmarks for detected faces (if enabled).
        """
        for _idx, s in enumerate(self._feat_stride_fpn):
            net_out = threshold
        det, landmarks = None, None
        return det, landmarks
    
    def _sort_boxes(self, boxes, probs, mask_scores, landmarks, max_num=1):
        # Based on original InsightFace python package implementation
        if max_num > 0 and boxes.shape[0] > max_num:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            values = area  # some extra weight on the centering
            # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]

            boxes = boxes[bindex, :]
            probs = probs[bindex]
            mask_scores = mask_scores[bindex]

            landmarks = landmarks[bindex, :]

        return boxes, probs, mask_scores, landmarks

    def preprocess_batch(self, batch_images):
        """
        Preprocess a batch of images for RetinaFace model.

        This function takes a list of Image objects and applies preprocessing steps to each image.
        The preprocessing steps include resizing the image, converting it to RGB format, and transposing
        the dimensions to match the expected input format for the RetinaFace model.

        Parameters:
        - batch_images (list): A list of Image objects. Each Image object represents an input image.

        Returns:
        - list: A list of preprocessed images. Each preprocessed image is a NumPy array of shape (3, height, width)
        and dtype float32, representing the input image ready for inference.
        """
        return [self.preprocess(img) for img in batch_images]
    
    def run(self, list_batches, user_data):
        """
        This function processes a batch of images and performs object detection using a RetinaFace model.

        Parameters:
        - list_batches (list): A list of batches, where each batch is a list of Image objects.
        - user_data (any): Additional user-defined data that can be passed to the model.

        Returns:
        - batch_image_data (list): A list of Image objects, representing the processed images.
        - batch_boxes (list): A list of bounding boxes for detected objects in each image.
        - batch_probs (list): A list of confidence scores for the detected objects.
        - batch_landmarks (list): A list of facial landmarks for detected faces.
        - batch_mask_probs (list): A list of mask scores for detected objects (if masks are enabled).
        """
        list_batches_preproces_imgs = [self.preprocess_batch(batch_images) for batch_images in list_batches]
        responses = self.execute(list_batches_preproces_imgs, user_data)
        batch_image_data = []
        batch_boxes = []
        batch_probs = []
        batch_landmarks = []
        batch_mask_probs = []
        for batch_images, response in zip(list_batches, responses):
            results = [response.as_numpy(output_name) for output_name in self.output_names]
            for index, result in enumerate(zip(*results)):
                result = [np.expand_dims(element, axis=0) for element in result]
                bboxes, landmarks = self.postprocess(result, self.retinaface_config['threshold'])
                if bboxes is None:
                    continue
                boxes = bboxes[:, 0:4]
                probs = bboxes[:, 4]
                mask_probs = bboxes[:, 5]
                boxes, probs, mask_probs, landmarks = self._sort_boxes(
                    boxes, probs, mask_probs, landmarks)
                batch_boxes.append(boxes)
                batch_probs.append(probs)
                batch_landmarks.append(landmarks)
                batch_mask_probs.append(mask_probs)
                batch_image_data.append(batch_images[index])
        return batch_image_data, batch_boxes, batch_probs, batch_landmarks, batch_mask_probs