from .align import norm_crop
import collections
from ..utils.imagedata import ImageData
from ..utils.helpers import to_chunks, get_portrait
from numpy.linalg import norm
from .face_classifier import Classifiers
from .face_detector import RetinafaceMnetCov2
from .face_embedder import WebfaceR50

Face = collections.namedtuple("Face", ['source_id', 'user_label', 'bbox', 'landmark', 'det_score', 'embedding',
                                       'normed_embedding', 'mask_prob', 'portrait', 'align', 'body_image', 'hard_hat', 'safety_vest', 'safety_hardnes'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)



class FaceAnalysis:
    def __init__(self, retinaface_config, arcface_config):
        """
        Initialize the FaceAnalysis class with the provided configurations.

        Parameters:
        retinaface_config (dict): A dictionary containing configuration parameters for the RetinafaceMnetCov2 face detector.
        arcface_config (dict): A dictionary containing configuration parameters for the WebfaceR50 face embedder.

        Attributes:
        embedder (WebfaceR50): An instance of the WebfaceR50 face embedder.
        detector (RetinafaceMnetCov2): An instance of the RetinafaceMnetCov2 face detector.
        classifiers (Classifiers): An instance of the Classifiers class for managing face classifiers.
        face_detector_input_size (int): The input size for the face detector.
        max_rec_batch_size (int): The maximum batch size for face recognition operations.
        """
        self.embedder = WebfaceR50(arcface_config)
        self.detector = RetinafaceMnetCov2(retinaface_config)
        self.classifiers = Classifiers()
        self.face_detector_input_size = retinaface_config.get("input_size")
        self.max_rec_batch_size = 8

    def reproject_points(self, dets, scale: float):
        """
        Reproject the detected points based on the provided scale factor.

        Parameters:
        dets (numpy.ndarray): A numpy array representing the detected points.
        scale (float): The scale factor by which the points need to be reprojected.

        Returns:
        numpy.ndarray: The reprojected points after applying the scale factor.
        """
        if scale != 1.0:
            dets = dets / scale
        return dets

    def check_valid_face(self, pupil_distance, det_score, pupil_distance_thresh=20, detection_thresh=0.4):
        return pupil_distance >= pupil_distance_thresh and det_score >= detection_thresh

    def process_faces(self, user_data,  faces):
        """
        Check if the detected face is valid based on the provided pupil distance and detection score.

        Parameters:
        pupil_distance (float): The distance between the pupils of the detected face.
        det_score (float): The detection score of the detected face.
        pupil_distance_thresh (float, optional): The minimum pupil distance threshold for a face to be considered valid. Defaults to 20.
        detection_thresh (float, optional): The minimum detection score threshold for a face to be considered valid. Defaults to 0.4.

        Returns:
        bool: True if the face is valid based on the provided thresholds, False otherwise.
        """
        chunked_faces = to_chunks(faces, self.max_rec_batch_size)
        for chunk in chunked_faces:
            chunk = list(chunk)
            aligns = [face.align for face in chunk]
            embeddings = self.embedder.run(aligns, user_data)

            for face, embedding in zip(chunk, embeddings):
                embedding_norm = norm(embedding)
                normed_embedding = embedding / embedding_norm
                face = face._replace(embedding=embedding, embedding_norm=embedding_norm,
                                            normed_embedding=normed_embedding)
                yield face
    
    def get_face_infos(self, batch_image_data, batch_boxes, batch_landmarks, batch_probs, batch_mask_probs):
        """
        Process batch of detected faces to extract relevant information.

        Parameters:
        batch_image_data (list): A list of ImageData instances representing the input images.
        batch_boxes (list): A list of bounding boxes for detected faces in each image.
        batch_landmarks (list): A list of facial landmarks for detected faces in each image.
        batch_probs (list): A list of detection scores for detected faces in each image.
        batch_mask_probs (list): A list of mask probabilities for detected faces in each image.

        Returns:
        list: A list of Face namedtuples containing extracted face information.
        """
        list_faces = []
        for image_data, boxes, landmarks, probs, mask_probs in zip(batch_image_data, batch_boxes, batch_landmarks, batch_probs, batch_mask_probs):
            for bbox, landmark, det_score, mask_prob in zip(boxes, landmarks, probs, mask_probs):
                pupil_distance = abs(landmark[0][0] - landmark[1][0])
                if not self.check_valid_face(pupil_distance, det_score):
                    continue
                align = norm_crop(image_data.orig_image, landmark=landmark)
                body_image = image_data.orig_image
                portrait = get_portrait(body_image)
                face = Face(bbox=bbox, landmark=landmark, det_score=det_score,
                            mask_prob=mask_prob, align=align, source_id=image_data.source_id, portrait=portrait, body_image=body_image)
                list_faces.append(face)
        return list_faces

    def detect_face(self, image_infos, user_data):
        """
        Detect faces in the given list of image information.

        Parameters:
        image_infos (list): A list of ImageInfo instances, each containing an image and its source ID.
        user_data (dict): A dictionary containing user-specific data for processing.

        Returns:
        list: A list of Face namedtuples representing the detected faces. Each Face namedtuple contains information such as
        bounding box, landmarks, detection score, mask probability, aligned face image, source ID, portrait, and body image.
        """
        list_imgs = [ImageData(image_info.image, source_id=image_info.source_id, max_size=self.face_detector_input_size) for image_info in image_infos]
        batch_boxes = []
        batch_landmarks = []
        max_batch_size = self.detector.max_batch_size
        
        if len(list_imgs) > max_batch_size:
            list_batches = [list_imgs[i:i + max_batch_size] for i in range(0, len(list_imgs), max_batch_size)]
        else:
            list_batches = [list_imgs]
        batch_image_data, _batch_boxes, batch_probs, _batch_landmarks, batch_mask_probs = self.detector.run(list_batches, user_data)
        for image_data, _boxes, _landmarks in zip(batch_image_data, _batch_boxes, _batch_landmarks):
            boxes = [self.reproject_points(bbox, image_data.scale_factor) for bbox in _boxes]
            landmarks = [self.reproject_points(landmark, image_data.scale_factor) for landmark in _landmarks]
            batch_boxes.append(boxes)
            batch_landmarks.append(landmarks)
        list_faces = self.get_face_infos(batch_image_data, batch_boxes, batch_landmarks, batch_probs, batch_mask_probs)
        return list_faces

    def get_embedding(self, list_faces, user_data):
        """
        Extract face embeddings from a list of detected faces.

        This function takes a list of Face namedtuples and user-specific data as input.
        It extracts the aligned face images from the list of Face namedtuples,
        then processes these aligned face images to obtain face embeddings.
        The face embeddings are then normalized to have unit length.

        Parameters:
        list_faces (list): A list of Face namedtuples, where each Face namedtuple contains an aligned face image.
        user_data (dict): A dictionary containing user-specific data for processing.

        Returns:
        tuple: A tuple containing two lists:
            1. embeddings (list): A list of face embeddings extracted from the input list of Face namedtuples.
            2. normed_embeddings (list): A list of normalized face embeddings.
        """
        list_aligns = [face.align for face in list_faces]
        max_batch_size = self.embedder.max_batch_size
        if len(list_aligns) > max_batch_size:
            list_batches = [list_aligns[i:i + max_batch_size] for i in range(0, len(list_aligns), max_batch_size)]
        else:
            list_batches = [list_aligns]
        embeddings = self.embedder.run(list_batches, user_data)
        embedding_norms = [norm(embedding) for embedding in embeddings]
        normed_embeddings = [embedding / embedding_norm for embedding, embedding_norm in zip(embeddings, embedding_norms)]
        return embeddings, normed_embeddings

    def get_embedding_from_align(self, list_aligns, user_data):
        """
        Extract face embeddings from a list of aligned face images.

        This function takes a list of aligned face images and user-specific data as input.
        It processes these aligned face images to obtain face embeddings.
        The face embeddings are then normalized to have unit length.

        Parameters:
        list_aligns (list): A list of aligned face images. Each image is represented as a numpy array.
        user_data (dict): A dictionary containing user-specific data for processing.

        Returns:
        tuple: A tuple containing two lists:
            1. embeddings (list): A list of face embeddings extracted from the input list of aligned face images.
            2. normed_embeddings (list): A list of normalized face embeddings.
        """
        max_batch_size = self.embedder.max_batch_size
        if len(list_aligns) > max_batch_size:
            list_batches = [list_aligns[i:i + max_batch_size] for i in range(0, len(list_aligns), max_batch_size)]
        else:
            list_batches = [list_aligns]
        embeddings = self.embedder.run(list_batches, user_data)
        embedding_norms = [norm(embedding) for embedding in embeddings]
        normed_embeddings = [embedding / embedding_norm for embedding, embedding_norm in zip(embeddings, embedding_norms)]
        return embeddings, normed_embeddings
        

    def analyse(self, image_infos, organization, user_data):
        """
        Analyzes a list of image information to identify and classify detected faces.

        Parameters:
        image_infos (list): A list of ImageInfo instances, each containing an image and its source ID.
        organization (str): The organization for which the analysis is being performed.
        user_data (dict): A dictionary containing user-specific data for processing.

        Returns:
        list: A list of Face namedtuples representing the detected and classified faces.
        Each Face namedtuple contains information such as bounding box, landmarks, detection score, mask probability,
        aligned face image, source ID, portrait, body image, user label, and face embeddings.
        """
        classifier = self.classifiers.get_classifier(organization)
        if classifier is None:
            return []
        list_faces = self.detect_face(image_infos, user_data)
        list_embeddings, list_normed_embedding = self.get_embedding(list_faces, user_data)
        searched_labels, distances = classifier.classify(list_normed_embedding)
        list_faces = [face._replace(user_label=searched_label, embedding=embedding, normed_embedding=normed_embedding)
                      for face, searched_label, embedding, normed_embedding in zip(list_faces, searched_labels, list_embeddings, list_normed_embedding)]
        return list_faces