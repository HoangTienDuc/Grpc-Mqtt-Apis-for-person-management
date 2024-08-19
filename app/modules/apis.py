import queue
from collections import namedtuple
from sklearn.metrics.pairwise import euclidean_distances
from cfg.labels import SafetyConstructionLabels, StatusType

class UserData:
    def __init__(self):
        """
        Initialize the UserData class.

        This class is responsible for managing completed requests. It uses a queue to store completed requests.

        Attributes:
        _completed_requests (queue.Queue): A queue to store completed requests.

        Note:
        - The queue is initialized with a maximum size of 1000.
        """
        self._completed_requests = queue.Queue()

ImageInfo = namedtuple('ImageInfo', ['source_id', 'image'])

def is_duplicate_faces(embs):
    """
    Check if there are any duplicate faces in the given embeddings.

    Parameters:
    embs (list): A list of face embeddings. Each embedding is a 128-dimensional vector.

    Returns:
    bool: True if there are duplicate faces (i.e., embeddings with a Euclidean distance less than 0.05), False otherwise.
    """
    if len(embs) >= 2:
        for emb in embs:
            distances = euclidean_distances([emb], embs)
            for dis in distances[0]:
                if dis < 0.05:
                    return True
    return False

def is_the_same_person(embs):
    """
    Check if the given embeddings belong to the same person.

    This function calculates the pairwise Euclidean distances between the embeddings and checks if any maximum distance
    is less than 1. If so, it means that the embeddings belong to the same person.

    Parameters:
    embs (list): A list of face embeddings. Each embedding is a 128-dimensional vector.

    Returns:
    bool: True if the embeddings belong to the same person, False otherwise.
    """
    if len(embs) >= 2:
        distances = euclidean_distances(embs, embs)
        return any(max(row) < 1 for row in distances)
    return True

class UserApis:
    """
    This class provides APIs for user management and face recognition.

    Attributes:
    yolo_inference: An instance of YOLO inference engine.
    face_analysis: An instance of face analysis module.
    safety_construction_inference: An instance of safety construction inference engine.
    """
    def __init__(self, yolo_inference, face_analysis, safety_construction_inference):
        """
        Initialize the UserApis class with the provided inference engines.

        Parameters:
        yolo_inference (object): An instance of YOLO inference engine.
        face_analysis (object): An instance of face analysis module.
        safety_construction_inference (object): An instance of safety construction inference engine.
        """
        self.yolo_inference = yolo_inference
        self.face_analysis = face_analysis
        self.safety_construction_inference = safety_construction_inference
    
    def detect_face(self, images):
        """
        Detect faces in the given images using YOLO inference engine and face analysis module.

        Parameters:
        images (list): A list of input images for face detection.

        Returns:
        list: A list of detected faces. Each face is represented as an instance of ImageInfo.
        """
        user_data = UserData()
        batch_crops = self.yolo_inference.run(images, user_data)
        image_infos = [ImageInfo(source_id=index, image=crop) for index, batch_crop in enumerate(batch_crops) for crop in batch_crop]
        list_faces = self.face_analysis.detect_face(image_infos, user_data)
        return list_faces

    def add_update_user(self, aligns, user_label):
        """
        Adds or updates a user in the specified organization's classifier.

        Parameters:
        aligns (list): A list of aligned face images for the user.
        user_label (str): The label of the user in the format "name|id|organization".

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - NOT_FOUND: The specified organization does not exist.
            - DIFFERENCE_PERSON: The provided face images belong to different persons.
            - DUPLICATE_FACE: The provided face images contain duplicate faces.
            - EXISTED: The user already exists in the classifier.
        """
        user_data = UserData()
        organization = user_label.split("|")[2]
        classifier = self.face_analysis.classifiers.get_classifier(organization)
        if classifier is None:
            return False, StatusType.NOT_FOUND.value
        embeddings, normed_embeddings = self.face_analysis.get_embedding_from_align(aligns, user_data)
        if not is_the_same_person(normed_embeddings):
            return False, StatusType.DIFFERENCE_PERSON.value
        if is_duplicate_faces(normed_embeddings):
            return False, StatusType.DUPLICATE_FACE.value
        
        #TODO process exists
        searched_labels, distances = classifier.classify(normed_embeddings)
        for searched_label, distance in zip(searched_labels, distances):
            if searched_label is not None:
                if "|".join(user_label.split("|")[:3]) not in searched_label:
                    return False, StatusType.DIFFERENCE_PERSON.value
                elif distance < 0.01:
                    return False, StatusType.EXISTED.value
        classifier.register_user(aligns, normed_embeddings, user_label)
        return True, None

    def delete_user(self, user_label):
        """
        Deletes a user from the specified organization's classifier.

        Parameters:
        user_label (str): The label of the user in the format "name|id|organization".

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - NOT_FOUND: The specified organization does not exist.
            - OTHER: An unspecified error occurred during the deletion process.

        Note:
        - The user's face images are removed from the classifier.
        - The user's label is no longer associated with the organization.
        """
        organization = user_label.split("|")[2]
        classifier = self.face_analysis.classifiers.get_classifier(organization)
        if classifier is None:
            return False, StatusType.NOT_FOUND.value
        status, content = classifier.unregister_user(user_label)
        return status, content
    
    def change_user_name(self, old_user_label, new_user_label):
        """
        Changes the name of a user in the specified organization's classifier.

        Parameters:
        old_user_label (str): The current label of the user in the format "name|id|organization".
        new_user_label (str): The new label of the user in the format "name|id|organization".

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - NOT_FOUND: The specified organization does not exist.
            - OTHER: An unspecified error occurred during the name change process.

        Note:
        - The user's face images and other associated data remain unchanged.
        - The user's label is updated to the new label in the organization's classifier.
        """
        organization = old_user_label.split("|")[2]
        classifier = self.face_analysis.classifiers.get_classifier(organization)
        if classifier is None:
            return False, StatusType.NOT_FOUND.value
        status, content = classifier.change_user_label(old_user_label, new_user_label)
        return status, content
    
    def recognize(self, images, organization, is_recognize_safety=True):   
        """
        Recognize faces in the given images and optionally analyze safety-related attributes.

        Parameters:
        images (list): A list of input images for face recognition.
        organization (str): The organization for which the recognition is performed.
        is_recognize_safety (bool): A flag indicating whether to analyze safety-related attributes. Default is True.

        Returns:
        list: A list of recognized faces. Each face is represented as an instance of Face with optional safety-related attributes.
        """ 
        user_data = UserData()
        batch_crops = self.yolo_inference.run(images, user_data)
        image_infos = [ImageInfo(source_id=index, image=crop) for index, batch_crop in enumerate(batch_crops) for crop in batch_crop]
        _faces = self.face_analysis.analyse(image_infos, organization, user_data)
        if len(_faces) > 0:
            if is_recognize_safety:
                safety_labels_mapping = {
                    "HART_HAT": "hard_hat",
                    "SAFETY_VEST": "safety_vest",
                    "SAFETY_HARDNESS": "safety_hardness"
                }

                safety_construction_input_images = [face.body_image for face in _faces]
                safety_construction_outputs = self.safety_construction_inference.run(safety_construction_input_images, user_data)

                faces = [
                    face._replace(**{safety_labels_mapping[SafetyConstructionLabels(output.classID).name]: True 
                                    for output in outputs})
                    for face, outputs in zip(_faces, safety_construction_outputs)
                ]
            else:
                faces = _faces
        else:
            faces = []

        return faces
    
    def create_organization(self, organization):
        """
        Creates a new organization in the system.

        Parameters:
        organization (str): The name of the organization to be created.

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - EXISTED: The specified organization already exists.

        Note:
        - This function creates a new organization in the system, including a classifier for user management.
        """
        if self.face_analysis.classifiers.is_contain_organization(organization):
            return False, StatusType.EXISTED.value
        self.face_analysis.classifiers.create_classifier(organization)
        return True, None
    
    def remove_organization(self, organization):
        """
        Removes an organization from the system.

        Parameters:
        organization (str): The name of the organization to be removed.

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - NOT_FOUND: The specified organization does not exist.

        Note:
        - This function removes the organization from the system, including all associated user data.
        - The organization's classifier is also removed.
        """
        if not self.face_analysis.classifiers.is_contain_organization(organization):
            return False, StatusType.NOT_FOUND.value
        self.face_analysis.classifiers.remove_classifier(organization)
        return True, None
    
    def deactive_organization(self, organization):
        """
        Deactivates an organization in the system.

        Parameters:
        organization (str): The name of the organization to be deactivated.

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - NOT_FOUND: The specified organization does not exist.

        Note:
        - This function deactivates the organization in the system, preventing user management operations.
        - The organization's classifier remains intact.
        """
        if not self.face_analysis.classifiers.is_contain_organization(organization):
            return False, StatusType.NOT_FOUND.value
        self.face_analysis.classifiers.deactive_classifier(organization)
        return True, None
    
    def active_organization(self, organization):
        """
        Activates an organization in the system.

        Parameters:
        organization (str): The name of the organization to be activated.

        Returns:
        tuple: A tuple containing a boolean indicating success and a status message.
        - If successful, the boolean is True and the status message is None.
        - If unsuccessful, the boolean is False and the status message is one of the following:
            - EXISTED: The specified organization is already active.

        Note:
        - This function activates the organization in the system, allowing user management operations.
        - The organization's classifier remains intact.
        """
        if self.face_analysis.classifiers.is_contain_organization(organization):
            return False, StatusType.EXISTED.value
        self.face_analysis.classifiers.active_classifier(organization)
        return True, None
