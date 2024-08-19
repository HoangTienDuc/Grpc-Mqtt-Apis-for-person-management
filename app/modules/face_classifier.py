import faiss
import os
import cv2
import numpy as np
from collections import Counter
import time
from cfg.labels import StatusType
import shutil

working_dir = os.getcwd()
classifiers_dir_path = os.path.join(working_dir, "user_data", "classifiers")
tmp_classifiers_dir_path = os.path.join(working_dir, "user_data", "tmp_classifiers")
registry_dir_path = os.path.join(working_dir, "user_data", "registry")

class Classifiers:
    def __init__(self):
        self._classifiers = {}
        if not os.path.exists(registry_dir_path):
            os.makedirs(registry_dir_path)
        if not os.path.exists(tmp_classifiers_dir_path):
            os.makedirs(tmp_classifiers_dir_path)
        if not os.path.exists(classifiers_dir_path):
            os.makedirs(classifiers_dir_path)
        else:
            organizations = os.listdir(classifiers_dir_path)
            for organization in organizations:
                self._classifiers[organization] = Classifier(organization)
    
    def is_contain_organization(self, organization):
        """
        Check if the given organization exists in the classifiers dictionary.

        Parameters:
        organization (str): The name of the organization to check.

        Returns:
        bool: True if the organization exists, False otherwise.
        """
        return organization in self._classifiers

    def create_classifier(self, organization):
        self._classifiers[organization] = Classifier(organization)

    def get_classifier(self, organization):
        if organization not in self._classifiers:
            return None
        return self._classifiers[organization]

    def remove_classifier(self, organization):
        if organization in self._classifiers:
            del self._classifiers[organization]
        shutil.rmtree(os.path.join(classifiers_dir_path, organization))
    
    def deactive_classifier(self, organization):
        if organization in self._classifiers:
            del self._classifiers[organization]
        shutil.move(os.path.join(classifiers_dir_path, organization), tmp_classifiers_dir_path)

    def active_classifier(self, organization):
        shutil.move(os.path.join(tmp_classifiers_dir_path, organization), classifiers_dir_path)
        self._classifiers[organization] = Classifier(organization)

    def reload_classifier_dataset(self, organization):
        if organization in self._classifiers:
            self._classifiers[organization].reload_classifier_dataset()


class Classifier:
    def __init__(self, organization):
        """
        Initialize a new instance of the Classifier class for a specific organization.

        Parameters:
        organization (str): The name of the organization for which the classifier is being initialized.

        Attributes:
        dir_about_user_registry_data (str): The path to the directory where user registry data is stored.
        dir_about_classifier_model (str): The path to the directory where the classifier model is stored.
        distance_threshold (float): The maximum distance allowed for a user to be considered a match.

        Methods:
        init_classifier(): Initializes the classifier model and loads existing data.
        """
        self.dir_about_user_registry_data = os.path.join(registry_dir_path, organization)
        self.dir_about_classifier_model = os.path.join(classifiers_dir_path, organization)
        if not os.path.exists(self.dir_about_user_registry_data):
            os.makedirs(self.dir_about_user_registry_data)
        if not os.path.exists(self.dir_about_classifier_model):
            os.makedirs(self.dir_about_classifier_model)
        self.distance_threshold = 1.0
        self.init_classifier()

    def init_classifier(self):
        self.index_path = os.path.join(
            self.dir_about_classifier_model, 'users.index')
        self.label_path = os.path.join(
            self.dir_about_classifier_model, 'users.npy')



    def is_exists(self, name_UUID):
        repeat_count = Counter(self.classifier_labels)[name_UUID]
        if repeat_count == 0:
            return True
        else:
            return False

    def register_user(self, aligns, embs, user_label):
        """ user: "name_UUID", "track_id", "emb", "crop", "align" """

        current_ntotal = len(self.classifier_labels)
        self.classifier_model.add_to_database_with_ids(np.array(embs).astype(
            'float32'), np.arange(current_ntotal, current_ntotal + 1))
        self.classifier_labels.append(user_label)
        self.resave_classifier()
        self.save_user_image_to_local(aligns, user_label)
    
    def unregister_user(self, user_label):
        remove_ids = np.array([index for index, saved_label in enumerate(
            self.classifier_labels) if user_label in saved_label])
        if len(remove_ids) > 0:
            self.classifier_model.remove_neighbors(remove_ids)
            for id in remove_ids:
                self.classifier_labels[id] = ""
            self.resave_classifier()
            return True, None
        else:
            return True, StatusType.NOT_FOUND.value
    
    def change_user_label(self, old_user_label, new_user_label):
        user_ids = np.array([index for index, saved_label in enumerate(
            self.classifier_labels) if old_user_label in saved_label])
        if len(user_ids) > 0:
            for id in user_ids:
                self.classifier_labels[id] = new_user_label
            self.resave_classifier()
            return True, None
        else:
            return True, StatusType.NOT_FOUND.value
        
    def resave_classifier(self):
        pass

    def save_user_image_to_local(self, aligns, user_name):
        user_dir = os.path.join(self.dir_about_user_registry_data, user_name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        for index, align in enumerate(aligns):
            user_image_path = os.path.join(
            user_dir, str(int(time.time())) + f"{str(index)}.jpg")
            image = cv2.cvtColor(align, cv2.COLOR_RGB2BGR)
            cv2.imwrite(user_image_path, image)

    def reload_classifier_dataset(self):
        if os.path.exists(self.index_path):
            self.classifier_model.load(self.index_path)
            self.classifier_labels = list(np.load(self.label_path))
    
    def classify(self, embs):
        pass

class FaissSearch:
    def __init__(self, d, k, use_gpu=False, add_with_ids=True):
        """
        Initialize the class with the dimension of vectors
        :param k: Number of neighbors to search
        :param d: dimension of the database and query vectors
        """
        self.d = d
        self.index = faiss.IndexFlatL2(self.d)
        self.add_with_ids = add_with_ids
        if self.add_with_ids:
            self.index = faiss.IndexIDMap2(self.index)
        self.use_gpu = use_gpu
        if self.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            self.convert_to_gpu()
            # self.index = faiss.GpuIndexFlatL2(res, self.d, flat_config)  # Does brute force neighbor search

        # self.index = faiss.IndexFlatIP(d)
        self.k = k

    def convert_to_gpu(self):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        self.index = faiss.index_cpu_to_gpu(res, flat_config.device, self.index)

    def load(self, index_path):
        if not os.path.exists(index_path):
            return False
        self.index = faiss.read_index(index_path)

        if self.use_gpu:
            self.convert_to_gpu()
        return True

    def save(self, index_path):
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def add_to_database(self, x):
        # x = x/LA.norm(x, axis=1, keepdims=True)
        if self.add_with_ids:
            raise SyntaxError("Missing id argument for data to be added. Use add_to_database_with_ids(...) "
                              "function instead or set add_with_ids flag to False at initialization")
        self.index.add(x)

    def add_to_database_with_ids(self, x, ids):
        if not self.add_with_ids:
            raise SyntaxError("Additional id argument for data to be added. Use add_to_database(...) "
                              "function instead or set add_with_ids flag to True at initialization")
        self.index.add_with_ids(x, ids)

    def search_neighbors(self, q):
        # q = q / LA.norm(q, axis=1, keepdims=True)
        return self.index.search(x=q, k=self.k)

    def get_neighbors(self, indices):
        return np.array([self.index.reconstruct(int(ind)) for ind in indices])

    def remove_neighbors(self, indices):
        self.index.remove_ids(indices)
    
    def get_ntotal(self, folder_name):
        self.load(folder_name)
        return self.index.ntotal