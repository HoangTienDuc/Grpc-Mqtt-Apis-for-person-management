from .triton_backend import TritonIS
import numpy as np


class WebfaceR50(TritonIS):
    INPUT_MEAN = 127.5
    INPUT_STD = 127.5
    def __init__(self, arcface_config):
        super().__init__(arcface_config)
    
    def preprocess_batch(self, list_batches):
        """
        Preprocess a batch of face images for the WebfaceR50 model.

        This function takes a list of batches of face images and applies the necessary preprocessing steps
        to each image in the batch. The preprocessing steps include transposing the image dimensions,
        subtracting the mean, dividing by the standard deviation, and converting the image to a float32
        data type.

        Parameters:
        list_batches (list): A list of batches of face images. Each batch is a list of face images,
                            where each face image is a numpy array with shape (height, width, channels).

        Returns:
        list: A list of preprocessed batches. Each preprocessed batch is a list of preprocessed face images,
            where each preprocessed face image is a numpy array with shape (channels, height, width)
            and data type np.float32.
        """
        return [self.preprocess(img) for img in list_batches]
    
    def preprocess(self, batch):
        """
        Preprocess a batch of face images for the WebfaceR50 model.

        This function takes a batch of face images and applies the necessary preprocessing steps
        to each image in the batch. The preprocessing steps include transposing the image dimensions,
        subtracting the mean, dividing by the standard deviation, and converting the image to a float32
        data type.

        Parameters:
        batch (list): A batch of face images. Each face image is a numpy array with shape (height, width, channels).

        Returns:
        list: A list of preprocessed face images. Each preprocessed face image is a numpy array with shape (channels, height, width)
            and data type np.float32.
        """
        processed_batch = []
        for face_img in batch:
            face_img = np.transpose(face_img, (2, 0, 1))
            face_img = np.subtract(face_img, self.INPUT_MEAN, dtype=np.float32)
            face_img = np.multiply(face_img, 1 / self.INPUT_STD)
            face_img = face_img.astype(np.float32)
            processed_batch.append(face_img)
        return processed_batch

    def run(self, list_batches, user_data):
        """
        Run the inference on a batch of preprocessed face images using the Triton Inference Server.

        This function takes a list of batches of preprocessed face images and user data,
        executes the inference using the Triton Inference Server, and returns the results.

        Parameters:
        list_batches (list): A list of batches of preprocessed face images. Each batch is a list of preprocessed face images,
                            where each preprocessed face image is a numpy array with shape (channels, height, width)
                            and data type np.float32.
        user_data (object): Additional user-defined data to be passed to the inference function.

        Returns:
        numpy.ndarray: The results of the inference. The shape and content of the results depend on the specific model.
        """
        processed_batch = [self.preprocess(batch) for batch in list_batches]
        responses = self.execute(processed_batch, user_data)
        results = []
        for idx, response in enumerate(responses):
            result = response.as_numpy(self.output_names[0]) # self.output_names = ['683'] 
            if idx == 0:
                results = result
            else:
                results = np.vstack((results, result))
        return results