import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.utils import InferenceServerException


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    """
    Callback function for asynchronous inference requests.

    This function is called by the Triton Inference Server client when an inference
    request is completed. It places the result and error (if any) into a queue for
    further processing.

    Parameters:
    user_data (object): User-defined data passed to the callback function. In this case,
                        it is expected to be an instance of the TritonIS class.
    result (tritonclient.grpc.InferResult): The result of the completed inference request.
    error (Exception): An exception object if an error occurred during the inference request.
                       If no error occurred, this parameter will be None.

    Returns:
    None
    """
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))

def split_array(arr, sub_array_size):
    """
    Splits a given array into smaller sub-arrays of a specified size.

    This function divides the input array into multiple sub-arrays, each containing
    a specified number of elements. If the length of the input array is not an exact
    multiple of the sub-array size, the remaining elements are added to the last
    sub-array.

    Parameters:
    arr (list): The input array to be split.
    sub_array_size (int): The size of each sub-array.

    Returns:
    list: A list of sub-arrays, where each sub-array contains 'sub_array_size' elements.
          If the length of 'arr' is not an exact multiple of 'sub_array_size', the
          last sub-array may contain fewer elements.
    """
    num_sub_arrays = len(arr) // sub_array_size
    sub_arrays = [arr[i * sub_array_size: (i + 1) * sub_array_size] for i in range(num_sub_arrays)]

    if len(arr) % sub_array_size != 0:
        sub_arrays.append(arr[num_sub_arrays * sub_array_size:])

    return sub_arrays

class TritonIS:
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self._triton_client = grpcclient.InferenceServerClient(
            url=config.get("url"))
        self.max_batch_size, self.input_name, self.output_names, self.c, self.h, self.w, self.format, self.dtype = self.parse_model_grpc()
        
    def parse_model_grpc(self):
        """
        Parses the model configuration and metadata from the Triton Inference Server.

        Retrieves the model metadata and configuration using the Triton client.
        Extracts the input and output details, including the batch size, input name,
        output names, dimensions, format, and data type.

        Returns:
        tuple: A tuple containing the following elements:
            - max_batch_size (int): Maximum batch size for the model.
            - input_name (str): Name of the input tensor.
            - output_names (list): List of names of the output tensors.
            - c (int): Number of channels in the input tensor.
            - h (int): Height of the input tensor.
            - w (int): Width of the input tensor.
            - format (str): Format of the input tensor (e.g., NHWC, NCHW).
            - dtype (str): Data type of the input tensor.
        """
        model_metadata = self._triton_client.get_model_metadata(
                model_name=self.model_name)
        model_config = self._triton_client.get_model_config(
                model_name=self.model_name)
        
        input_metadata = model_metadata.inputs[0]
        input_config = model_config.config.input[0]
        # output_metadata = model_metadata.outputs[0]
        output_metadata_names = [output.name for output in model_metadata.outputs]
        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.config.max_batch_size > 0)

        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
        return (model_config.config.max_batch_size, input_metadata.name,
                output_metadata_names, c, h, w, input_config.format,
                input_metadata.datatype)

    def requestGenerator(self, batched_image_data):
        """
        A generator function to create inference requests for the Triton Inference Server.

        This function takes a batch of image data as input and yields a tuple containing
        the inputs and outputs for a single inference request. The inputs are set with
        the provided image data, and the outputs are requested for all the output tensors
        specified in the model configuration.

        Parameters:
        batched_image_data (numpy.ndarray): A batch of image data with shape (batch_size, height, width, channels)
                                        or (batch_size, channels, height, width) depending on the input format.

        Yields:
        tuple: A tuple containing the following elements:
            - inputs (list): A list containing one InferInput object representing the input tensor.
            - outputs (list): A list containing InferRequestedOutput objects representing the output tensors.

        """
        # Set the input data
        inputs = []
        inputs.append(
            grpcclient.InferInput(self.input_name, batched_image_data.shape, self.dtype))
        inputs[0].set_data_from_numpy(batched_image_data)

        outputs = [grpcclient.InferRequestedOutput(output_name) for output_name in self.output_names]     
        yield inputs, outputs

    def execute(self, image_datas, user_data):
        """
        Executes inference requests for a batch of image data using the Triton Inference Server.

        This function sends requests of `self.batch_size` images to the Triton server. If the number of
        images isn't an exact multiple of `self.batch_size`, it starts over with the first images until
        the batch is filled. The function handles asynchronous inference requests and collects the
        responses.

        Parameters:
        image_datas (list): A list of image data arrays with shape (height, width, channels) or
                            (channels, height, width) depending on the input format. Each array represents
                            a single image.
        user_data (object): User-defined data passed to the completion callback function. This object
                            should be an instance of the TritonIS class.

        Returns:
        list: A list of inference results. Each result is an instance of `tritonclient.grpc.InferResult`
            representing the output of a single inference request. The results are ordered based on
            their request IDs.
        """
        responses = [None] * len(image_datas)
        sent_count = 0
        async_requests = []

        for input_data in image_datas:
            # Send request
            for inputs, outputs in self.requestGenerator(
                    np.array(input_data)):
                try:
                    async_requests.append(self._triton_client.async_infer(
                        self.model_name,
                        inputs,
                        partial(completion_callback, user_data),
                        request_id=str(sent_count),
                        outputs=outputs))
                except InferenceServerException as e:
                    print("inference failed: " + str(e))
                sent_count += 1

        processed_count = 0
        while processed_count < sent_count:
            (results, error) = user_data._completed_requests.get()
            processed_count += 1
            if error is not None:
                continue
            responses[int(results.get_response().id)] = (results)
        return responses
        
        