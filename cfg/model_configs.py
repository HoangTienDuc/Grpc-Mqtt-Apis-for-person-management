retinaface_config = {
    "model_name": "mnet_cov2",
    "model_version": "",
    "protocol": "GRPC",
    "url": "192.168.55.1:8001",
    "verbose": False,
    "streaming": False,
    "async_set": True,
    "batch_size": 1,
    "input_size": [320, 320],
    "threshold": 0.4,
    "nms_threshold": 0.45,
    "landmark_std": 1.0,
    "is_masks": True,
    "rac":'net3l'
}

arcface_config = {
    "model_name": "webface_r50",
    "model_version": "",
    "protocol": "GRPC",
    "url": "192.168.55.1:8001",
    "verbose": False,
    "streaming": False,
    "async_set": True,
    "batch_size": 1
}

yolo_config = {
    "model_name": "yolov7",
    "model_version": "",
    "protocol": "GRPC",
    "url": "192.168.55.1:8001",
    "verbose": False,
    "streaming": False,
    "async_set": True,
    "batch_size": 1,
    "input_size": [460, 640], # [height, width]
}

safety_construction_config = {
    "model_name": "construction_safety",
    "model_version": "",
    "protocol": "GRPC",
    "url": "192.168.55.1:8001",
    "verbose": False,
    "streaming": False,
    "async_set": True,
    "batch_size": 1,
    "input_size": [460, 640], # [height, width]
}