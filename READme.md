# HUMAN MANAGEMENT CLIENT
In this repo, config.ini or .txt format is not used to secure data. By using cython, .py code will be compiled to .so format.

## Table of contents
- Introduction
- Prerequires
- Installation 
- APIS
- How to use 
- STATUS CODE
- TODO


## INTRODUCTION
This repo demonstrates Mqtt API / sample python code for human management problem.


## PREREQUIRES
- Ubuntu 18.04
- Python 3.10
- Paho-mqtt 1.6.1
- GRPC
- Triton Inference Server
- Tensorrt


## INSTALLATION
- docker load -i api_client_wm:202423041326

## APIS
Currently includes 9 APIs. The implementation description of the APIs is as follows:
Step 1. First, when a new company/organization requests to use the product. The Web will send a request to create an organization.
Step 2: Add employees to the data for identification.

2.1: Step of extracting facial images: Send photos containing employees to the AI ​​side (the image can include many people), then the AI ​​side sends many images of the face cropped out of the image
2.2: The Web receives the faces, selects the face according to the user who wants to add. Then sends the cropped face/name and term to identify that employee to the AI ​​side.

Step 3: Identify employees: In this step, the Web sends any image + configuration to identify whether or not it is safe to work (identify wearing a hat or protective clothing), the AI ​​side will send the result of the name and corresponding image in the photo.

Step 4: In addition, add APIs such as delete user, edit user, delete organization, deactivate organization, activate organization.

The API section includes 2 topics, 1 topic for the web when sending requests, 1 topic for the AI ​​side to return responses


## HOW TO USE
```
docker run -it -d --net=host --name api_client_wm_container api_client_wm:202423041326

docker exec -it api_client_wm_container bash

cd /ws

python3 test_api_create_organization.py
python3 test_api_actve_organization.py
python3 test_api_detect_face.py
```

## STATUS CODE
```
NONE = 0
NOT_FOUND = 1 
EXISTED = 2 
DUPLICATE_FACE = 3
DIFFERENCE_PERSON = 4 
UNKNOWN = 5
```

## TODO
[ ] Add face antispoofing