import asyncio
import concurrent
from cfg.model_configs import retinaface_config, arcface_config, yolo_config, safety_construction_config
from app.modules.face_controller import FaceAnalysis
from app.modules.yolo_detector import YoloInference
from app.modules.safety_construction_detector import SafetyConstructionInference
from app.modules.apis import UserApis
from communication.mqtt_handler import MqttHandler

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()
    yolo_inference = YoloInference(yolo_config)
    safety_construction_inference = SafetyConstructionInference(safety_construction_config)
    face_analysis = FaceAnalysis(retinaface_config, arcface_config)
    user_apis = UserApis(yolo_inference, face_analysis, safety_construction_inference)
    mqtt_handler = MqttHandler(user_apis, loop)
    asyncio.ensure_future(
                loop.run_in_executor(executor, mqtt_handler.start), loop=loop
            )
    print("#"*59)
    print("#"*20 + " "*7 + "START" + " "*7 + "#"*20)
    print("#"*59)
    loop.run_forever()