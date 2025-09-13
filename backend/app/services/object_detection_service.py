from PIL import Image
from typing import List, Dict
import numpy as np
from app.core.config import settings
from ultralytics import YOLO

LABEL_TRANSLATIONS = {
    "person": "kişi",
    "bicycle": "bisiklet",
    "car": "araba",
    "motorcycle": "motosiklet",
    "airplane": "uçak",
    "bus": "otobüs",
    "train": "tren",
    "truck": "kamyon",
    "boat": "tekne",
    "traffic light": "trafik ışığı",
    "fire hydrant": "yangın musluğu",
    "stop sign": "dur işareti",
    "parking meter": "parkmetre",
    "bench": "bank",
    "bird": "kuş",
    "cat": "kedi",
    "dog": "köpek",
    "horse": "at",
    "sheep": "koyun",
    "cow": "inek",
    "elephant": "fil",
    "bear": "ayı",
    "zebra": "zebra",
    "giraffe": "zürafa",
    "backpack": "sırt çantası",
    "umbrella": "şemsiye",
    "handbag": "çantası",
    "tie": "kravat",
    "suitcase": "valiz",
    "frisbee": "frizbi",
    "skis": "kayak",
    "snowboard": "snowboard",
    "sports ball": "spor topu",
    "kite": "uçurtma",
    "baseball bat": "bejzbol sopası",
    "baseball glove": "bejzbol eldiveni",
    "skateboard": "kaykay",
    "surfboard": "sörf tahtası",
    "tennis racket": "tenis raketi",
    "bottle": "şişe",
    "wine glass": "şarap bardağı",
    "cup": "bardak",
    "fork": "çatal",
    "knife": "bıçak",
    "spoon": "kaşık",
    "bowl": "kase",
    "banana": "muz",
    "apple": "elma",
    "sandwich": "sandviç",
    "orange": "portakal",
    "broccoli": "brokoli",
    "carrot": "havuç",
    "hot dog": "sosisli",
    "pizza": "pizza",
    "donut": "lokma",
    "cake": "kek",
    "chair": "sandalye",
    "couch": "kanepe",
    "potted plant": "saksı bitkisi",
    "bed": "yatak",
    "dining table": "yemek masası",
    "toilet": "tuvalet",
    "tv": "televizyon",
    "laptop": "dizüstü bilgisayar",
    "mouse": "fare",
    "remote": "kumanda",
    "keyboard": "klavye",
    "cell phone": "cep telefonu",
    "microwave": "mikrodalga fırın",
    "oven": "fırın",
    "toaster": "tost makinesi",
    "sink": "lavabo",
    "refrigerator": "buzdolabı",
    "book": "kitap",
    "clock": "saat",
    "vase": "vazo",
    "scissors": "makas",
    "teddy bear": "oyuncak ayı",
    "hair drier": "saç kurutma makinesi",
    "toothbrush": "diş fırçası"
}

class ObjectDetectionService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_objects(self, image: Image.Image) -> List[Dict]:
        image_np = np.array(image)
        image_width, image_height = image.size

        results = self.model(image_np)
        boxes = results[0].boxes

        detected_objects = []

        for box in boxes:
            class_id = int(box.cls.cpu().numpy())
            obj_name_en = results[0].names[class_id].lower()
            obj_name_tr = LABEL_TRANSLATIONS.get(obj_name_en, obj_name_en)

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            horiz = (
                "sol" if x_center < image_width / 3 else
                "sağ" if x_center > 2 * image_width / 3 else
                "ortada"
            )
            vert = (
                "üst" if y_center < image_height / 3 else
                "alt" if y_center > 2 * image_height / 3 else
                "ortada"
            )

            detected_objects.append({
                "label": obj_name_tr,
                "position": f"{vert}-{horiz}",
                "bbox": [x1, y1, x2, y2]
            })

        return detected_objects

    def format_objects_naturally(self, objects: List[Dict]) -> str:
        if not objects:
            return "Hiç nesne tespit edilmedi."
        return ", ".join(f"{obj['position']} konumunda bir {obj['label']}" for obj in objects)


# Singleton instance
object_detection_service = ObjectDetectionService(settings.YOLO_MODEL_PATH)
