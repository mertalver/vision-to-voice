from typing import List
from fer import FER
import numpy as np
from PIL import Image
from app.core.config import settings
from collections import defaultdict, Counter

# İngilizce -> Türkçe duygu çeviri sözlüğü
EMOTION_MAP = {
    "neutral": "ifadesiz",
    "happy": "mutlu",
    "sad": "üzgün",
    "angry": "kızgın",
    "surprise": "şaşkın",
    "fear": "korkmuş",
    "disgust": "iğrenmiş"
}

class EmotionDetectionService:
    def __init__(self):
        self.detector = FER()

    def detect_emotions(self, image: Image.Image) -> List[str]:
        image_np = np.array(image)
        image_width, image_height = image.size
        faces = self.detector.detect_emotions(image_np)

        face_emotions = []

        for face in faces:
            emotions = face["emotions"]
            dominant_emotion_en = max(emotions, key=emotions.get)
            dominant_emotion_tr = EMOTION_MAP.get(dominant_emotion_en, dominant_emotion_en)

            (x, y, w, h) = face["box"]
            x_center = x + w / 2
            y_center = y + h / 2

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

            face_emotions.append(f"{vert}-{horiz} köşede bir kişi, yüz ifadesi: {dominant_emotion_tr}")

        return face_emotions

    def summarize_emotions(self, emotions_list: List[str]) -> str:
        positions = defaultdict(list)

        for e in emotions_list:
            if "yüz ifadesi:" in e:
                try:
                    pos_part, emotion_part = e.split("yüz ifadesi:")
                    pos = pos_part.strip().replace("köşede bir kişi", "").strip()
                    emotion = emotion_part.strip()
                    positions[pos].append(emotion)
                except:
                    continue

        summary_lines = []
        for pos, em_list in positions.items():
            most_common = Counter(em_list).most_common(1)[0][0]
            summary_lines.append(f"{pos} konumunda bir kişi, yüz ifadesi: {most_common}")

        return "\n".join(summary_lines) if summary_lines else "Resimde yüz ifadeleri tespit edilemedi."


# Singleton instance
emotion_detection_service = EmotionDetectionService()