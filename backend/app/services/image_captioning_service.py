from PIL import Image
from app.core.config import settings
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageCaptioningService:
    def __init__(self, model_path):
        self.model_path = model_path
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        print(f"Image Captioning model loaded from {self.model_path}")

    def generate_caption(self, image: Image.Image) -> str:
        print("Generating caption for image...")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption


# Singleton instance
captioning_service = ImageCaptioningService(settings.BLIP_MODEL_PATH) 