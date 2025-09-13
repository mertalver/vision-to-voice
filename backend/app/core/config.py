from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BLIP_MODEL_PATH: str = "Salesforce/blip-image-captioning-base"
    YOLO_MODEL_PATH: str = "yolov5s.pt"
    QWEN_MODEL_PATH: str = "path/to/your/local/Qwen_Qwen3-4B-Q5_K_L.gguf"
    TTS_MODEL_PATH: str = "path/to/your/local/tts/model"

    class Config:
        case_sensitive = True

settings = Settings()