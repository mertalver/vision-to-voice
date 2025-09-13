# Voice-Based Image Description

A **multimodal AI system** that generates natural image descriptions
using **captioning, object detection, emotion recognition, and
text-to-speech (TTS)**.
Designed to assist visually impaired users, the system analyzes images
and provides spoken descriptions with clear, concise, and context-aware
details.

------------------------------------------------------------------------

## ✨ Features

-   🖼 **Image Captioning** → Generates an initial caption using
    BLIP.
-   🔍 **Object Detection** → Identifies objects and their positions in
    the image (YOLOv5).
-   🙂 **Emotion Recognition** → Detects facial expressions and
    summarizes emotions.
-   🧠 **Text Generation (LLM)** → Combines captions, objects, and
    emotions to create a natural and accessible description using a Qwen-3
    model.
-   🔊 **Text-to-Speech** → Converts the generated description into
    speech (gTTS).
-   ⚡ **Real-Time API** → FastAPI-based REST API for easy
    integration.
-   📂 **Dataset Utilities** → Scripts to download sample datasets
    (Flickr1K, COCO2017).
-   🔑 **Customizable Models** → Configurable paths for BLIP, YOLO,
    Qwen-3, and TTS models.

------------------------------------------------------------------------

## 🛠 System Architecture

1.  **Frontend** → Simple HTML/JS interface for uploading images.
2.  **FastAPI Backend** (`main.py`) → Routes requests and orchestrates
    services.
3.  **Services**:
    -   `image_captioning_service.py` → BLIP-based
        captioning
    -   `object_detection_service.py` → YOLO-based detection
    -   `emotion_detection_service.py` → Facial emotion recognition
    -   `text_generation_service.py` → LLM-powered description
        synthesis
    -   `tts_service.py` → Converts description to audio
4.  **Endpoints** (`endpoints.py`) → `/api/describe-image/` pipeline:
    -   Caption → Object detection → Emotion recognition → LLM
        description → TTS audio
5.  **Utilities**:
    -   `get_dataset.py` → Downloads datasets for testing
    -   `config.py` → Model paths & settings

------------------------------------------------------------------------

## 📂 Project Structure

    ├── app/
    │   ├── main.py                     # FastAPI app
    │   ├── config.py                   # Model paths & config
    │   ├── endpoints.py                # API endpoints
    │   ├── services/
    │   │   ├── image_captioning_service.py  # BLIP captioning
    │   │   ├── object_detection_service.py  # YOLO detection
    │   │   ├── emotion_detection_service.py # FER detection
    │   │   ├── text_generation_service.py   # Qwen text generation
    │   │   └── tts_service.py               # gTTS speech
    ├── get_dataset.py                  # Dataset downloader
    ├── requirements.txt                # Dependencies
    ├── README.md                       # Documentation

------------------------------------------------------------------------

## ⚙️ Installation & Usage

### Prerequisites

-   Python 3.9+
-   PyTorch & Transformers
-   FastAPI & Uvicorn
-   YOLOv5 weights (`yolov5s.pt`)
-   BLIP model (e.g., `Salesforce/blip-image-captioning-base`)
-   Qwen-3 LLM model (`.gguf` format, configured in `config.py`)

### Setup

1.  Clone repository:

    ``` bash
    git clone https://github.com/yourusername/voice-based-image-description.git
    cd voice-based-image-description
    ```

2.  Create environment and install dependencies:

    ``` bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venvScriptsactivate    # Windows

    pip install -r requirements.txt
    ```

3.  Start API server:

    ``` bash
    uvicorn app.main:app --reload
    ```

4.  Access frontend at `http://localhost:8000` (index.html).

------------------------------------------------------------------------

## 🎮 Example API Usage

### Describe Image

``` http
POST /api/describe-image/
Content-Type: multipart/form-data
file: <image.jpg>
```

**Response** → Audio stream (`audio/wav`) with spoken description.

Example pipeline:
1. Upload an image with multiple people.
2. System generates a caption, detects objects, counts people,
recognizes faces, and summarizes emotions.
3. Qwen-3 model produces a clear **Turkish description** suitable for
visually impaired users.
4. gTTS converts description to speech and returns audio.

------------------------------------------------------------------------
