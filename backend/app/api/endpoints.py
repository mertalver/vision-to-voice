from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from collections import Counter

from app.services.image_captioning_service import captioning_service
from app.services.object_detection_service import object_detection_service
from app.services.text_generation_service import text_generation_service
from app.services.tts_service import tts_service
from app.services.emotion_detection_service import emotion_detection_service

router = APIRouter()

@router.post("/describe-image/", response_class=StreamingResponse)
async def describe_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="There was an error processing the image file.")

    # 1. Caption
    caption = captioning_service.generate_caption(image)
    print("Caption:", caption)

    # 2. Object Detection
    objects = object_detection_service.detect_objects(image)

    # 3. Kişi kontrolü ve özetleme
    person_labels = ['kişi', 'person']
    persons = [obj for obj in objects if obj['label'] in person_labels]

    if persons:
        person_count = len(persons)
        positions = [obj['position'] for obj in persons]
        pos_counts = Counter(positions)
        pos_summary = ", ".join(f"{count} kişi {pos} konumunda" for pos, count in pos_counts.items())
        object_text = f"Toplam {person_count} kişi var. Konumlar: {pos_summary}."
        
        # Tüm resmi FER'ye gönder, kendi yüz algılamasını yapsın
        emotions_list = emotion_detection_service.detect_emotions(image)
        emotions_text = emotion_detection_service.summarize_emotions(emotions_list)
    else:
        object_text = object_detection_service.format_objects_naturally(objects)
        emotions_text = "Resimde insan yer almıyor."

    print("Objects summary:", object_text)
    print("Emotions:", emotions_text)

    # 4. Qwen için açıklama üret
    description = text_generation_service.generate_description(
        caption=caption,
        objects=object_text,
        emotions=emotions_text
    )
    description = text_generation_service.clean_qwen_output(description)
    print("Description:", description)

    # 5. Convert text to speech
    audio_bytes = tts_service.text_to_speech(description)

    with open("output.mp3", "wb") as f:
        f.write(audio_bytes)

    # 6. Stream the audio back
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
