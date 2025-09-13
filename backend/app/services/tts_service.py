import io
from gtts import gTTS

class TTSService:
    def __init__(self, lang="tr"):
        self.lang = lang

    def text_to_speech(self, text: str) -> bytes:
        tts = gTTS(text=text, lang=self.lang)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()

tts_service = TTSService()
