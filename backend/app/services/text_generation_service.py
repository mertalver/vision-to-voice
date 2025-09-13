from typing import List
from app.core.config import settings
from typing import List
from llama_cpp import Llama
import re

class TextGenerationService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=2048)
        print(f"Text Generation model loaded from {self.model_path}")

    def generate_description(self, caption: str, objects: List[str], emotions: List[str]) -> str:
        object_text = "; ".join(objects)

        system_prompt = (
            "You are an assistive AI for visually impaired users. "
            "Your job is to describe images clearly, concisely, and helpfully. "
            "Focus only on key visual elements such as people, meaningful objects, and relevant emotions. "
            "Avoid unnecessary or overly detailed descriptions. "
            "If positions are relevant, describe them using clear spatial references like left, right, center, top, or bottom. "
            "Mention facial expressions only if they significantly change the meaning of the scene or relate to an action (e.g. someone angry holding a knife)."
        )

        user_prompt = f"""
        Görselde genel olarak: {caption}
        Tespit edilen nesneler: {object_text}
        Yüz ifadeleri: {emotions if emotions else "Resimde insan yer almıyor."}

        Bu bilgilere dayanarak görseli bir görme engelli bireye açıklamanı istiyorum. Lütfen:
        - Kısa ve net cümleler kullan.
        - Sadece önemli detaylara odaklan.
        - Konumları yön bilgisiyle belirt (örn: sol üstte, sağda, ortada).
        - Yüz ifadelerini yalnızca davranışla ilişkili veya dikkat çekici olduğunda belirt.
        - Teknik veya anlamsız tekrarlar yapma.

        Gereksiz ayrıntıdan kaçın, açıklaman kolay anlaşılır, kısa ve net olsun.
        """

        formatted_prompt = f"""
        <|im_start|>system
        {system_prompt}
        <|im_end|>
        <|im_start|>user
        {user_prompt}
        <|im_end|>
        <|im_start|>assistant
        Önce tespit edilen nesneleri ve yön bilgilerini sınıflandıracağım. Ardından yüz ifadelerinin genel duygusunu değerlendireceğim, tehlike arz eden bir yüz ifadesi varsa sadece bundan bahsedeceğim. Sonuç olarak bir görme engelli kullanıcıya uygun, yön tarifi yapan bir açıklama oluşturacağım. Cevabım kesinlikle TÜRKÇE olmalı ve 600 token'a sığdırmalıyım.
        <|im_end|>
        """

        # LLM çağrısı
        response = self.llm(formatted_prompt, max_tokens=600, temperature=0.7, top_p=0.9)
        return response['choices'][0]['text'].strip()

    def clean_qwen_output(self, output: str) -> str:
        # <think> ve </think> arasında kalan kısmı temizle
        return re.sub(r"<\/?think>", "", output).strip()

# Singleton instance
text_generation_service = TextGenerationService(settings.QWEN_MODEL_PATH) 