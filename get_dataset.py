import os
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

def download_images(dataset_name: str,
                    split: str,
                    img_column: str,
                    n: int,
                    out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(dataset_name, split=split, streaming=True)

    for idx, item in enumerate(ds.take(n)):
        img_data = item[img_column]

        if isinstance(img_data, Image.Image):
            img = img_data
        else:
            resp = requests.get(img_data, timeout=30)
            img  = Image.open(BytesIO(resp.content))

        img.save(os.path.join(out_dir, f"{idx:04d}.jpg"))
        if (idx + 1) % 50 == 0:
            print(f"{out_dir}: {idx + 1}/{n} tamam")

# ---- Flickr1K----
download_images(
    dataset_name="nlphuji/flickr_1k_test_image_text_retrieval",
    split="test",
    img_column="image",
    n=500,
    out_dir="flickr1k_images"
)

# ---- COCO2017 ----
download_images(
    dataset_name="phiyodr/coco2017",
    split="train",
    img_column="coco_url",
    n=500,
    out_dir="coco2017_images"
)
