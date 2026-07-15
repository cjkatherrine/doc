import asyncio
import io
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from detector import LocateAnythingDetector

MODEL_NAME = os.getenv("MODEL_NAME", "locate_-anything")
MODEL_CARD_NAME = os.getenv("MODEL_CARD_NAME", "nvidia/LocateAnything-3B")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))

app = FastAPI(title="LocateAnything service", version="1.0.0")


def _build_detector():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LocateAnythingDetector(device=device)


detector = _build_detector()
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


@app.get("/health")
def health():
    return {"status": "ok", "model_name": MODEL_NAME, "model_card": MODEL_CARD_NAME}


@app.post("/v1/models/{model_name}/infer")
async def infer_model(
    model_name: str,
    image: UploadFile = File(...),
    classes: str = Form(default=""),
    temperature: float = Form(default=DEFAULT_TEMPERATURE),
):
    if model_name != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Unknown model name")

    async with inference_semaphore:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        class_list = [item.strip() for item in classes.split(",") if item.strip()]
        if not class_list:
            class_list = ["document"]

        def run_detection():
            return detector.detect(img, class_list, temperature=float(temperature))

        results, timing_info = await asyncio.to_thread(run_detection)

        return {
            "model_name": MODEL_NAME,
            "model_card": MODEL_CARD_NAME,
            "temperature": float(temperature),
            "results": results,
            "timing_info": timing_info,
        }
