import asyncio
import io
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from detector import LocateAnythingDetector

MODEL_NAME = os.getenv("MODEL_NAME", "locate-anything")
MODEL_CARD_NAME = os.getenv("MODEL_CARD_NAME", "nvidia/LocateAnything-3B")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", os.getenv("DEFAULT_TEMPERATURE", "0.7")))
DEFAULT_GENERATION_MODE = os.getenv("GENERATION_MODE", "hybrid")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "8192"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "0"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

app = FastAPI(title="LocateAnything service", version="1.0.0")


def _build_detector():
    requested_device = os.getenv("DEVICE", "cuda")
    device = requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    return LocateAnythingDetector(model_id=MODEL_CARD_NAME, device=device)


detector = _build_detector()
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_card": MODEL_CARD_NAME,
        "device": detector.device,
    }


@app.post("/v1/models/{model_name}/infer")
async def infer_model(
    model_name: str,
    image: UploadFile = File(...),
    classes: str = Form(default=""),
    generation_mode: str = Form(default=DEFAULT_GENERATION_MODE),
    max_new_tokens: int = Form(default=DEFAULT_MAX_NEW_TOKENS),
    temperature: float = Form(default=DEFAULT_TEMPERATURE),
    top_p: float = Form(default=DEFAULT_TOP_P),
    top_k: int = Form(default=DEFAULT_TOP_K),
    repetition_penalty: float = Form(default=DEFAULT_REPETITION_PENALTY),
):
    if model_name != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Unknown model name")

    async with inference_semaphore:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        class_list = [item.strip() for item in classes.split(",") if item.strip()]
        if not class_list:
            class_list = ["title", "paragraph", "table", "figure", "caption", "list"]

        def run_detection():
            return detector.detect(
                img,
                class_list,
                generation_mode=generation_mode,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
            )

        results, timing_info = await asyncio.to_thread(run_detection)

        return {
            "model_name": MODEL_NAME,
            "model_card": MODEL_CARD_NAME,
            "generation_mode": generation_mode,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "results": results,
            "timing_info": timing_info,
        }
