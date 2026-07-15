import json
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "locate-anything")


def draw_boxes(image, results):
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)

    for item in results:
        box = item.get("box")
        label = item.get("class", "box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill="red")

    return preview


def call_model_api(image_file, params):
    files = {
        "image": (
            image_file.name,
            image_file.getvalue(),
            image_file.type or "application/octet-stream",
        )
    }
    response = requests.post(
        f"{API_URL}/v1/models/{MODEL_NAME}/infer",
        data=params,
        files=files,
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="LocateAnything", layout="wide")
st.title("LocateAnything Layout Detection")

with st.sidebar:
    st.subheader("Inference")
    classes = st.text_area(
        "Classes",
        value="title,paragraph,table,figure,caption,list",
        height=90,
    )
    generation_mode = st.selectbox("Generation mode", ["hybrid", "fast", "slow"], index=0)
    max_new_tokens = st.slider("Max new tokens", 256, 8192, 8192, 256)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.number_input("Top k", min_value=0, value=0, step=1)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)

image_file = st.file_uploader("Upload a document page or screenshot", type=["png", "jpg", "jpeg", "webp"])

left, right = st.columns(2)

if image_file:
    image = Image.open(image_file).convert("RGB")
    left.image(image, caption="Input", use_container_width=True)

    if st.button("Run detection", type="primary"):
        params = {
            "classes": classes,
            "generation_mode": generation_mode,
            "max_new_tokens": str(max_new_tokens),
            "temperature": str(temperature),
            "top_p": str(top_p),
            "top_k": str(top_k),
            "repetition_penalty": str(repetition_penalty),
        }

        with st.spinner("Running LocateAnything inference..."):
            try:
                payload = call_model_api(image_file, params)
            except requests.RequestException as exc:
                st.error(f"Model API request failed: {exc}")
                st.stop()

        results = payload.get("results", [])
        right.image(draw_boxes(image, results), caption="Detected boxes", use_container_width=True)
        st.subheader("Raw API response")
        st.code(json.dumps(payload, indent=2), language="json")
else:
    st.info("Upload an image to test the hosted LocateAnything model.")
