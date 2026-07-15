FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CARD_NAME=nvidia/LocateAnything-3B \
    MODEL_NAME=locate_-anything \
    MAX_CONCURRENCY=5

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY locate_anything/requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app/locate_anything

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
