# LocateAnything Docker Host

This repository runs NVIDIA `nvidia/LocateAnything-3B` as a small FastAPI service plus a Streamlit browser UI for document layout detection, OCR localization, GUI grounding, and object grounding.

The service accepts an image and a comma-separated list of layout classes, then returns bounding boxes in image pixel coordinates. The UI lets users upload images from a browser, tune inference settings, and inspect detected boxes.

## Requirements

- Linux host with an NVIDIA GPU.
- Recent NVIDIA driver.
- Docker and Docker Compose.
- NVIDIA Container Toolkit installed.
- Hugging Face access to `nvidia/LocateAnything-3B`.

Check GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Configure

Copy the example env file:

```bash
cp .env.example .env
```

Important settings:

```text
HOST_PORT              Port exposed on the host. Default: 8000.
UI_HOST_PORT           Browser UI port exposed on the host. Default: 8501.
GPU_DEVICE_ID          GPU index to use. Default: 0.
MODEL_CARD_NAME        Hugging Face model id.
MODEL_NAME             Local API model name.
GENERATION_MODE        fast, slow, or hybrid. Use hybrid by default.
MAX_NEW_TOKENS         Output length. NVIDIA recommends 8192 for LocateAnything.
MAX_IMAGE_SIZE         Longest image side before inference. Default: 2500.
MAX_CONCURRENCY        Number of requests allowed at the same time. Start with 1.
SHM_SIZE               Docker shared memory. Default: 16g.
```

Use lower values if the GPU runs out of memory:

```text
MAX_IMAGE_SIZE=1600
MAX_NEW_TOKENS=4096
MAX_CONCURRENCY=1
```

## Run

Build and start:

```bash
docker compose up -d --build
```

Follow logs:

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

The first start downloads the model into the Docker volume `hf-cache`. Later starts reuse that cache.

Open the UI:

```text
http://localhost:8501
```

From another machine on the same network:

```text
http://SERVER_IP:8501
```

## Test

Health check:

```bash
curl http://localhost:8000/health
```

Run layout detection:

```bash
curl -X POST "http://localhost:8000/v1/models/locate-anything/infer" \
  -F "image=@/path/to/page.jpg" \
  -F "classes=title,paragraph,table,figure,caption,list"
```

Override inference parameters per request:

```bash
curl -X POST "http://localhost:8000/v1/models/locate-anything/infer" \
  -F "image=@/path/to/page.jpg" \
  -F "classes=title,paragraph,table,figure,caption,list" \
  -F "generation_mode=hybrid" \
  -F "max_new_tokens=8192" \
  -F "temperature=0.7" \
  -F "top_p=0.9" \
  -F "top_k=0" \
  -F "repetition_penalty=1.1"
```

Response shape:

```json
{
  "model_name": "locate-anything",
  "model_card": "nvidia/LocateAnything-3B",
  "generation_mode": "hybrid",
  "results": [
    {
      "class": "table",
      "box": [120.5, 240.1, 850.0, 600.4]
    }
  ],
  "timing_info": [
    {
      "classes": ["title", "paragraph", "table"],
      "seconds": 4.31,
      "num_detections": 3,
      "raw_output": "<ref>table</ref><box><120><240><850><600></box>"
    }
  ]
}
```

## Docker Compose

`docker-compose.yml` stores the full run configuration so another user does not need to type a long `docker run` command.

It controls:

```text
model API image build
UI image build
host port mapping for API and UI
GPU selection
shared memory
model cache volume
restart policy
LocateAnything inference parameters
```

The key section is:

```yaml
environment:
  MODEL_CARD_NAME: "${MODEL_CARD_NAME:-nvidia/LocateAnything-3B}"
  GENERATION_MODE: "${GENERATION_MODE:-hybrid}"
  MAX_NEW_TOKENS: "${MAX_NEW_TOKENS:-8192}"
  MAX_IMAGE_SIZE: "${MAX_IMAGE_SIZE:-2500}"
  MAX_CONCURRENCY: "${MAX_CONCURRENCY:-1}"
```

Compose reads `.env`, substitutes the values into the YAML, creates the container, attaches the GPU, mounts the model cache, and starts the FastAPI server.

There are two containers:

```text
locate-anything       GPU model API on container port 8000.
locate-anything-ui    Streamlit browser UI on container port 8501.
```

The UI talks to the API through Docker's internal network:

```text
http://locate-anything:8000
```

Users connect only to the UI port unless they need direct API access.

## Hosting

For local LAN use, expose the configured UI port directly:

```text
http://SERVER_IP:8501
```

For public hosting, put Nginx in front of the UI and expose HTTPS only:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Then add TLS:

```bash
sudo certbot --nginx -d yourdomain.com
```

Keep ports `8000` and `8501` blocked from the public internet if Nginx is serving the UI. Expose only ports `80` and `443`.

For multiple users, keep model concurrency conservative:

```text
MAX_CONCURRENCY=1
```

Streamlit can serve multiple browser sessions, but the model API should usually process one GPU request at a time to avoid VRAM failures. Increase `MAX_CONCURRENCY` only after confirming your GPU has enough memory.

## Notes

LocateAnything coordinates are normalized internally from `0` to `1000`. This service converts boxes back to pixel coordinates using the uploaded image size.

The released model is a research model under NVIDIA's model license. Review the Hugging Face model card before using it in a commercial or production setting.
