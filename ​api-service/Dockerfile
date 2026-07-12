# MUDU-VLLM — tek imaj (API + Web)
# ÖNEMLİ: pip katmanı koddan ÖNCE — kod değişince torch yeniden inmez.
# Günlük kullanım için Docker şart değil: .\venv + python app.py yeterli.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1) Sadece bağımlılıklar (nadiren değişir → cache kalır)
COPY api-service/requirements.docker.txt /tmp/requirements.docker.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements.docker.txt \
    && pip install ultralytics --no-deps \
    && pip install matplotlib pyyaml requests scipy pandas seaborn psutil polars ultralytics-thop pillow

# 2) Yerel ağırlık varsa kopyala (build sırasında internetten YOLO çekmez)
COPY yolov8n.pt /app/yolov8n.pt
ENV YOLO_CONFIG_DIR=/tmp

# 3) Uygulama kodu EN SONDA (sık değişir → pip tekrar çalışmaz)
COPY api-service/app.py /app/api_app.py
COPY web-ui/app.py /app/web_app.py
COPY web-ui/index.html /app/index.html
COPY yolo/ /app/yolo/

ENV PYTHONPATH=/app/yolo
ENV HOST=0.0.0.0

EXPOSE 8000 7860
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
