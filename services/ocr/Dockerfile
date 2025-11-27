FROM python:3.10-slim

# System deps: build tools + minimal GL stuff for OpenCV/MMOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python base deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OpenMMLab stack via openmim
RUN pip install --no-cache-dir openmim && \
    mim install "mmengine==0.10.5" && \
    mim install "mmcv==2.1.0" && \
    mim install "mmdet==3.3.0" && \
    mim install "mmocr==1.0.1"

# Your app
COPY app ./app

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
