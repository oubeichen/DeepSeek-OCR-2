FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_USE_V1=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deepseek_ocr2_api/requirements.txt ./deepseek_ocr2_api/requirements.txt
COPY requirements.txt ./requirements.txt

ARG VLLM_WHL_URL="https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "${VLLM_WHL_URL}" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -r deepseek_ocr2_api/requirements.txt \
    && python -m pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

COPY . .

EXPOSE 8000

CMD ["python", "-m", "deepseek_ocr2_api"]
