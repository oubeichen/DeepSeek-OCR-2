# DeepSeek-OCR-2 API Server

基于 DeepSeek-OCR-2 模型的 FastAPI OCR 服务，支持图片和 PDF 文档的文字识别。

## 特性

- 单例模式引擎管理：模型只加载一次，后续请求复用
- 同步/异步模式：支持 vLLM 的 LLM 和 AsyncLLMEngine
- 统一配置管理：支持 .env 文件、环境变量和命令行参数
- 多种输入格式：支持 PNG、JPG、JPEG、WebP、BMP、TIFF、GIF 图片和 PDF
- 丰富的输出：Markdown 文本、标注图片、提取的图片区域、JSON 元数据
- OpenAPI 文档：完整的 Swagger UI 和 ReDoc 文档

## 环境要求

- CUDA 11.8
- Python 3.12.9

## 安装

```bash
# 1. 创建 conda 环境
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2

# 2. 下载 vLLM whl
# 从 https://github.com/vllm-project/vllm/releases/tag/v0.8.5 下载
# vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 3. 安装依赖
pip install -r deepseek_ocr2_api/requirements.txt
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install flash-attn==2.7.3 --no-build-isolation
```

## 配置

配置优先级（从高到低）：
1. 命令行参数
2. 环境变量 (`DEEPSEEK_OCR_*`)
3. .env 文件
4. 默认值

### 使用 .env 文件配置

```bash
cp deepseek_ocr2_api/.env.example .env
vim .env
```

### 主要配置项

| 环境变量 | 默认值 | 描述 |
|----------|-------|------|
| `DEEPSEEK_OCR_MODEL_PATH` | `deepseek-ai/DeepSeek-OCR-2` | 模型路径 |
| `DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION` | `0.9` | GPU 内存使用率 |
| `DEEPSEEK_OCR_TENSOR_PARALLEL_SIZE` | `1` | 张量并行大小 |
| `DEEPSEEK_OCR_HOST` | `0.0.0.0` | 服务地址 |
| `DEEPSEEK_OCR_PORT` | `8000` | 服务端口 |
| `DEEPSEEK_OCR_ENGINE_MODE` | `sync` | 引擎模式 (sync/async) |

完整配置项见 `.env.example`。

## 启动

```bash
# 使用 .env 配置启动
./deepseek_ocr2_api/scripts/start.sh

# 临时覆盖配置
./deepseek_ocr2_api/scripts/start.sh --gpu-memory-utilization 0.8 --port 8080

# 使用 Python 模块启动
python -m deepseek_ocr2_api
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/ocr/image` | POST | 单张图片 OCR，返回 ZIP |
| `/api/v1/ocr/image/json` | POST | 单张图片 OCR，返回 JSON |
| `/api/v1/ocr/pdf` | POST | PDF 文档 OCR，返回 ZIP |
| `/api/v1/ocr/batch` | POST | 批量图片 OCR，返回 ZIP |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/config` | GET | 当前配置 |
| `/api/v1/engine/status` | GET | 引擎状态 |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

## 使用示例

### cURL

```bash
# 单张图片 OCR
curl -X POST "http://localhost:8000/api/v1/ocr/image" \
  -F "file=@image.png" \
  -o result.zip

# PDF OCR
curl -X POST "http://localhost:8000/api/v1/ocr/pdf" \
  -F "file=@document.pdf" \
  -o result.zip

# JSON 响应
curl -X POST "http://localhost:8000/api/v1/ocr/image/json" \
  -F "file=@image.png"
```

### Python

```python
import requests

# 单张图片 OCR
with open("image.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/image",
        files={"file": f}
    )
    with open("result.zip", "wb") as out:
        out.write(response.content)

# JSON 响应
with open("image.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/image/json",
        files={"file": f}
    )
    print(response.json()["results"][0]["markdown"])
```

## 输出格式

### ZIP 包内容

```
result.zip
├── output.md           # Markdown 文本
├── metadata.json       # 处理元数据
├── annotated_0.jpg     # 标注图片
├── annotated.pdf       # 标注 PDF (仅 PDF 输入)
└── images/             # 提取的图片
    ├── 0_0.jpg
    └── ...
```

### JSON 响应

```json
{
  "success": true,
  "request_id": "img_abc123",
  "processing_time": 2.5,
  "results": [
    {
      "page_index": 0,
      "markdown": "# Title\n\nContent..."
    }
  ],
  "total_pages": 1
}
```

## 项目结构

```
deepseek_ocr2_api/
├── config.py            # 配置管理
├── main.py              # FastAPI 应用
├── __main__.py          # CLI 入口
├── engine/
│   ├── manager.py       # 单例引擎管理器
│   └── inference.py     # 推理接口
├── processors/
│   ├── image.py         # 图片处理
│   ├── pdf.py           # PDF 处理
│   └── postprocess.py   # 后处理
├── routers/
│   ├── ocr.py           # OCR 路由
│   └── health.py        # 健康检查路由
├── schemas/
│   ├── request.py       # 请求模型
│   └── response.py      # 响应模型
├── utils/
│   └── packaging.py     # 打包工具
├── scripts/
│   └── start.sh         # 启动脚本
├── requirements.txt     # 依赖
├── .env.example         # 配置示例
└── README.md
```
