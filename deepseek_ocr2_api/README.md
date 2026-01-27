# DeepSeek-OCR-2 API Server

基于 DeepSeek-OCR-2 模型的 FastAPI OCR 服务，支持图片和 PDF 文档的文字识别。

## 特性

- **单例模式引擎管理**：模型只加载一次，后续请求复用
- **同步/异步模式**：支持 vLLM 的 LLM 和 AsyncLLMEngine
- **完整的参数配置**：支持环境变量、.env 文件和命令行参数
- **多种输入格式**：支持 PNG、JPG、JPEG、WebP、BMP、TIFF、GIF 图片和 PDF
- **丰富的输出**：Markdown 文本、标注图片、提取的图片区域、JSON 元数据
- **OpenAPI 文档**：完整的 Swagger UI 和 ReDoc 文档

## 安装

```bash
# 安装依赖
pip install -r deepseek_ocr2_api/requirements.txt

# 或者使用 pip install
pip install fastapi uvicorn python-multipart pydantic-settings pillow pymupdf tqdm numpy matplotlib
```

## 快速开始

### 使用启动脚本

```bash
# 基本启动
./deepseek_ocr2_api/scripts/start.sh

# 指定模型路径和 GPU 内存
./deepseek_ocr2_api/scripts/start.sh --model-path /path/to/model --gpu-memory-utilization 0.8

# 使用异步模式
./deepseek_ocr2_api/scripts/start.sh --mode async

# 查看所有选项
./deepseek_ocr2_api/scripts/start.sh --help
```

### 使用 Python 模块

```bash
# 基本启动
python -m deepseek_ocr2_api

# 指定参数
python -m deepseek_ocr2_api --model-path /path/to/model --gpu-memory-utilization 0.8 --port 8080
```

### 使用环境变量

```bash
export DEEPSEEK_OCR_MODEL_PATH=/path/to/model
export DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION=0.8
export DEEPSEEK_OCR_HOST=0.0.0.0
export DEEPSEEK_OCR_PORT=8080

python -m deepseek_ocr2_api
```

### 使用 .env 文件

创建 `.env` 文件：

```env
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR2-Pro
DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION=0.9
DEEPSEEK_OCR_TENSOR_PARALLEL_SIZE=1
DEEPSEEK_OCR_MAX_TOKENS=8192
```

## API 端点

### OCR 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/ocr/image` | POST | 单张图片 OCR，返回 ZIP |
| `/api/v1/ocr/image/json` | POST | 单张图片 OCR，返回 JSON |
| `/api/v1/ocr/pdf` | POST | PDF 文档 OCR，返回 ZIP |
| `/api/v1/ocr/batch` | POST | 批量图片 OCR，返回 ZIP |

### 健康检查端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/config` | GET | 当前配置 |
| `/api/v1/engine/status` | GET | 引擎状态 |

### 文档端点

| 端点 | 描述 |
|------|------|
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |
| `/openapi.json` | OpenAPI Schema |

## 使用示例

### cURL

```bash
# 单张图片 OCR
curl -X POST "http://localhost:8000/api/v1/ocr/image" \
  -F "file=@image.png" \
  -F "prompt=OCR with format" \
  -o result.zip

# PDF OCR
curl -X POST "http://localhost:8000/api/v1/ocr/pdf" \
  -F "file=@document.pdf" \
  -F "dpi=200" \
  -o result.zip

# JSON 响应
curl -X POST "http://localhost:8000/api/v1/ocr/image/json" \
  -F "file=@image.png" \
  | jq .
```

### Python

```python
import requests

# 单张图片 OCR
with open("image.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/image",
        files={"file": f},
        data={"prompt": "OCR with format"}
    )
    with open("result.zip", "wb") as out:
        out.write(response.content)

# JSON 响应
with open("image.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/image/json",
        files={"file": f}
    )
    result = response.json()
    print(result["results"][0]["markdown"])
```

## 配置参数

### 模型配置

| 参数 | 环境变量 | 默认值 | 描述 |
|------|----------|--------|------|
| `--model-path` | `DEEPSEEK_OCR_MODEL_PATH` | `deepseek-ai/DeepSeek-OCR2-Pro` | 模型路径 |
| `--dtype` | `DEEPSEEK_OCR_DTYPE` | `bfloat16` | 模型数据类型 |
| `--trust-remote-code` | `DEEPSEEK_OCR_TRUST_REMOTE_CODE` | `true` | 信任远程代码 |

### GPU 配置

| 参数 | 环境变量 | 默认值 | 描述 |
|------|----------|--------|------|
| `--gpu-memory-utilization` | `DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION` | `0.9` | GPU 内存使用率 |
| `--tensor-parallel-size` | `DEEPSEEK_OCR_TENSOR_PARALLEL_SIZE` | `1` | 张量并行大小 |
| `--max-model-len` | `DEEPSEEK_OCR_MAX_MODEL_LEN` | `16384` | 最大模型长度 |

### 采样配置

| 参数 | 环境变量 | 默认值 | 描述 |
|------|----------|--------|------|
| `--temperature` | `DEEPSEEK_OCR_TEMPERATURE` | `0.0` | 采样温度 |
| `--max-tokens` | `DEEPSEEK_OCR_MAX_TOKENS` | `8192` | 最大生成 token 数 |
| `--ngram-size` | `DEEPSEEK_OCR_NGRAM_SIZE` | `20` | N-gram 重复惩罚大小 |
| `--window-size` | `DEEPSEEK_OCR_WINDOW_SIZE` | `100` | N-gram 检查窗口大小 |

### 服务器配置

| 参数 | 环境变量 | 默认值 | 描述 |
|------|----------|--------|------|
| `--host` | `DEEPSEEK_OCR_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `DEEPSEEK_OCR_PORT` | `8000` | 监听端口 |
| `--mode` | `DEEPSEEK_OCR_ENGINE_MODE` | `sync` | 引擎模式 (sync/async) |

## 输出格式

### ZIP 包内容

```
result.zip
├── output.md           # 合并的 Markdown 文本
├── metadata.json       # 处理元数据
├── annotated_0.jpg     # 标注后的图片
├── annotated.pdf       # 标注后的 PDF (仅 PDF 输入)
└── images/             # 提取的图片区域
    ├── 0_0.jpg
    ├── 0_1.jpg
    └── ...
```

### JSON 响应格式

```json
{
  "success": true,
  "message": "OCR completed successfully",
  "request_id": "img_abc123",
  "processing_time": 2.5,
  "results": [
    {
      "page_index": 0,
      "markdown": "# Title\n\nContent...",
      "raw_output": null,
      "extracted_images": [],
      "annotated_image": null
    }
  ],
  "total_pages": 1
}
```

## 项目结构

```
deepseek_ocr2_api/
├── __init__.py
├── __main__.py          # CLI 入口
├── main.py              # FastAPI 应用
├── config.py            # 配置管理
├── engine/
│   ├── __init__.py
│   ├── manager.py       # 单例引擎管理器
│   └── inference.py     # 推理接口
├── processors/
│   ├── __init__.py
│   ├── image.py         # 图片处理
│   ├── pdf.py           # PDF 处理
│   └── postprocess.py   # 后处理
├── routers/
│   ├── __init__.py
│   ├── ocr.py           # OCR 路由
│   └── health.py        # 健康检查路由
├── schemas/
│   ├── __init__.py
│   ├── request.py       # 请求模型
│   └── response.py      # 响应模型
├── utils/
│   ├── __init__.py
│   └── packaging.py     # 打包工具
├── scripts/
│   └── start.sh         # 启动脚本
├── requirements.txt     # 依赖
└── README.md            # 文档
```

## 许可证

本项目基于 DeepSeek-OCR-2 模型，请遵循相应的许可证条款。
