# DeepSeek-OCR-2 API Server

基于 FastAPI 的 DeepSeek-OCR-2 模型推理服务，支持 PDF 和图片 OCR。

## 特性

- **单图片 OCR**: 处理单张图片，提取文本为 Markdown
- **PDF OCR**: 处理 PDF 文档，支持多页
- **批量处理**: 单次请求处理多张图片
- **参数可配置**: 自定义 Prompt、采样参数、处理选项
- **结果打包**: ZIP 文件包含 Markdown、标注图片、提取内容
- **模型复用**: 单例模式确保模型只加载一次

## 快速开始

### 安装依赖

```bash
cd deepseek_ocr2_api
pip install -r requirements.txt
```

### 启动服务

#### 方式 1: 使用启动脚本

```bash
./scripts/start.sh --gpu-memory-utilization 0.8
```

#### 方式 2: 使用 Python 模块

```bash
python -m deepseek_ocr2_api --port 8000 --gpu-memory-utilization 0.8
```

#### 方式 3: 使用环境变量

```bash
export DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION=0.8
export DEEPSEEK_OCR2_PORT=8000
python -m deepseek_ocr2_api
```

### 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### OCR 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/ocr/image` | POST | 单图片 OCR，返回 ZIP |
| `/api/v1/ocr/pdf` | POST | PDF OCR，返回 ZIP |
| `/api/v1/ocr/batch` | POST | 批量图片 OCR，返回 ZIP |
| `/api/v1/ocr/image/json` | POST | 单图片 OCR，返回 JSON |

### 健康检查端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/health` | GET | 服务健康状态 |
| `/api/v1/config` | GET | 当前配置 |
| `/api/v1/engine/status` | GET | 引擎状态 |

## 配置参数

所有参数支持通过以下方式配置：
- 环境变量（前缀 `DEEPSEEK_OCR2_`）
- 命令行参数
- API 请求参数（覆盖默认值）

### 主要参数

| 参数 | 环境变量 | 默认值 | 描述 |
|------|----------|--------|------|
| `--model-path` | `DEEPSEEK_OCR2_MODEL_PATH` | `deepseek-ai/DeepSeek-OCR-2` | 模型路径 |
| `--gpu-memory-utilization` | `DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION` | `0.8` | GPU 内存利用率 |
| `--cuda-devices` | `DEEPSEEK_OCR2_CUDA_VISIBLE_DEVICES` | `0` | CUDA 设备 |
| `--tensor-parallel-size` | `DEEPSEEK_OCR2_TENSOR_PARALLEL_SIZE` | `1` | 张量并行 GPU 数 |
| `--max-model-len` | `DEEPSEEK_OCR2_MAX_MODEL_LEN` | `8192` | 最大序列长度 |
| `--engine-mode` | `DEEPSEEK_OCR2_ENGINE_MODE` | `sync` | 引擎模式 |
| `--port` | `DEEPSEEK_OCR2_PORT` | `8000` | 服务端口 |

### 完整参数列表

```bash
python -m deepseek_ocr2_api --help
```

## 使用示例

### cURL

```bash
# 单图片 OCR
curl -X POST "http://localhost:8000/api/v1/ocr/image" \
  -F "file=@document.png" \
  -F "temperature=0.0" \
  -o result.zip

# PDF OCR
curl -X POST "http://localhost:8000/api/v1/ocr/pdf" \
  -F "file=@document.pdf" \
  -F "dpi=144" \
  -o result.zip

# 批量图片 OCR
curl -X POST "http://localhost:8000/api/v1/ocr/batch" \
  -F "files=@page1.png" \
  -F "files=@page2.png" \
  -o result.zip
```

### Python

```python
import requests

# 单图片 OCR
with open("document.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/image",
        files={"file": f},
        data={"temperature": 0.0}
    )
    with open("result.zip", "wb") as out:
        out.write(response.content)

# JSON 响应
with open("document.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/image/json",
        files={"file": f}
    )
    result = response.json()
    print(result["results"][0]["markdown"])
```

## 输出格式

### ZIP 文件结构（图片）

```
result.zip
├── page_0.md           # 单页 Markdown
├── combined.md         # 合并 Markdown
├── annotated/
│   └── page_0.jpg      # 标注图片
├── images/
│   ├── 0_0.jpg         # 提取的图片
│   └── 0_1.jpg
└── metadata.json       # 元数据
```

### ZIP 文件结构（PDF）

```
document_ocr.zip
├── document.md         # 合并 Markdown
├── document_raw.md     # 原始输出
├── document_annotated.pdf  # 标注 PDF
├── images/
│   ├── 0_0.jpg         # 第 0 页提取的图片
│   └── 1_0.jpg         # 第 1 页提取的图片
└── metadata.json       # 元数据
```

## 支持的文件格式

- **图片**: PNG, JPG, JPEG, WebP, BMP, TIFF, GIF
- **文档**: PDF

## 注意事项

1. **GPU 内存**: 根据 GPU 显存调整 `--gpu-memory-utilization`
2. **模型加载**: 首次启动需要下载模型，请确保网络连接
3. **并发处理**: 使用 `--max-num-seqs` 控制最大并发数
4. **PDF 质量**: 使用 `--pdf-dpi` 调整 PDF 转换质量

## License

MIT
