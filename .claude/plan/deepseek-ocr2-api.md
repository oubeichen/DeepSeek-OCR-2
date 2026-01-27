# DeepSeek-OCR-2 FastAPI 服务开发计划

## 任务概述

基于 DeepSeek-OCR2-vllm 代码，使用 FastAPI 创建 OCR API 服务，支持 PDF/图片上传和解析。

## 完成状态

- [x] 阶段 0：准备工作 - 创建项目目录结构
- [x] 阶段 1：配置管理 - Pydantic Settings
- [x] 阶段 2：引擎管理 - 单例模式引擎管理器
- [x] 阶段 3：处理器模块 - 图片、PDF、后处理
- [x] 阶段 4：请求/响应模型 - Pydantic schemas
- [x] 阶段 5：工具模块 - 结果打包
- [x] 阶段 6：路由模块 - OCR 和健康检查端点
- [x] 阶段 7：主应用 - FastAPI 入口
- [x] 阶段 8：启动脚本 - Shell 脚本和依赖

## 项目结构

```
deepseek_ocr2_api/
├── __init__.py              # 包初始化
├── __main__.py              # CLI 入口
├── main.py                  # FastAPI 应用
├── config.py                # 配置管理
├── engine/
│   ├── __init__.py
│   ├── manager.py           # 引擎单例管理器
│   └── inference.py         # 推理接口
├── processors/
│   ├── __init__.py
│   ├── image.py             # 图片处理
│   ├── pdf.py               # PDF 处理
│   └── postprocess.py       # 后处理
├── routers/
│   ├── __init__.py
│   ├── ocr.py               # OCR 端点
│   └── health.py            # 健康检查
├── schemas/
│   ├── __init__.py
│   ├── request.py           # 请求模型
│   └── response.py          # 响应模型
├── utils/
│   ├── __init__.py
│   └── packaging.py         # 打包工具
├── scripts/
│   └── start.sh             # 启动脚本
├── requirements.txt         # 依赖
├── README.md                # 文档
└── .env.example             # 环境变量示例
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/ocr/image` | POST | 单图片 OCR → ZIP |
| `/api/v1/ocr/pdf` | POST | PDF OCR → ZIP |
| `/api/v1/ocr/batch` | POST | 批量图片 OCR → ZIP |
| `/api/v1/ocr/image/json` | POST | 单图片 OCR → JSON |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/config` | GET | 配置查询 |
| `/api/v1/engine/status` | GET | 引擎状态 |

## 关键特性

1. **模型单例**: EngineManager 确保模型只加载一次
2. **参数可配置**: 所有常量通过环境变量/CLI/API 参数配置
3. **同步/异步**: 支持 LLM (sync) 和 AsyncLLMEngine (async)
4. **结果打包**: ZIP 文件包含 markdown、图片、标注
5. **OpenAPI 文档**: 完善的 Swagger/ReDoc 文档

## 启动方式

```bash
# 方式 1: 启动脚本
./scripts/start.sh --gpu-memory-utilization 0.8

# 方式 2: Python 模块
python -m deepseek_ocr2_api --port 8000

# 方式 3: 环境变量
export DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION=0.8
python -m deepseek_ocr2_api
```

## Git 提交历史

1. `feat(api): add project structure and config module`
2. `feat(api): add engine manager with singleton pattern`
3. `feat(api): add image, pdf and postprocess modules`
4. `feat(api): add request and response schemas`
5. `feat(api): add result packaging utility`
6. `feat(api): add OCR and health check routers`
7. `feat(api): add FastAPI application entry point`
8. `feat(api): add startup script and dependencies`
