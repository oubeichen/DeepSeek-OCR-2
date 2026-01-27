# 前端配置参数完善计划

## 目标

实现三层配置覆盖机制：
1. **代码默认值** - `config.py` 中的 Field default
2. **.env 覆盖** - 环境变量/`.env` 文件覆盖默认值
3. **API 调用覆盖** - 每次请求可以传入参数覆盖

前端需要：
- 获取当前生效的配置（.env 覆盖后的默认值）
- 展示所有可配置参数
- 允许用户修改并在上传时提交

---

## 可配置参数分类

### 1. Sampling Settings（采样参数）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.0 | 采样温度，0.0 为确定性输出 |
| `max_tokens` | int | 8192 | 最大生成 token 数 |
| `ngram_size` | int | 20 | N-gram 重复惩罚大小 |
| `window_size` | int | 90 | N-gram 检查窗口大小 |

### 2. Image Processing Settings（图像处理参数）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `crop_mode` | bool | true | 是否启用动态裁剪 |
| `min_crops` | int | 2 | 最小裁剪数 |
| `max_crops` | int | 6 | 最大裁剪数 |
| `image_size` | int | 768 | 局部视图图像尺寸 |
| `base_size` | int | 1024 | 全局视图基础尺寸 |

### 3. PDF Processing Settings（PDF 处理参数）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pdf_dpi` | int | 144 | PDF 转图像 DPI |
| `page_separator` | string | `\n<--- Page Split --->\n` | 页面分隔符 |
| `skip_repeat_pages` | bool | true | 跳过重复/不完整页面 |

### 4. Prompt Settings（提示词）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | string | `<image>\n<|grounding|>Convert...` | OCR 提示词 |

---

## 实现步骤

### Step 1: 添加配置获取 API

**文件**: `deepseek_ocr2_api/routers/tasks.py`

新增接口：
```
GET /api/config
```

返回当前生效的可配置参数（.env 覆盖后的值）：
```json
{
  "sampling": {
    "temperature": 0.0,
    "max_tokens": 8192,
    "ngram_size": 20,
    "window_size": 90
  },
  "image_processing": {
    "crop_mode": true,
    "min_crops": 2,
    "max_crops": 6,
    "image_size": 768,
    "base_size": 1024
  },
  "pdf_processing": {
    "pdf_dpi": 144,
    "page_separator": "\n<--- Page Split --->\n",
    "skip_repeat_pages": true
  },
  "prompt": {
    "default_prompt": "<image>\n<|grounding|>Convert the document to markdown."
  }
}
```

### Step 2: 更新上传接口参数

**文件**: `deepseek_ocr2_api/routers/tasks.py`

`POST /api/upload` 接口添加所有可配置参数：
- Sampling: `temperature`, `max_tokens`, `ngram_size`, `window_size`
- Image: `crop_mode`, `min_crops`, `max_crops`, `image_size`, `base_size`
- PDF: `pdf_dpi`, `page_separator`, `skip_repeat_pages`
- Prompt: `prompt`

### Step 3: 更新任务处理逻辑

**文件**: `deepseek_ocr2_api/task_manager.py`

在 `_process_queue` 中使用任务的 `ocr_params` 覆盖默认配置。

### Step 4: 更新前端界面

**文件**: `deepseek_ocr2_api/static/index.html`

#### 4.1 页面加载时获取配置
```javascript
async function loadConfig() {
    const response = await fetch(`${API_BASE}/api/config`);
    const config = await response.json();
    // 填充表单默认值
}
```

#### 4.2 配置表单分组

**采样参数组**:
- Temperature: 数字输入 (0-2, step 0.1)
- Max Tokens: 数字输入 (256-16384, step 256)
- N-gram Size: 数字输入 (1-50)
- Window Size: 数字输入 (10-200)

**图像处理参数组**:
- Crop Mode: 下拉框 (启用/禁用)
- Min Crops: 数字输入 (1-9)
- Max Crops: 数字输入 (1-9)
- Image Size: 数字输入 (256-2048, step 64)
- Base Size: 数字输入 (256-2048, step 64)

**PDF 处理参数组**:
- PDF DPI: 数字输入 (72-600, step 36)
- Page Separator: 文本输入
- Skip Repeat Pages: 下拉框 (是/否)

**提示词组**:
- Prompt: 多行文本框
- 快捷提示词按钮

#### 4.3 表单提交
收集所有配置参数，通过 FormData 提交到 `/api/upload`

---

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `routers/tasks.py` | 添加 `GET /api/config`，扩展 `POST /api/upload` 参数 |
| `task_manager.py` | 使用完整的 ocr_params 处理任务 |
| `static/index.html` | 重构配置表单，添加配置加载逻辑 |

---

## 预设功能（可选）

保留预设按钮，点击后自动填充一组推荐配置：

| 预设名 | 说明 | 配置 |
|--------|------|------|
| 快速模式 | 速度优先 | crop_mode=false, image_size=512, base_size=512 |
| 标准模式 | 平衡 | crop_mode=true, image_size=768, base_size=1024 |
| 高质量模式 | 质量优先 | crop_mode=true, image_size=1024, base_size=1280, max_crops=9 |

---

## 验收标准

1. [ ] `GET /api/config` 返回当前生效配置
2. [ ] `POST /api/upload` 支持所有可配置参数
3. [ ] 前端加载时显示当前默认配置
4. [ ] 前端可修改所有参数
5. [ ] 修改后的参数正确传递到任务处理
6. [ ] 预设按钮可快速填充配置
