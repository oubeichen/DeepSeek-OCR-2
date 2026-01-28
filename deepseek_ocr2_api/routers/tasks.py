"""
Task management router for web interface.

Provides endpoints for:
- File upload and task creation
- Task status and listing
- Result download and preview
- Task deletion
- Configuration retrieval
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
import aiofiles
import logging

from ..task_manager import TaskManager, TaskType, TaskStatus
from ..config import get_settings
from ..utils import unescape_string

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Tasks"])


def get_task_manager() -> TaskManager:
    """Get the task manager instance."""
    return TaskManager.get_instance()


@router.get("/config")
async def get_config():
    """
    Get current OCR configuration.

    Returns the current effective configuration values (after .env overrides).
    These values serve as defaults for the frontend form.
    """
    settings = get_settings()

    return {
        "sampling": {
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "ngram_size": settings.ngram_size,
            "window_size": settings.window_size,
            "skip_special_tokens": settings.skip_special_tokens,
        },
        "image_processing": {
            "crop_mode": settings.crop_mode,
            "min_crops": settings.min_crops,
            "max_crops": settings.max_crops,
            # Note: image_size (768) and base_size (1024) are fixed due to model constraints
        },
        "pdf_processing": {
            "pdf_dpi": settings.pdf_dpi,
            "page_separator": settings.page_separator,
            "skip_repeat_pages": settings.skip_repeat_pages,
        },
        "prompt": {
            "default_prompt": settings.default_prompt,
        },
        "presets": {
            "fast": {
                "name": "快速模式",
                "description": "速度优先，适合简单文档",
                "crop_mode": False,
                "min_crops": 1,
                "max_crops": 4,
            },
            "standard": {
                "name": "标准模式",
                "description": "平衡速度和质量",
                "crop_mode": True,
                "min_crops": 2,
                "max_crops": 6,
            },
            "quality": {
                "name": "高质量模式",
                "description": "质量优先，适合复杂文档",
                "crop_mode": True,
                "min_crops": 2,
                "max_crops": 9,
            },
        },
        "prompt_templates": {
            "document": {
                "name": "文档排版",
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
            },
            "ocr": {
                "name": "图片OCR",
                "prompt": "<image>\n<|grounding|>OCR this image.",
            },
            "nolayout": {
                "name": "无布局OCR",
                "prompt": "<image>\nFree OCR.",
            },
            "figure": {
                "name": "图表解析",
                "prompt": "<image>\nParse the figure.",
            },
            "describe": {
                "name": "详细描述",
                "prompt": "<image>\nDescribe this image in detail.",
            },
            "locate": {
                "name": "目标定位",
                "prompt": "<image>\nLocate <|ref|>xxxx<|/ref|> in the image.",
            },
        },
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(..., description="File to process"),
    # Prompt
    prompt: Optional[str] = Form(default=None, description="Custom prompt (leave empty to use default)", examples=[""]),
    # Sampling parameters
    temperature: Optional[float] = Form(default=None, description="Sampling temperature (0.0-2.0)", examples=[None]),
    max_tokens: Optional[int] = Form(default=None, description="Maximum tokens to generate", examples=[None]),
    ngram_size: Optional[int] = Form(default=None, description="N-gram size for repetition penalty", examples=[None]),
    window_size: Optional[int] = Form(default=None, description="Window size for N-gram check", examples=[None]),
    # Image processing parameters
    crop_mode: Optional[bool] = Form(default=None, description="Enable dynamic cropping", examples=[None]),
    min_crops: Optional[int] = Form(default=None, description="Minimum number of crops", examples=[None]),
    max_crops: Optional[int] = Form(default=None, description="Maximum number of crops", examples=[None]),
    # Note: image_size and base_size are fixed at 768 and 1024 respectively
    # due to model constraints (n_query must be 144 or 256)
    # PDF processing parameters
    dpi: Optional[int] = Form(default=None, description="PDF conversion DPI", examples=[None]),
    page_separator: Optional[str] = Form(default=None, description="Page separator in output (leave empty to use default)", examples=[""]),
    skip_repeat_pages: Optional[bool] = Form(default=None, description="Skip repeated/incomplete pages", examples=[None]),
    # Output format
    result_format: str = Form(default="zip", description="Output format: zip, markdown, or json"),
):
    """
    Upload a file and create an OCR task.

    All parameters are optional and will use server defaults if not provided.
    Returns the task ID for status tracking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Determine task type
    filename_lower = file.filename.lower()
    if filename_lower.endswith(".pdf"):
        task_type = TaskType.PDF
    elif any(filename_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"]):
        task_type = TaskType.IMAGE
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported: PNG, JPG, JPEG, WebP, BMP, TIFF, GIF, PDF"
        )

    # Get task manager
    manager = get_task_manager()

    # Save uploaded file
    storage_dir = Path("/tmp/deepseek_ocr2_tasks/uploads")
    storage_dir.mkdir(parents=True, exist_ok=True)

    import uuid
    file_ext = os.path.splitext(file.filename)[1]
    saved_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = storage_dir / saved_filename

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Build OCR parameters - only include non-None values
    settings = get_settings()
    ocr_params = {}

    # Prompt (treat empty string as None, apply unescape)
    ocr_params["prompt"] = unescape_string(prompt) if prompt else settings.default_prompt

    # Sampling parameters
    if temperature is not None:
        ocr_params["temperature"] = temperature
    if max_tokens is not None:
        ocr_params["max_tokens"] = max_tokens
    if ngram_size is not None:
        ocr_params["ngram_size"] = ngram_size
    if window_size is not None:
        ocr_params["window_size"] = window_size

    # Image processing parameters
    ocr_params["crop_mode"] = crop_mode if crop_mode is not None else settings.crop_mode
    if min_crops is not None:
        ocr_params["min_crops"] = min_crops
    if max_crops is not None:
        ocr_params["max_crops"] = max_crops
    # Note: image_size and base_size are fixed at 768 and 1024 respectively
    # due to model constraints (n_query must be 144 or 256)
    # Any user-provided values are ignored

    # PDF processing parameters
    ocr_params["dpi"] = dpi if dpi is not None else settings.pdf_dpi
    # Treat empty string as None for page_separator, apply unescape
    if page_separator:
        ocr_params["page_separator"] = unescape_string(page_separator)
    if skip_repeat_pages is not None:
        ocr_params["skip_repeat_pages"] = skip_repeat_pages

    # Output format
    if result_format in ("zip", "markdown", "json"):
        ocr_params["result_format"] = result_format
    else:
        ocr_params["result_format"] = "zip"

    # Create task
    task = manager.create_task(
        filename=file.filename,
        task_type=task_type,
        input_file_path=str(file_path),
        ocr_params=ocr_params,
    )

    # Enqueue task
    await manager.enqueue_task(task)

    return {
        "success": True,
        "task_id": task.task_id,
        "message": f"Task created: {task.task_id}",
    }


@router.get("/tasks")
async def list_tasks(
    limit: int = 100,
    offset: int = 0,
):
    """
    List all tasks.

    Returns tasks sorted by creation time (newest first).
    """
    manager = get_task_manager()
    tasks = manager.get_all_tasks(limit=limit, offset=offset)

    return {
        "tasks": [task.to_dict() for task in tasks],
        "total": manager.total_tasks,
        "limit": limit,
        "offset": offset,
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a specific task by ID."""
    manager = get_task_manager()
    task = manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task.to_dict()


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its files."""
    manager = get_task_manager()

    if not manager.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    return {"success": True, "message": f"Task {task_id} deleted"}


@router.get("/tasks/{task_id}/download")
async def download_result(task_id: str):
    """Download the result file for a completed task."""
    manager = get_task_manager()
    task = manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    if not task.result_zip_path or not os.path.exists(task.result_zip_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    # Determine response type based on file extension
    result_path = task.result_zip_path
    ext = os.path.splitext(result_path)[1].lower()

    if ext == ".md":
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content=content, media_type="text/markdown; charset=utf-8")
    elif ext == ".json":
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content=content, media_type="application/json")
    else:
        # ZIP or other binary files
        return FileResponse(
            path=result_path,
            media_type="application/zip",
            filename=f"{task.task_id}.zip",
        )


@router.get("/tasks/{task_id}/markdown")
async def get_task_markdown(task_id: str):
    """Get markdown content for a completed task.
    
    Works regardless of the original result_format setting.
    If the task was saved as ZIP, extracts markdown from the archive.
    """
    import zipfile
    
    manager = get_task_manager()
    task = manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    if not task.result_zip_path or not os.path.exists(task.result_zip_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    result_path = task.result_zip_path
    ext = os.path.splitext(result_path)[1].lower()

    # If already markdown file, return directly
    if ext == ".md":
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content=content, media_type="text/markdown; charset=utf-8")

    # If ZIP, extract markdown from archive
    if ext == ".zip":
        with zipfile.ZipFile(result_path, 'r') as zf:
            # Find markdown file in archive
            md_files = [n for n in zf.namelist() if n.endswith('.md') and not n.endswith('_raw.md')]
            if not md_files:
                raise HTTPException(status_code=404, detail="No markdown file in archive")
            # Use the first non-raw markdown file
            md_name = md_files[0]
            content = zf.read(md_name).decode('utf-8')
            return Response(content=content, media_type="text/markdown; charset=utf-8")

    raise HTTPException(status_code=400, detail="Cannot extract markdown from this result format")


@router.get("/tasks/{task_id}/json")
async def get_task_json(task_id: str):
    """Get doc.json content for a completed task.
    
    Works regardless of the original result_format setting.
    If the task was saved as ZIP, extracts doc.json from the archive.
    """
    import zipfile
    import json
    
    manager = get_task_manager()
    task = manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    if not task.result_zip_path or not os.path.exists(task.result_zip_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    result_path = task.result_zip_path
    ext = os.path.splitext(result_path)[1].lower()

    # If already JSON file, return directly
    if ext == ".json":
        with open(result_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return JSONResponse(content=content)

    # If ZIP, extract doc.json from archive
    if ext == ".zip":
        with zipfile.ZipFile(result_path, 'r') as zf:
            if "doc.json" not in zf.namelist():
                raise HTTPException(status_code=404, detail="No doc.json in archive")
            content = json.loads(zf.read("doc.json").decode('utf-8'))
            return JSONResponse(content=content)

    raise HTTPException(status_code=400, detail="Cannot extract JSON from this result format")


@router.get("/tasks/{task_id}/preview/original")
async def preview_original(task_id: str):
    """Preview the original uploaded file."""
    manager = get_task_manager()
    task = manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.input_file_path or not os.path.exists(task.input_file_path):
        raise HTTPException(status_code=404, detail="Original file not found")

    # Determine media type
    mime_type, _ = mimetypes.guess_type(task.filename)
    if not mime_type:
        if task.task_type == TaskType.PDF:
            mime_type = "application/pdf"
        else:
            mime_type = "image/png"

        # Encode filename for Content-Disposition header (RFC 5987)
    from urllib.parse import quote
    encoded_filename = quote(task.filename)
    
    return FileResponse(
        path=task.input_file_path,
        media_type=mime_type,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}"},
    )


@router.get("/tasks/{task_id}/preview/result")
async def preview_result(task_id: str):
    """
    Preview the OCR result as rendered HTML.

    Renders the markdown content as HTML for display in an iframe.
    """
    import zipfile
    
    manager = get_task_manager()
    task = manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    # Get markdown content
    md_content = None
    
    # First: try output directory (no extra resource needed)
    if task.output_dir and os.path.exists(task.output_dir):
        md_files = list(Path(task.output_dir).glob("*.md"))
        if md_files:
            md_content = md_files[0].read_text(encoding="utf-8")
    
    # Second: try result file
    if not md_content and task.result_zip_path and os.path.exists(task.result_zip_path):
        result_path = task.result_zip_path
        ext = os.path.splitext(result_path)[1].lower()
        
        if ext == ".md":
            # Direct markdown file
            with open(result_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        elif ext == ".zip":
            # Extract from ZIP (more resource intensive)
            with zipfile.ZipFile(result_path, 'r') as zf:
                md_files = [n for n in zf.namelist() if n.endswith('.md') and not n.endswith('_raw.md')]
                if md_files:
                    md_content = zf.read(md_files[0]).decode('utf-8')
    
    if not md_content:
        raise HTTPException(status_code=404, detail="No markdown result found")

    # Render as HTML with basic styling
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 100%;
            margin: 0 auto;
            background: #fff;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background: #f4f4f4;
        }}
        blockquote {{
            border-left: 4px solid #667eea;
            margin: 0;
            padding-left: 15px;
            color: #666;
        }}
    </style>
</head>
<body>
<pre>{html_escape(md_content)}</pre>
</body>
</html>
"""

    return HTMLResponse(content=html_content)


@router.get("/status")
async def get_status():
    """Get the current system status."""
    from ..engine import EngineManager

    manager = get_task_manager()
    engine = EngineManager.get_instance()

    return {
        "queue_size": manager.queue_size,
        "total_tasks": manager.total_tasks,
        "engine_initialized": engine.is_initialized(),
    }


def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
