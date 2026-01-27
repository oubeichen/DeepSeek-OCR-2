"""
Response schemas for DeepSeek-OCR-2 API Server.

Defines Pydantic models for API responses with OpenAPI documentation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """
    Single OCR result for one image/page.
    """

    page_index: int = Field(
        description="Page or image index (0-based)"
    )
    markdown: str = Field(
        description="Extracted markdown content"
    )
    raw_output: Optional[str] = Field(
        default=None,
        description="Raw model output (if requested)"
    )
    extracted_images: List[str] = Field(
        default_factory=list,
        description="List of extracted image filenames"
    )
    annotated_image: Optional[str] = Field(
        default=None,
        description="Annotated image filename (if generated)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "page_index": 0,
                "markdown": "# Document Title\n\nThis is the extracted content...",
                "extracted_images": ["images/0_0.jpg", "images/0_1.jpg"],
                "annotated_image": "annotated_0.jpg"
            }
        }


class OCRResponse(BaseModel):
    """
    Complete OCR response.
    """

    success: bool = Field(
        description="Whether the request was successful"
    )
    message: str = Field(
        description="Status message or error description"
    )
    request_id: str = Field(
        description="Unique request identifier"
    )
    processing_time: float = Field(
        description="Total processing time in seconds"
    )
    results: List[OCRResult] = Field(
        default_factory=list,
        description="List of OCR results"
    )
    download_url: Optional[str] = Field(
        default=None,
        description="URL to download packaged results (ZIP file)"
    )
    total_pages: int = Field(
        default=0,
        description="Total number of pages/images processed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "OCR completed successfully",
                "request_id": "req_1234567890",
                "processing_time": 5.23,
                "results": [
                    {
                        "page_index": 0,
                        "markdown": "# Document Title\n\nContent...",
                        "extracted_images": ["images/0_0.jpg"],
                        "annotated_image": "annotated_0.jpg"
                    }
                ],
                "download_url": "/api/v1/download/req_1234567890.zip",
                "total_pages": 1
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(
        description="Service status: 'healthy', 'degraded', or 'unhealthy'"
    )
    version: str = Field(
        description="API version"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server timestamp"
    )
    engine_initialized: bool = Field(
        description="Whether the inference engine is initialized"
    )
    engine_mode: Optional[str] = Field(
        default=None,
        description="Engine mode: 'sync' or 'async'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "engine_initialized": True,
                "engine_mode": "sync"
            }
        }


class ConfigResponse(BaseModel):
    """
    Configuration response showing current settings.
    """

    model_path: str = Field(description="Model path")
    gpu_memory_utilization: float = Field(description="GPU memory utilization")
    tensor_parallel_size: int = Field(description="Tensor parallel size")
    max_model_len: int = Field(description="Maximum model length")
    engine_mode: str = Field(description="Engine mode")

    # Image processing
    image_size: int = Field(description="Local view image size")
    base_size: int = Field(description="Global view image size")
    min_crops: int = Field(description="Minimum crops")
    max_crops: int = Field(description="Maximum crops")
    crop_mode: bool = Field(description="Crop mode enabled")

    # Sampling
    temperature: float = Field(description="Default temperature")
    max_tokens: int = Field(description="Default max tokens")
    ngram_size: int = Field(description="N-gram size")
    window_size: int = Field(description="Window size")

    # PDF
    pdf_dpi: int = Field(description="PDF conversion DPI")

    # Server
    host: str = Field(description="Server host")
    port: int = Field(description="Server port")

    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "deepseek-ai/DeepSeek-OCR-2",
                "gpu_memory_utilization": 0.8,
                "tensor_parallel_size": 1,
                "max_model_len": 8192,
                "engine_mode": "sync",
                "image_size": 768,
                "base_size": 1024,
                "min_crops": 2,
                "max_crops": 6,
                "crop_mode": True,
                "temperature": 0.0,
                "max_tokens": 8192,
                "ngram_size": 20,
                "window_size": 90,
                "pdf_dpi": 144,
                "host": "0.0.0.0",
                "port": 8000
            }
        }


class EngineStatusResponse(BaseModel):
    """
    Engine status response.
    """

    initialized: bool = Field(
        description="Whether engine is initialized"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Engine mode if initialized"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Loaded model path"
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=None,
        description="GPU memory utilization setting"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "initialized": True,
                "mode": "sync",
                "model_path": "deepseek-ai/DeepSeek-OCR-2",
                "gpu_memory_utilization": 0.8
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response.
    """

    success: bool = Field(default=False)
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid file format. Expected PDF or image.",
                "detail": {"allowed_formats": ["pdf", "png", "jpg", "jpeg", "webp"]}
            }
        }
