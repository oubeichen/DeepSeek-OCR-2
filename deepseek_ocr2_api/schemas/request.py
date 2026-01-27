"""
Request schemas for DeepSeek-OCR-2 API Server.

Defines Pydantic models for API request validation with OpenAPI documentation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    """
    Base OCR request parameters.

    All parameters are optional and will use server defaults if not provided.
    """

    prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for OCR. If not provided, uses default: '<image>\n<|grounding|>Convert the document to markdown.'"
    )

    # Sampling parameters
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for deterministic output. Default: 0.0"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate. Default: 8192"
    )

    # N-gram repetition penalty
    ngram_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="N-gram size for repetition penalty. Default: 20"
    )
    window_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Window size for N-gram repetition check. Default: 90"
    )

    # Processing options
    crop_mode: Optional[bool] = Field(
        default=None,
        description="Enable dynamic cropping for better quality. Default: True"
    )

    # Output options
    return_raw_output: bool = Field(
        default=False,
        description="Include raw model output in response"
    )
    return_annotated_image: bool = Field(
        default=True,
        description="Include annotated image with bounding boxes"
    )
    extract_images: bool = Field(
        default=True,
        description="Extract image regions from document"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "temperature": 0.0,
                "max_tokens": 8192,
                "crop_mode": True,
                "return_annotated_image": True,
                "extract_images": True
            }
        }


class ImageOCRRequest(OCRRequest):
    """
    Request parameters for single image OCR.

    Inherits all parameters from OCRRequest.
    """

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.0,
                "max_tokens": 8192,
                "crop_mode": True,
                "return_annotated_image": True
            }
        }


class PDFOCRRequest(OCRRequest):
    """
    Request parameters for PDF OCR.

    Extends OCRRequest with PDF-specific options.
    """

    dpi: Optional[int] = Field(
        default=None,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion. Higher = better quality but slower. Default: 144"
    )
    page_separator: Optional[str] = Field(
        default=None,
        description="Separator between pages in output. Default: '\n<--- Page Split --->\n'"
    )
    skip_repeat_pages: Optional[bool] = Field(
        default=None,
        description="Skip pages that appear to be repeated. Default: True"
    )
    generate_annotated_pdf: bool = Field(
        default=True,
        description="Generate PDF with bounding box annotations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dpi": 144,
                "temperature": 0.0,
                "max_tokens": 8192,
                "page_separator": "\n<--- Page Split --->\n",
                "generate_annotated_pdf": True
            }
        }


class BatchOCRRequest(OCRRequest):
    """
    Request parameters for batch image OCR.

    Extends OCRRequest with batch-specific options.
    """

    num_workers: Optional[int] = Field(
        default=None,
        ge=1,
        le=128,
        description="Number of worker threads for preprocessing. Default: 64"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.0,
                "max_tokens": 8192,
                "num_workers": 64,
                "crop_mode": True
            }
        }


class SyncModeRequest(BaseModel):
    """
    Request to specify sync/async mode.

    Can be combined with other request types.
    """

    sync_mode: bool = Field(
        default=True,
        description="Use synchronous mode (True) or asynchronous mode (False). Default: True"
    )


class ImageOCRRequestWithMode(ImageOCRRequest, SyncModeRequest):
    """Image OCR request with sync mode option."""
    pass


class PDFOCRRequestWithMode(PDFOCRRequest, SyncModeRequest):
    """PDF OCR request with sync mode option."""
    pass


class BatchOCRRequestWithMode(BatchOCRRequest, SyncModeRequest):
    """Batch OCR request with sync mode option."""
    pass
