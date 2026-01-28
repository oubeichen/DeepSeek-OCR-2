"""
OCR routes for DeepSeek-OCR-2 API Server.

Provides endpoints for image and PDF OCR processing.
"""

import os
import time
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response, JSONResponse
from PIL import Image as PILImage

from ..config import get_settings
from ..engine.manager import EngineManager
from ..engine.inference import (
    async_generate_batch,
    create_sampling_params,
    prepare_image_input,
    prepare_batch_inputs,
)
from ..processors.image import load_image_from_upload, get_image_info
from ..processors.pdf import pdf_to_images_from_upload, images_to_pdf
from ..processors.postprocess import (
    process_output,
    extract_refs,
    clean_output,
    replace_image_refs,
)
from ..utils.packaging import (
    create_result_package,
    create_pdf_result_package,
    create_temp_directory,
    cleanup_temp_files,
    PackageResult,
)
from ..utils import unescape_string
from ..schemas.request import ImageOCRRequest, PDFOCRRequest, BatchOCRRequest, ResultFormat
from ..schemas.response import OCRResponse, OCRResult, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR"])

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
SUPPORTED_PDF_FORMATS = {".pdf"}


def validate_image_file(filename: str) -> bool:
    """Check if file is a supported image format."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUPPORTED_IMAGE_FORMATS


def validate_pdf_file(filename: str) -> bool:
    """Check if file is a PDF."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUPPORTED_PDF_FORMATS


@router.post(
    "/image",
    summary="Single Image OCR",
    description="""
Process a single image and extract text content as markdown.

**Supported formats:** PNG, JPG, JPEG, WebP, BMP, TIFF, GIF

**Returns:** Based on result_format parameter:
- zip: ZIP file with markdown, annotated image, extracted images, and metadata
- markdown: Plain text markdown content
- json: doc.json with structured data

**Parameters can be passed as form fields along with the file upload.**
    """,
    responses={
        200: {
            "description": "OCR results in requested format",
            "content": {
                "application/zip": {},
                "text/markdown": {},
                "application/json": {}
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    }
)
async def ocr_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(default=None, description="Custom prompt (leave empty to use default)", examples=[""]),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature"),
    max_tokens: Optional[int] = Form(default=None, description="Maximum tokens"),
    ngram_size: Optional[int] = Form(default=None, description="N-gram size"),
    window_size: Optional[int] = Form(default=None, description="Window size"),
    crop_mode: Optional[bool] = Form(default=None, description="Enable dynamic cropping"),
    return_raw_output: bool = Form(default=False, description="Include raw output"),
    return_annotated_image: bool = Form(default=True, description="Include annotated image"),
    extract_images: bool = Form(default=True, description="Extract image regions"),
    result_format: ResultFormat = Form(default=ResultFormat.ZIP, description="Output format: zip, markdown, or json"),
):
    """
    Process a single image for OCR.
    """
    start_time = time.time()
    request_id = f"img_{uuid.uuid4().hex[:12]}"
    temp_dir = None

    try:
        # Validate file
        if not file.filename or not validate_image_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )

        # Check engine
        manager = EngineManager.get_instance()
        if not manager.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Engine not initialized. Please wait for startup to complete."
            )

        settings = get_settings()

        # Create temp directory
        temp_dir = create_temp_directory(prefix=f"ocr_{request_id}_")

        # Load image
        logger.info(f"[{request_id}] Loading image: {file.filename}")
        image = await load_image_from_upload(file)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        image = image.convert('RGB')
        logger.info(f"[{request_id}] Image size: {image.size}")

        # Prepare prompt
        actual_prompt = unescape_string(prompt) if prompt else settings.default_prompt

        # Prepare input
        _crop_mode = crop_mode if crop_mode is not None else settings.crop_mode
        input_data = prepare_image_input(image, actual_prompt, _crop_mode)

        # Create sampling params
        sampling_params = create_sampling_params(
            settings=settings,
            temperature=temperature,
            max_tokens=max_tokens,
            ngram_size=ngram_size,
            window_size=window_size,
        )

        # Generate
        logger.info(f"[{request_id}] Starting inference...")
        outputs = await async_generate_batch([input_data], sampling_params, settings)
        output_text = outputs[0] if outputs else ""

        logger.info(f"[{request_id}] Inference complete, processing output...")

        # Post-process
        result = process_output(
            text=output_text,
            image=image,
            output_dir=temp_dir,
            page_index=0,
            save_annotated=return_annotated_image,
            extract_images=extract_images,
        )

        if not return_raw_output:
            result["raw_output"] = None

        # Package results
        package_result = create_result_package(
            results=[result],
            output_dir=temp_dir,
            package_name=request_id,
            include_raw_output=return_raw_output,
            result_format=result_format.value,
        )

        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Complete in {processing_time:.2f}s")

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, [temp_dir], True)

        # Return result in requested format
        if result_format == ResultFormat.MARKDOWN:
            return Response(content=package_result.content, media_type="text/markdown; charset=utf-8")
        elif result_format == ResultFormat.JSON:
            return JSONResponse(content=package_result.content)
        else:
            return FileResponse(
                path=package_result.path,
                media_type=package_result.media_type,
                filename=package_result.filename,
                background=background_tasks,
            )

    except HTTPException:
        if temp_dir:
            cleanup_temp_files([temp_dir], force=True)
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        if temp_dir:
            cleanup_temp_files([temp_dir], force=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/pdf",
    summary="PDF OCR",
    description="""
Process a PDF document and extract text content as markdown.

**Returns:** Based on result_format parameter:
- zip: ZIP file with combined markdown, annotated PDF, extracted images, and metadata
- markdown: Plain text markdown content with all pages
- json: doc.json with structured data

**Parameters can be passed as form fields along with the file upload.**
    """,
    responses={
        200: {
            "description": "OCR results in requested format",
            "content": {
                "application/zip": {},
                "text/markdown": {},
                "application/json": {}
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    }
)
async def ocr_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process"),
    prompt: Optional[str] = Form(default=None, description="Custom prompt (leave empty to use default)", examples=[""]),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature"),
    max_tokens: Optional[int] = Form(default=None, description="Maximum tokens"),
    ngram_size: Optional[int] = Form(default=None, description="N-gram size"),
    window_size: Optional[int] = Form(default=None, description="Window size"),
    crop_mode: Optional[bool] = Form(default=None, description="Enable dynamic cropping"),
    dpi: Optional[int] = Form(default=None, description="PDF conversion DPI"),
    page_separator: Optional[str] = Form(default=None, description="Page separator (leave empty to use default)", examples=[""]),
    skip_repeat_pages: Optional[bool] = Form(default=None, description="Skip repeated pages"),
    generate_annotated_pdf: bool = Form(default=True, description="Generate annotated PDF"),
    return_raw_output: bool = Form(default=False, description="Include raw output"),
    extract_images: bool = Form(default=True, description="Extract image regions"),
    result_format: ResultFormat = Form(default=ResultFormat.ZIP, description="Output format: zip, markdown, or json"),
):
    """
    Process a PDF document for OCR.
    """
    start_time = time.time()
    request_id = f"pdf_{uuid.uuid4().hex[:12]}"
    temp_dir = None

    try:
        # Validate file
        if not file.filename or not validate_pdf_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only PDF files are supported."
            )

        # Check engine
        manager = EngineManager.get_instance()
        if not manager.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Engine not initialized. Please wait for startup to complete."
            )

        settings = get_settings()

        # Create temp directory
        temp_dir = create_temp_directory(prefix=f"ocr_{request_id}_")

        # Convert PDF to images
        _dpi = dpi or settings.pdf_dpi
        logger.info(f"[{request_id}] Converting PDF to images (DPI={_dpi})...")
        images = await pdf_to_images_from_upload(file, dpi=_dpi)
        logger.info(f"[{request_id}] PDF has {len(images)} pages")

        # Prepare prompt
        actual_prompt = unescape_string(prompt) if prompt else settings.default_prompt

        # Prepare batch inputs
        _crop_mode = crop_mode if crop_mode is not None else settings.crop_mode
        batch_inputs = prepare_batch_inputs(
            images=images,
            prompt=actual_prompt,
            crop_mode=_crop_mode,
            num_workers=settings.num_workers,
        )

        # Create sampling params (use smaller window for PDF)
        _window_size = window_size or 50  # Smaller window for PDF
        sampling_params = create_sampling_params(
            settings=settings,
            temperature=temperature,
            max_tokens=max_tokens,
            ngram_size=ngram_size,
            window_size=_window_size,
            include_stop_str_in_output=True,
        )

        # Generate
        logger.info(f"[{request_id}] Starting batch inference...")
        outputs = await async_generate_batch(batch_inputs, sampling_params, settings)

        logger.info(f"[{request_id}] Inference complete, processing outputs...")

        # Post-process each page
        results = []
        annotated_images = []
        _skip_repeat = skip_repeat_pages if skip_repeat_pages is not None else settings.skip_repeat_pages

        for idx, (output_text, image) in enumerate(zip(outputs, images)):
            # Check for incomplete output (no EOS token)
            if '<｜end▁of▁sentence｜>' not in output_text and _skip_repeat:
                logger.warning(f"[{request_id}] Page {idx} appears incomplete, skipping")
                continue

            result = process_output(
                text=output_text,
                image=image,
                output_dir=temp_dir,
                page_index=idx,
                save_annotated=True,
                extract_images=extract_images,
            )

            if not return_raw_output:
                result["raw_output"] = None

            results.append(result)

            # Collect annotated images for PDF
            if result.get("annotated_image_path"):
                annotated_images.append(PILImage.open(result["annotated_image_path"]))

        # Generate annotated PDF
        annotated_pdf_path = None
        if generate_annotated_pdf and annotated_images:
            annotated_pdf_path = os.path.join(temp_dir, "annotated.pdf")
            images_to_pdf(annotated_images, annotated_pdf_path)

        # Package results
        original_name = os.path.splitext(file.filename)[0]
        _page_sep = unescape_string(page_separator) if page_separator else settings.page_separator

        package_result = create_pdf_result_package(
            results=results,
            output_dir=temp_dir,
            annotated_pdf_path=annotated_pdf_path,
            original_filename=original_name,
            page_separator=_page_sep,
            include_raw_output=return_raw_output,
            result_format=result_format.value,
        )

        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Complete in {processing_time:.2f}s ({len(results)} pages)")

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, [temp_dir], True)

        # Return result in requested format
        if result_format == ResultFormat.MARKDOWN:
            return Response(content=package_result.content, media_type="text/markdown; charset=utf-8")
        elif result_format == ResultFormat.JSON:
            return JSONResponse(content=package_result.content)
        else:
            return FileResponse(
                path=package_result.path,
                media_type=package_result.media_type,
                filename=package_result.filename,
                background=background_tasks,
            )

    except HTTPException:
        if temp_dir:
            cleanup_temp_files([temp_dir], force=True)
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        if temp_dir:
            cleanup_temp_files([temp_dir], force=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch",
    summary="Batch Image OCR",
    description="""
Process multiple images in a single request.

**Supported formats:** PNG, JPG, JPEG, WebP, BMP, TIFF, GIF

**Returns:** Based on result_format parameter:
- zip: ZIP file with results for all images
- markdown: Plain text markdown content with all images
- json: doc.json with structured data
    """,
    responses={
        200: {
            "description": "OCR results in requested format",
            "content": {
                "application/zip": {},
                "text/markdown": {},
                "application/json": {}
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    }
)
async def ocr_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Image files to process"),
    prompt: Optional[str] = Form(default=None, description="Custom prompt (leave empty to use default)", examples=[""]),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature"),
    max_tokens: Optional[int] = Form(default=None, description="Maximum tokens"),
    ngram_size: Optional[int] = Form(default=None, description="N-gram size"),
    window_size: Optional[int] = Form(default=None, description="Window size"),
    crop_mode: Optional[bool] = Form(default=None, description="Enable dynamic cropping"),
    num_workers: Optional[int] = Form(default=None, description="Number of preprocessing workers"),
    return_raw_output: bool = Form(default=False, description="Include raw output"),
    return_annotated_image: bool = Form(default=True, description="Include annotated images"),
    extract_images: bool = Form(default=True, description="Extract image regions"),
    result_format: ResultFormat = Form(default=ResultFormat.ZIP, description="Output format: zip, markdown, or json"),
):
    """
    Process multiple images for OCR in batch.
    """
    start_time = time.time()
    request_id = f"batch_{uuid.uuid4().hex[:12]}"
    temp_dir = None

    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        for f in files:
            if not f.filename or not validate_image_file(f.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format: {f.filename}. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
                )

        # Check engine
        manager = EngineManager.get_instance()
        if not manager.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Engine not initialized. Please wait for startup to complete."
            )

        settings = get_settings()

        # Create temp directory
        temp_dir = create_temp_directory(prefix=f"ocr_{request_id}_")

        # Load all images
        logger.info(f"[{request_id}] Loading {len(files)} images...")
        images = []
        for f in files:
            image = await load_image_from_upload(f)
            if image is None:
                raise HTTPException(status_code=400, detail=f"Failed to load image: {f.filename}")
            images.append(image.convert('RGB'))

        # Prepare prompt
        actual_prompt = unescape_string(prompt) if prompt else settings.default_prompt

        # Prepare batch inputs
        _crop_mode = crop_mode if crop_mode is not None else settings.crop_mode
        _num_workers = num_workers or settings.num_workers

        batch_inputs = prepare_batch_inputs(
            images=images,
            prompt=actual_prompt,
            crop_mode=_crop_mode,
            num_workers=_num_workers,
        )

        # Create sampling params
        _ngram_size = ngram_size or 40  # Larger ngram for batch
        sampling_params = create_sampling_params(
            settings=settings,
            temperature=temperature,
            max_tokens=max_tokens,
            ngram_size=_ngram_size,
            window_size=window_size,
        )

        # Generate
        logger.info(f"[{request_id}] Starting batch inference for {len(images)} images...")
        outputs = await async_generate_batch(batch_inputs, sampling_params, settings)

        logger.info(f"[{request_id}] Inference complete, processing outputs...")

        # Post-process each image
        results = []
        for idx, (output_text, image) in enumerate(zip(outputs, images)):
            result = process_output(
                text=output_text,
                image=image,
                output_dir=temp_dir,
                page_index=idx,
                save_annotated=return_annotated_image,
                extract_images=extract_images,
            )

            if not return_raw_output:
                result["raw_output"] = None

            results.append(result)

        # Package results
        package_result = create_result_package(
            results=results,
            output_dir=temp_dir,
            package_name=request_id,
            include_raw_output=return_raw_output,
            result_format=result_format.value,
        )

        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Complete in {processing_time:.2f}s ({len(results)} images)")

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, [temp_dir], True)

        # Return result in requested format
        if result_format == ResultFormat.MARKDOWN:
            return Response(content=package_result.content, media_type="text/markdown; charset=utf-8")
        elif result_format == ResultFormat.JSON:
            return JSONResponse(content=package_result.content)
        else:
            return FileResponse(
                path=package_result.path,
                media_type=package_result.media_type,
                filename=package_result.filename,
                background=background_tasks,
            )

    except HTTPException:
        if temp_dir:
            cleanup_temp_files([temp_dir], force=True)
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        if temp_dir:
            cleanup_temp_files([temp_dir], force=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/image/json",
    response_model=OCRResponse,
    summary="Single Image OCR (JSON Response)",
    description="""
Process a single image and return results as JSON instead of ZIP file.

Useful for programmatic access when you don't need the packaged files.
    """,
)
async def ocr_image_json(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(default=None, description="Custom prompt (leave empty to use default)", examples=[""]),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature"),
    max_tokens: Optional[int] = Form(default=None, description="Maximum tokens"),
    ngram_size: Optional[int] = Form(default=None, description="N-gram size"),
    window_size: Optional[int] = Form(default=None, description="Window size"),
    crop_mode: Optional[bool] = Form(default=None, description="Enable dynamic cropping"),
    return_raw_output: bool = Form(default=False, description="Include raw output"),
) -> OCRResponse:
    """
    Process a single image and return JSON response.
    """
    start_time = time.time()
    request_id = f"img_{uuid.uuid4().hex[:12]}"

    try:
        # Validate file
        if not file.filename or not validate_image_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )

        # Check engine
        manager = EngineManager.get_instance()
        if not manager.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Engine not initialized. Please wait for startup to complete."
            )

        settings = get_settings()

        # Load image
        image = await load_image_from_upload(file)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        image = image.convert('RGB')

        # Prepare prompt
        actual_prompt = unescape_string(prompt) if prompt else settings.default_prompt

        # Prepare input
        _crop_mode = crop_mode if crop_mode is not None else settings.crop_mode
        input_data = prepare_image_input(image, actual_prompt, _crop_mode)

        # Create sampling params
        sampling_params = create_sampling_params(
            settings=settings,
            temperature=temperature,
            max_tokens=max_tokens,
            ngram_size=ngram_size,
            window_size=window_size,
        )

        # Generate
        outputs = await async_generate_batch([input_data], sampling_params, settings)
        output_text = outputs[0] if outputs else ""

        # Simple post-processing (no file saving)
        refs, image_refs, other_refs = extract_refs(output_text)
        markdown = replace_image_refs(output_text, image_refs, "images", 0)
        markdown = clean_output(markdown, other_refs)

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            message="OCR completed successfully",
            request_id=request_id,
            processing_time=processing_time,
            results=[
                OCRResult(
                    page_index=0,
                    markdown=markdown,
                    raw_output=output_text if return_raw_output else None,
                    extracted_images=[],
                    annotated_image=None,
                )
            ],
            total_pages=1,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
