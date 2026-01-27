"""
Image processing module for DeepSeek-OCR-2 API Server.

Handles image loading, preprocessing, and EXIF correction.
"""

import io
import logging
from typing import Union, Optional, BinaryIO
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image(
    source: Union[str, Path, BinaryIO, bytes],
    convert_rgb: bool = True,
) -> Optional[Image.Image]:
    """
    Load an image from various sources with EXIF correction.

    Args:
        source: Image source - can be:
            - File path (str or Path)
            - File-like object (BinaryIO)
            - Bytes data
        convert_rgb: Whether to convert to RGB mode.

    Returns:
        PIL Image object or None if loading fails.
    """
    try:
        # Handle different source types
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        elif isinstance(source, bytes):
            image = Image.open(io.BytesIO(source))
        else:
            # Assume file-like object
            image = Image.open(source)

        # Apply EXIF transpose to correct orientation
        try:
            corrected_image = ImageOps.exif_transpose(image)
        except Exception as e:
            logger.warning(f"Failed to apply EXIF transpose: {e}")
            corrected_image = image

        # Convert to RGB if requested
        if convert_rgb and corrected_image.mode != 'RGB':
            corrected_image = corrected_image.convert('RGB')

        return corrected_image

    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


async def load_image_from_upload(upload_file) -> Optional[Image.Image]:
    """
    Load an image from FastAPI UploadFile.

    Args:
        upload_file: FastAPI UploadFile object.

    Returns:
        PIL Image object or None if loading fails.
    """
    try:
        contents = await upload_file.read()
        return load_image(contents, convert_rgb=True)
    except Exception as e:
        logger.error(f"Failed to load image from upload: {e}")
        return None
    finally:
        await upload_file.seek(0)


def preprocess_image(
    image: Image.Image,
    min_crops: int = 2,
    max_crops: int = 6,
    image_size: int = 768,
    base_size: int = 1024,
    crop_mode: bool = True,
) -> Image.Image:
    """
    Preprocess image for model input.

    This is a simplified version - the actual preprocessing is done
    by the DeepseekOCR2Processor in the engine module.

    Args:
        image: PIL Image to preprocess.
        min_crops: Minimum number of crops.
        max_crops: Maximum number of crops.
        image_size: Local view size.
        base_size: Global view size.
        crop_mode: Whether to use dynamic cropping.

    Returns:
        Preprocessed PIL Image.
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def get_image_info(image: Image.Image) -> dict:
    """
    Get information about an image.

    Args:
        image: PIL Image object.

    Returns:
        Dict with image information.
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
    }


def resize_image(
    image: Image.Image,
    max_size: int = 4096,
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Resize image if it exceeds maximum size.

    Args:
        image: PIL Image to resize.
        max_size: Maximum dimension (width or height).
        maintain_aspect: Whether to maintain aspect ratio.

    Returns:
        Resized PIL Image.
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if maintain_aspect:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
    else:
        new_width = min(width, max_size)
        new_height = min(height, max_size)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
