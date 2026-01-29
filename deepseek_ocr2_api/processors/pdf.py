"""
PDF processing module for DeepSeek-OCR-2 API Server.

Handles PDF to image conversion and PDF output generation.
"""

import io
import logging
from typing import List, Optional, Union, BinaryIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(
    source: Union[str, Path, BinaryIO, bytes],
    dpi: int = 144,
    image_format: str = "PNG",
    num_workers: int = 4,
) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images with parallel processing.

    Args:
        source: PDF source - can be:
            - File path (str or Path)
            - File-like object (BinaryIO)
            - Bytes data
        dpi: Resolution for conversion (default 144).
        image_format: Output format (PNG or JPEG).
        num_workers: Number of parallel workers for conversion (default 4).

    Returns:
        List of PIL Image objects, one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install PyMuPDF")

    from concurrent.futures import ThreadPoolExecutor

    # Allow large images
    Image.MAX_IMAGE_PIXELS = None

    # Calculate zoom factor
    zoom = dpi / 72.0

    try:
        # Open PDF from different sources
        if isinstance(source, (str, Path)):
            pdf_path = str(source)
            pdf_bytes = None
        elif isinstance(source, bytes):
            pdf_path = None
            pdf_bytes = source
        else:
            # File-like object - read contents
            pdf_path = None
            pdf_bytes = source.read()

        # Get page count first
        if pdf_path:
            pdf_document = fitz.open(pdf_path)
        else:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        page_count = pdf_document.page_count
        pdf_document.close()

        def convert_page(page_num: int) -> tuple:
            """Convert a single page to PIL Image."""
            # Each thread needs its own document handle
            if pdf_path:
                doc = fitz.open(pdf_path)
            else:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            try:
                page = doc[page_num]
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Convert to PIL Image
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Handle RGBA conversion for JPEG
                if image_format.upper() == "JPEG" and img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                
                # Make a copy to ensure the image data is fully loaded
                img = img.copy()
                return page_num, img
            finally:
                doc.close()

        # Use ThreadPoolExecutor for parallel conversion
        effective_workers = min(num_workers, page_count)
        
        if effective_workers <= 1 or page_count <= 2:
            # For small PDFs, use sequential processing
            images = []
            if pdf_path:
                doc = fitz.open(pdf_path)
            else:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            try:
                matrix = fitz.Matrix(zoom, zoom)
                for page_num in range(page_count):
                    page = doc[page_num]
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    img_data = pixmap.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    if image_format.upper() == "JPEG" and img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    
                    images.append(img)
            finally:
                doc.close()
            
            return images
        else:
            # Parallel processing for larger PDFs
            logger.info(f"Converting PDF with {effective_workers} workers ({page_count} pages)")
            
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                results = list(executor.map(convert_page, range(page_count)))
            
            # Sort by page number and extract images
            results.sort(key=lambda x: x[0])
            images = [img for _, img in results]
            
            logger.info(f"PDF conversion complete: {len(images)} pages")
            return images

    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise


async def pdf_to_images_from_upload(
    upload_file, 
    dpi: int = 144,
    num_workers: int = 4,
) -> List[Image.Image]:
    """
    Convert PDF from FastAPI UploadFile to images.

    Args:
        upload_file: FastAPI UploadFile object.
        dpi: Resolution for conversion.
        num_workers: Number of parallel workers for conversion.

    Returns:
        List of PIL Image objects.
    """
    try:
        contents = await upload_file.read()
        return pdf_to_images(contents, dpi=dpi, num_workers=num_workers)
    except Exception as e:
        logger.error(f"Failed to convert uploaded PDF: {e}")
        raise
    finally:
        await upload_file.seek(0)


def images_to_pdf(
    images: List[Image.Image],
    output_path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Convert PIL Images to PDF.

    Args:
        images: List of PIL Image objects.
        output_path: Output PDF file path.
        quality: JPEG quality for compression.
    """
    try:
        import img2pdf
    except ImportError:
        raise ImportError("img2pdf is required for PDF generation. Install with: pip install img2pdf")

    if not images:
        logger.warning("No images to convert to PDF")
        return

    image_bytes_list = []

    for img in images:
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to JPEG bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=quality)
        image_bytes_list.append(img_buffer.getvalue())

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"PDF saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to create PDF: {e}")
        raise


def get_pdf_info(source: Union[str, Path, bytes]) -> dict:
    """
    Get information about a PDF file.

    Args:
        source: PDF source (path or bytes).

    Returns:
        Dict with PDF information.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

    try:
        if isinstance(source, (str, Path)):
            pdf_document = fitz.open(str(source))
        else:
            pdf_document = fitz.open(stream=source, filetype="pdf")

        info = {
            "page_count": pdf_document.page_count,
            "metadata": pdf_document.metadata,
            "pages": []
        }

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            rect = page.rect
            info["pages"].append({
                "page_number": page_num + 1,
                "width": rect.width,
                "height": rect.height,
            })

        pdf_document.close()
        return info

    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        raise
