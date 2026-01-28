"""
Packaging utilities for DeepSeek-OCR-2 API Server.

Handles result packaging into ZIP files and temporary file management.
"""

import os
import json
import shutil
import zipfile
import tempfile
import logging
from typing import List, Dict, Any, Optional, NamedTuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class PackageResult(NamedTuple):
    """Result of packaging operation."""
    path: Optional[str]  # File path for zip, None for text formats
    media_type: str
    filename: str
    content: Optional[str] = None  # Text content for markdown/json, None for zip


def create_result_package(
    results: List[Dict[str, Any]],
    output_dir: str,
    package_name: Optional[str] = None,
    include_metadata: bool = True,
    include_raw_output: bool = False,
    result_format: str = "zip",
) -> PackageResult:
    """
    Create a result package containing OCR results.

    Args:
        results: List of result dictionaries from process_output().
        output_dir: Directory containing result files.
        package_name: Optional name for the output file.
        include_metadata: Whether to include metadata JSON (zip only).
        include_raw_output: Whether to include raw model output in doc.json.
        result_format: Output format - 'zip', 'markdown', or 'json'.

    Returns:
        PackageResult with path/content, media_type, and filename.
    """
    if package_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"ocr_results_{timestamp}"

    # Collect all markdown content
    all_markdown = []
    for result in results:
        markdown_content = result.get("markdown", "")
        all_markdown.append(markdown_content)

    combined_md = "\n\n---\n\n".join(all_markdown)

    # Return markdown directly without writing to file
    if result_format == "markdown":
        logger.info(f"Returning markdown content directly ({len(combined_md)} chars)")
        return PackageResult(
            path=None,
            media_type="text/markdown; charset=utf-8",
            filename=f"{package_name}.md",
            content=combined_md,
        )

    # Build doc.json structure
    document_data = {
        "created_at": datetime.now().isoformat(),
        "total_pages": len(results),
        "pages": [],
    }

    for idx, result in enumerate(results):
        raw_output = result.get("raw_output")
        annotated_path = result.get("annotated_image_path")
        annotated_arcname = f"annotated/page_{idx}.jpg" if annotated_path and os.path.exists(annotated_path) else None

        extracted_arcnames = []
        for img_path in result.get("extracted_images", []):
            if os.path.exists(img_path):
                img_name = os.path.basename(img_path)
                extracted_arcnames.append(f"images/{img_name}")

        page_data = {
            "page_index": idx,
            "annotated_image": annotated_arcname,
            "extracted_images": extracted_arcnames,
            "elements": result.get("elements", []),
            "image_info": result.get("image_info", {}),
        }
        if include_raw_output and raw_output:
            page_data["raw_output"] = raw_output
        document_data["pages"].append(page_data)

    document_json_content = json.dumps(document_data, indent=2, ensure_ascii=False)

    # Return json directly without writing to file
    if result_format == "json":
        logger.info(f"Returning JSON content directly ({len(document_json_content)} chars)")
        return PackageResult(
            path=None,
            media_type="application/json",
            filename=f"{package_name}.json",
            content=document_json_content,
        )

    # Default: create ZIP package
    zip_path = os.path.join(output_dir, f"{package_name}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for idx, result in enumerate(results):
            markdown_content = result.get("markdown", "")
            md_filename = f"page_{idx}.md"
            zipf.writestr(md_filename, markdown_content)

            raw_output = result.get("raw_output")
            if raw_output:
                raw_filename = f"page_{idx}_raw.txt"
                zipf.writestr(raw_filename, raw_output)

            annotated_path = result.get("annotated_image_path")
            if annotated_path and os.path.exists(annotated_path):
                zipf.write(annotated_path, f"annotated/page_{idx}.jpg")

            for img_path in result.get("extracted_images", []):
                if os.path.exists(img_path):
                    img_name = os.path.basename(img_path)
                    zipf.write(img_path, f"images/{img_name}")

        zipf.writestr("combined.md", combined_md)
        zipf.writestr("doc.json", document_json_content)

        if include_metadata:
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_pages": len(results),
                "files": {
                    "markdown": [f"page_{i}.md" for i in range(len(results))],
                    "combined": "combined.md",
                    "document_json": "doc.json",
                    "annotated": [f"annotated/page_{i}.jpg" for i in range(len(results))
                                 if results[i].get("annotated_image_path")],
                    "images": []
                }
            }
            for result in results:
                for img_path in result.get("extracted_images", []):
                    img_name = os.path.basename(img_path)
                    metadata["files"]["images"].append(f"images/{img_name}")
            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    logger.info(f"Created result package: {zip_path}")
    return PackageResult(
        path=zip_path,
        media_type="application/zip",
        filename=f"{package_name}.zip"
    )


def create_pdf_result_package(
    results: List[Dict[str, Any]],
    output_dir: str,
    annotated_pdf_path: Optional[str] = None,
    original_filename: str = "document",
    page_separator: str = "\n<--- Page Split --->\n",
    include_raw_output: bool = False,
    result_format: str = "zip",
) -> PackageResult:
    """
    Create a result package for PDF OCR results.

    Args:
        results: List of result dictionaries.
        output_dir: Directory containing result files.
        annotated_pdf_path: Path to annotated PDF if generated.
        original_filename: Original PDF filename (without extension).
        page_separator: Separator between pages.
        include_raw_output: Whether to include raw model output in doc.json.
        result_format: Output format - 'zip', 'markdown', or 'json'.

    Returns:
        PackageResult with path, media_type, and filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"{original_filename}_{timestamp}"

    # Combine all markdown with page separators
    all_markdown = []
    all_raw = []

    for result in results:
        markdown = result.get("markdown", "")
        all_markdown.append(markdown)
        raw = result.get("raw_output", "")
        if raw:
            all_raw.append(raw)

    combined_md = page_separator.join(all_markdown)

    # Build doc.json structure
    document_data = {
        "created_at": datetime.now().isoformat(),
        "original_filename": original_filename,
        "total_pages": len(results),
        "annotated_pdf": f"{original_filename}_annotated.pdf" if annotated_pdf_path else None,
        "pages": [],
    }

    for idx, result in enumerate(results):
        raw = result.get("raw_output", "")
        extracted_arcnames = []
        for img_path in result.get("extracted_images", []):
            if os.path.exists(img_path):
                img_name = os.path.basename(img_path)
                extracted_arcnames.append(f"images/{img_name}")

        annotated_image = result.get("annotated_image_path")
        annotated_arcname = f"annotated/page_{idx}.jpg" if annotated_image and os.path.exists(annotated_image) else None

        page_data = {
            "page_index": idx,
            "annotated_image": annotated_arcname,
            "extracted_images": extracted_arcnames,
            "elements": result.get("elements", []),
            "image_info": result.get("image_info", {}),
        }
        if include_raw_output and raw:
            page_data["raw_output"] = raw
        document_data["pages"].append(page_data)

    # Return based on format - avoid file I/O for markdown and json
    if result_format == "markdown":
        logger.info(f"Returning markdown content directly for {original_filename}")
        return PackageResult(
            path=None,
            media_type="text/markdown; charset=utf-8",
            filename=f"{original_filename}.md",
            content=combined_md
        )

    if result_format == "json":
        logger.info(f"Returning JSON content directly for {original_filename}")
        return PackageResult(
            path=None,
            media_type="application/json",
            filename=f"{original_filename}.json",
            content=document_data  # Return dict directly for JSONResponse
        )

    # Default: create ZIP package (requires file I/O)
    document_json_content = json.dumps(document_data, indent=2, ensure_ascii=False)
    zip_path = os.path.join(output_dir, f"{package_name}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for idx, result in enumerate(results):
            annotated_image = result.get("annotated_image_path")
            if annotated_image and os.path.exists(annotated_image):
                zipf.write(annotated_image, f"annotated/page_{idx}.jpg")

        zipf.writestr(f"{original_filename}.md", combined_md)

        if all_raw:
            combined_raw = page_separator.join(all_raw)
            zipf.writestr(f"{original_filename}_raw.md", combined_raw)

        if annotated_pdf_path and os.path.exists(annotated_pdf_path):
            zipf.write(annotated_pdf_path, f"{original_filename}_annotated.pdf")

        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                if os.path.isfile(img_path):
                    zipf.write(img_path, f"images/{img_file}")

        zipf.writestr("doc.json", document_json_content)

        metadata = {
            "created_at": datetime.now().isoformat(),
            "original_filename": original_filename,
            "total_pages": len(results),
            "files": {
                "markdown": f"{original_filename}.md",
                "document_json": "doc.json",
                "annotated_pdf": f"{original_filename}_annotated.pdf" if annotated_pdf_path else None,
                "images_dir": "images/"
            }
        }
        zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    logger.info(f"Created PDF result package: {zip_path}")
    return PackageResult(
        path=zip_path,
        media_type="application/zip",
        filename=f"{original_filename}_ocr.zip"
    )


def create_temp_directory(prefix: str = "deepseek_ocr2_") -> str:
    """
    Create a temporary directory for processing.

    Args:
        prefix: Prefix for the directory name.

    Returns:
        Path to the created directory.
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
    logger.debug(f"Created temp directory: {temp_dir}")
    return temp_dir


def cleanup_temp_files(paths: List[str], force: bool = False) -> None:
    """
    Clean up temporary files and directories.

    Args:
        paths: List of file or directory paths to clean up.
        force: If True, ignore errors during cleanup.
    """
    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.debug(f"Removed file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                logger.debug(f"Removed directory: {path}")
        except Exception as e:
            if force:
                logger.warning(f"Failed to cleanup {path}: {e}")
            else:
                raise


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.

    Args:
        path: File path.

    Returns:
        File size in bytes.
    """
    return os.path.getsize(path) if os.path.exists(path) else 0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "1.5 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
