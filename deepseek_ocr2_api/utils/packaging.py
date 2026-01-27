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
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def create_result_package(
    results: List[Dict[str, Any]],
    output_dir: str,
    package_name: Optional[str] = None,
    include_metadata: bool = True,
) -> str:
    """
    Create a ZIP package containing all OCR results.

    Args:
        results: List of result dictionaries from process_output().
        output_dir: Directory containing result files.
        package_name: Optional name for the ZIP file.
        include_metadata: Whether to include metadata JSON.

    Returns:
        Path to the created ZIP file.
    """
    if package_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"ocr_results_{timestamp}"

    zip_path = os.path.join(output_dir, f"{package_name}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Collect all markdown content
        all_markdown = []

        for idx, result in enumerate(results):
            # Add markdown file for each page
            markdown_content = result.get("markdown", "")
            all_markdown.append(markdown_content)

            md_filename = f"page_{idx}.md"
            zipf.writestr(md_filename, markdown_content)

            # Add raw output if present
            raw_output = result.get("raw_output")
            if raw_output:
                raw_filename = f"page_{idx}_raw.txt"
                zipf.writestr(raw_filename, raw_output)

            # Add annotated image
            annotated_path = result.get("annotated_image_path")
            if annotated_path and os.path.exists(annotated_path):
                arcname = f"annotated/page_{idx}.jpg"
                zipf.write(annotated_path, arcname)

            # Add extracted images
            for img_path in result.get("extracted_images", []):
                if os.path.exists(img_path):
                    img_name = os.path.basename(img_path)
                    arcname = f"images/{img_name}"
                    zipf.write(img_path, arcname)

        # Add combined markdown
        combined_md = "\n\n---\n\n".join(all_markdown)
        zipf.writestr("combined.md", combined_md)

        # Also save combined markdown to output_dir for preview
        combined_md_path = os.path.join(output_dir, "combined.md")
        with open(combined_md_path, 'w', encoding='utf-8') as f:
            f.write(combined_md)

        # Add metadata
        if include_metadata:
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_pages": len(results),
                "files": {
                    "markdown": [f"page_{i}.md" for i in range(len(results))],
                    "combined": "combined.md",
                    "annotated": [f"annotated/page_{i}.jpg" for i in range(len(results))
                                 if results[i].get("annotated_image_path")],
                    "images": []
                }
            }

            # Collect all image names
            for result in results:
                for img_path in result.get("extracted_images", []):
                    img_name = os.path.basename(img_path)
                    metadata["files"]["images"].append(f"images/{img_name}")

            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    logger.info(f"Created result package: {zip_path}")
    return zip_path


def create_pdf_result_package(
    results: List[Dict[str, Any]],
    output_dir: str,
    annotated_pdf_path: Optional[str] = None,
    original_filename: str = "document",
    page_separator: str = "\n<--- Page Split --->\n",
) -> str:
    """
    Create a ZIP package for PDF OCR results.

    Args:
        results: List of result dictionaries.
        output_dir: Directory containing result files.
        annotated_pdf_path: Path to annotated PDF if generated.
        original_filename: Original PDF filename (without extension).
        page_separator: Separator between pages.

    Returns:
        Path to the created ZIP file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"{original_filename}_{timestamp}"
    zip_path = os.path.join(output_dir, f"{package_name}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Combine all markdown with page separators
        all_markdown = []
        all_raw = []

        for idx, result in enumerate(results):
            markdown = result.get("markdown", "")
            all_markdown.append(markdown)

            raw = result.get("raw_output", "")
            if raw:
                all_raw.append(raw)

        # Combined markdown
        combined_md = page_separator.join(all_markdown)
        zipf.writestr(f"{original_filename}.md", combined_md)

        # Also save combined markdown to output_dir for preview
        combined_md_path = os.path.join(output_dir, f"{original_filename}.md")
        with open(combined_md_path, 'w', encoding='utf-8') as f:
            f.write(combined_md)

        # Combined raw output
        if all_raw:
            combined_raw = page_separator.join(all_raw)
            zipf.writestr(f"{original_filename}_raw.md", combined_raw)

        # Add annotated PDF
        if annotated_pdf_path and os.path.exists(annotated_pdf_path):
            zipf.write(annotated_pdf_path, f"{original_filename}_annotated.pdf")

        # Add extracted images
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                if os.path.isfile(img_path):
                    zipf.write(img_path, f"images/{img_file}")

        # Add metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "original_filename": original_filename,
            "total_pages": len(results),
            "files": {
                "markdown": f"{original_filename}.md",
                "annotated_pdf": f"{original_filename}_annotated.pdf" if annotated_pdf_path else None,
                "images_dir": "images/"
            }
        }
        zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    logger.info(f"Created PDF result package: {zip_path}")
    return zip_path


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
