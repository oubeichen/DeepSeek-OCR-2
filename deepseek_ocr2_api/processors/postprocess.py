"""
Post-processing module for DeepSeek-OCR-2 API Server.

Handles output parsing, coordinate conversion, bounding box drawing,
and image extraction from model outputs.
"""

import re
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class RefMatch:
    """Represents a matched reference in the output."""
    full_match: str
    label_type: str
    coordinates: List[List[int]]


def extract_refs(text: str) -> Tuple[List[RefMatch], List[str], List[str]]:
    """
    Extract reference tags from model output.

    Pattern: <|ref|>label<|/ref|><|det|>coordinates<|/det|>

    Args:
        text: Model output text.

    Returns:
        Tuple of:
            - List of RefMatch objects
            - List of image reference strings
            - List of other reference strings
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    ref_matches = []
    image_refs = []
    other_refs = []

    for match in matches:
        full_match, label_type, coords_str = match

        # Parse coordinates
        try:
            coords = eval(coords_str)
            if not isinstance(coords, list):
                coords = [coords]
        except Exception as e:
            logger.warning(f"Failed to parse coordinates: {e}")
            coords = []

        ref_match = RefMatch(
            full_match=full_match,
            label_type=label_type,
            coordinates=coords
        )
        ref_matches.append(ref_match)

        # Categorize
        if label_type == 'image':
            image_refs.append(full_match)
        else:
            other_refs.append(full_match)

    return ref_matches, image_refs, other_refs


def convert_coordinates(
    coords: List[int],
    image_width: int,
    image_height: int,
    coord_range: int = 999,
) -> Tuple[int, int, int, int]:
    """
    Convert model coordinates (0-999) to pixel coordinates.

    Args:
        coords: [x1, y1, x2, y2] in model coordinate space.
        image_width: Actual image width.
        image_height: Actual image height.
        coord_range: Model coordinate range (default 999).

    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates.
    """
    x1, y1, x2, y2 = coords
    x1 = int(x1 / coord_range * image_width)
    y1 = int(y1 / coord_range * image_height)
    x2 = int(x2 / coord_range * image_width)
    y2 = int(y2 / coord_range * image_height)
    return x1, y1, x2, y2


def draw_boxes(
    image: Image.Image,
    refs: List[RefMatch],
    output_dir: Optional[str] = None,
    page_index: int = 0,
) -> Tuple[Image.Image, List[str]]:
    """
    Draw bounding boxes on image and extract image regions.

    Args:
        image: PIL Image to draw on.
        refs: List of RefMatch objects.
        output_dir: Directory to save extracted images.
        page_index: Page index for naming extracted images.

    Returns:
        Tuple of:
            - Annotated image
            - List of extracted image paths
    """
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Create overlay for semi-transparent fills
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    # Load default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    extracted_images = []
    img_idx = 0

    for ref in refs:
        try:
            label_type = ref.label_type
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 255)
            )
            color_a = color + (20,)

            for coords in ref.coordinates:
                if len(coords) != 4:
                    continue

                x1, y1, x2, y2 = convert_coordinates(
                    coords, image_width, image_height
                )

                # Extract image regions
                if label_type == 'image' and output_dir:
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        img_path = os.path.join(
                            output_dir,
                            f"{page_index}_{img_idx}.jpg"
                        )
                        cropped.save(img_path)
                        extracted_images.append(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to extract image: {e}")
                    img_idx += 1

                # Draw bounding box
                try:
                    line_width = 4 if label_type == 'title' else 2
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                    # Draw label
                    if font:
                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30)
                        )
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception as e:
                    logger.warning(f"Failed to draw box: {e}")

        except Exception as e:
            logger.warning(f"Failed to process ref: {e}")
            continue

    # Composite overlay
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, extracted_images


def replace_image_refs(
    text: str,
    image_refs: List[str],
    image_dir: str = "images",
    page_index: int = 0,
) -> str:
    """
    Replace image reference tags with markdown image links.

    Args:
        text: Model output text.
        image_refs: List of image reference strings to replace.
        image_dir: Directory name for images in markdown.
        page_index: Page index for naming.

    Returns:
        Text with image references replaced.
    """
    for idx, ref in enumerate(image_refs):
        img_path = f"![{image_dir}/{page_index}_{idx}.jpg]"
        text = text.replace(ref, img_path + '\n')
    return text


def clean_output(text: str, other_refs: List[str]) -> str:
    """
    Clean model output by removing non-image references and fixing formatting.

    Args:
        text: Model output text.
        other_refs: List of non-image reference strings to remove.

    Returns:
        Cleaned text.
    """
    # Remove other references
    for ref in other_refs:
        text = text.replace(ref, '')

    # Fix LaTeX symbols
    text = text.replace('\\coloneqq', ':=')
    text = text.replace('\\eqqcolon', '=:')

    # Remove excessive newlines
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r'\n{3}', '\n\n', text)

    return text


def process_output(
    text: str,
    image: Image.Image,
    output_dir: str,
    page_index: int = 0,
    save_annotated: bool = True,
    extract_images: bool = True,
) -> Dict[str, Any]:
    """
    Complete post-processing of model output.

    Args:
        text: Model output text.
        image: Original image.
        output_dir: Directory for output files.
        page_index: Page index for naming.
        save_annotated: Whether to save annotated image.
        extract_images: Whether to extract image regions.

    Returns:
        Dict with:
            - markdown: Cleaned markdown text
            - raw_output: Original output
            - annotated_image_path: Path to annotated image
            - extracted_images: List of extracted image paths
    """
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Extract references
    refs, image_refs, other_refs = extract_refs(text)

    # Draw boxes and extract images
    annotated_image = None
    extracted_images = []

    if refs:
        annotated_image, extracted_images = draw_boxes(
            image,
            refs,
            output_dir=images_dir if extract_images else None,
            page_index=page_index
        )

    # Save annotated image
    annotated_path = None
    if save_annotated and annotated_image:
        annotated_path = os.path.join(output_dir, f"annotated_{page_index}.jpg")
        annotated_image.save(annotated_path)

    # Process text
    markdown = text
    markdown = replace_image_refs(markdown, image_refs, "images", page_index)
    markdown = clean_output(markdown, other_refs)

    return {
        "markdown": markdown,
        "raw_output": text,
        "annotated_image_path": annotated_path,
        "extracted_images": extracted_images,
    }


def process_geometry_output(text: str, output_dir: str) -> Optional[str]:
    """
    Process geometry output and generate visualization.

    Args:
        text: Model output containing geometry data.
        output_dir: Directory for output files.

    Returns:
        Path to generated geometry image or None.
    """
    if 'line_type' not in text:
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        data = eval(text)
        lines = data['Line']['line']
        line_type = data['Line']['line_type']
        endpoints = data['Line']['line_endpoint']

        fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        for idx, line in enumerate(lines):
            try:
                p0 = eval(line.split(' -- ')[0])
                p1 = eval(line.split(' -- ')[-1])
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                ax.scatter(p0[0], p0[1], s=5, color='k')
                ax.scatter(p1[0], p1[1], s=5, color='k')
            except Exception:
                pass

        for endpoint in endpoints:
            try:
                label = endpoint.split(': ')[0]
                x, y = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points',
                           fontsize=5, fontweight='light')
            except Exception:
                pass

        # Handle circles
        if 'Circle' in data:
            try:
                circle_centers = data['Circle']['circle_center']
                radius = data['Circle']['radius']
                for center, r in zip(circle_centers, radius):
                    center = eval(center.split(': ')[1])
                    circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                    ax.add_patch(circle)
            except Exception:
                pass

        geo_path = os.path.join(output_dir, 'geo.jpg')
        plt.savefig(geo_path)
        plt.close()
        return geo_path

    except Exception as e:
        logger.warning(f"Failed to process geometry output: {e}")
        return None
