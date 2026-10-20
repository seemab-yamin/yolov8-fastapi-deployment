import io
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.schemas import DetectionResponse

# Allowed image formats for the detection endpoint
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

# Max upload size (10 MB) — adjust as needed
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


async def read_upload_file(file: UploadFile) -> bytes:
    """
    Read an uploaded file into bytes and enforce size limit.
    Raises HTTPException on failure.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    return contents


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """
    Convert raw bytes to a PIL Image in RGB mode.
    Raises HTTPException if the bytes are not a valid image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # checks integrity without decoding fully
        # Re-open after verify() because verify() leaves the file unusable
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(e)}",
        )


def pil_image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """
    Serialize a PIL Image back to bytes (useful for saving annotated output).
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def validate_image_dimensions(
    image: Image.Image,
    min_side: int = 32,
    max_side: int = 8192,
) -> None:
    """
    Guard against absurdly small or large images before inference.
    """
    width, height = image.size
    if min(width, height) < min_side:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({width}x{height}). Min side: {min_side}px.",
        )
    if max(width, height) > max_side:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({width}x{height}). Max side: {max_side}px.",
        )


def annotate_image(image, detections: DetectionResponse):
    """
    Annotate the image with bounding boxes and class names based on detections.
    """
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections.detections:
        bbox = det.bbox
        # Draw rectangle
        draw.rectangle(
            [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            outline="red",
            width=2,
        )
        # Draw label
        label = f"{det.class_name} ({det.confidence:.2f})"
        draw.text((bbox.x1, bbox.y1 - 10), label, fill="red", font=font)

    return image


def save_image(image: Image.Image, dir_path: Path, file_name: str):
    try:
        image_path = dir_path / file_name
        image.save(image_path)
        return image_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save image: {str(e)}",
        )
