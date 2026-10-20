import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas import DetectionResponse
from app.utils import (
    annotate_image,
    bytes_to_pil_image,
    read_upload_file,
    save_image,
    validate_image_dimensions,
)

router = APIRouter(tags=["Annotation"])


@router.post("/annotate")
async def annotate(
    file: UploadFile = File(...),
    detections_json: str = Form(...),
):

    try:
        payload = DetectionResponse.model_validate_json(detections_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid detections JSON: {e}")
    raw = await read_upload_file(file)
    file_name = file.filename
    image = bytes_to_pil_image(raw)
    validate_image_dimensions(image)

    image = annotate_image(image, payload)
    filename = f"{uuid.uuid4()}_{file_name}"
    save_image_path = save_image(image, settings.SAVE_DIR, filename)
    return {
        "result": "Image annotated successfully",
        "annotated_image_path": save_image_path,
    }
