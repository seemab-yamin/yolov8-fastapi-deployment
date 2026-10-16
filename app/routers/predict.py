from fastapi import APIRouter, File, Request, UploadFile

from app.utils import (
    bytes_to_pil_image,
    read_upload_file,
    validate_image_dimensions,
)

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    raw = await read_upload_file(file)
    image = bytes_to_pil_image(raw)
    validate_image_dimensions(image)

    result = request.app.state.inference_service.predict(image)
    return result
