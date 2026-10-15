from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class Detection(BaseModel):
    bbox: BBox
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    class_id: int = Field(..., description="Numeric class index")
    class_name: str = Field(..., description="Human-readable class label")


class DetectionResponse(BaseModel):
    detections: list[Detection]
    count: int = Field(..., description="Total number of detections")
    image_width: int
    image_height: int
