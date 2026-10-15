import torch
from PIL import Image
from ultralytics import YOLO

from app.schemas import BBox, Detection, DetectionResponse

class YOLOInferenceService:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.model.to(device)

    def predict(self, image: Image.Image) -> DetectionResponse:
        """
        image: PIL Image object
        Returns: DetectionResponse with typed detections, count, and image dimensions
        """

        results = self.model(image)
        boxes = results[0].boxes

        detections = [
            Detection(
                bbox=BBox(
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                ),
                confidence=round(float(box.conf[0]), 3),
                class_id=int(box.cls[0]),
                class_name=self.model.names[int(box.cls[0])],
            )
            for box in boxes
        ]

        return DetectionResponse(
            detections=detections,
            count=len(detections),
            image_width=image.width,
            image_height=image.height,
        )
