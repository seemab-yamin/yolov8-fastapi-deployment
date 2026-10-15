import torch
from ultralytics import YOLO
from PIL import Image


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

    def predict(self, image: Image.Image) -> dict:
        """
        image: PIL Image object
        Returns: dict with detections
        """
        results = self.model(image)
        detections = []
        for r in results[0].boxes:
            detections.append(
                {
                    "class": self.model.names[int(r.cls)],
                    "confidence": round(float(r.conf), 3),
                    "bbox": r.xyxy[0].tolist(),
                }
            )
        return {"detections": detections}
