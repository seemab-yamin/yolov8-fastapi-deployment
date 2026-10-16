from datetime import datetime

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/")
def health_check(request: Request):
    """
    Health check endpoint to confirm the API is running.
    Returns basic status and current server timestamp.
    """
    return {
        "status": "running",
        "model_loaded": request.app.state.inference_service.model is not None,
        "device": request.app.state.inference_service.device,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
