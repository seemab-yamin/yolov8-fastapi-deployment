from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["Health"])


@router.get("/ping")
def health_check(request: Request):
    """
    Health check endpoint to confirm the API is running.
    Returns basic status and current server timestamp.
    """
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "status": "running",
        "model_loaded": request.app.state.inference_service.model is not None,
        "device": request.app.state.inference_service.device,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
