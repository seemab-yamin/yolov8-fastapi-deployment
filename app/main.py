from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers import health, predict
from app.services import YOLOInferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.inference_service = YOLOInferenceService(settings.model_path)
    app.state.ready = True
    yield
    app.state.ready = False
    app.state.inference_service = None


app = FastAPI(title="Object Detection API", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
