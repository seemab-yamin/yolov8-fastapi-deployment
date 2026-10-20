from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import settings
from app.routers import annotate, download, health, predict
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
app.include_router(annotate.router)
app.include_router(download.router)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")
