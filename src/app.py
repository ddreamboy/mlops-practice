from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.celery_app  # noqa: F401
from src.api.routes import router

app = FastAPI(title="Emotion Classification Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
