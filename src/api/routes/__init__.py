from fastapi import APIRouter

from src.api.routes.healthcheck import router as healthcheck_router
from src.api.routes.predict import router as predict_router

router = APIRouter()
router.include_router(healthcheck_router)
router.include_router(predict_router)
