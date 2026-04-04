from datetime import datetime
from fastapi import APIRouter
from src.schemas.healthcheck import HealthcheckResult

router = APIRouter()


@router.get("/healthcheck/", response_model=HealthcheckResult)
async def healthcheck() -> HealthcheckResult:
    return HealthcheckResult(
        is_alive=True,
        date=datetime.now().isoformat(),
    )
