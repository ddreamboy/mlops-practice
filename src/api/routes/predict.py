from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.schemas.requests import TextRequest
from src.schemas.tasks import TaskCreateResponse, TaskResultResponse, TaskStatusResponse
from src.tasks import predict_emotion_task

router = APIRouter()


@router.post("/predict/", response_model=TaskCreateResponse)
async def predict(request: TextRequest) -> TaskCreateResponse:
    """
    Постановка задачи классификации текста в очередь Celery.

    Args:
        request (TextRequest): Запрос с текстом для анализа.

    Returns:
        TaskCreateResponse: Идентификатор задачи и ее начальный статус.
    """
    try:
        task = await run_in_threadpool(
            predict_emotion_task.apply_async,
            kwargs={"task": request.model_dump()},
            queue="predict",
        )
        return TaskCreateResponse(task_id=task.id, status="PENDING")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/predict/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Получение статуса задачи по ее идентификатору.

    Args:
        task_id (str): Идентификатор задачи Celery.

    Returns:
        TaskStatusResponse: Идентификатор задачи и ее текущий статус.
    """
    result = AsyncResult(task_id)
    return TaskStatusResponse(task_id=task_id, status=result.status)


@router.get("/predict/{task_id}/result", response_model=TaskResultResponse)
async def get_task_result(task_id: str) -> TaskResultResponse:
    """
    Получение результата задачи по ее идентификатору.

    Args:
        task_id (str): Идентификатор задачи Celery.

    Returns:
        TaskResultResponse: Результат задачи или сообщение об ошибке.

    Raises:
        HTTPException: 202 если задача еще не готова, 500 если задача завершилась с ошибкой.
    """
    result = AsyncResult(task_id)
    if not result.ready():
        raise HTTPException(status_code=202, detail="Задача еще не завершена")
    if result.failed():
        raise HTTPException(status_code=500, detail=str(result.result))
    return TaskResultResponse(
        task_id=task_id,
        status=result.status,
        result=result.result,
    )
