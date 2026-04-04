from pydantic import BaseModel


class TaskCreateResponse(BaseModel):
    """
    Ответ при создании асинхронной задачи.
    """

    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    """
    Ответ при запросе статуса задачи.
    """

    task_id: str
    status: str


class TaskResultResponse(BaseModel):
    """
    Ответ при запросе результата задачи.
    """

    task_id: str
    status: str
    result: dict | list | None = None
    message: str | None = None
