from celery import Celery
from src.settings import CelerySettings


def get_celery() -> Celery:
    settings = CelerySettings()
    celery_app = Celery(
        main=settings.main,
        broker=settings.broker,
        backend=settings.backend,
        include=settings.include,
    )
    return celery_app
