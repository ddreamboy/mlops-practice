from celery import shared_task

from src.schemas.requests import TextRequest
from src.services.model import SentimentClassifier


@shared_task(name="predict_emotion_task", pydantic=True, pydantic_model=TextRequest)
def predict_emotion_task(task: TextRequest) -> dict:
    return SentimentClassifier.predict_sentiment(task.text)
