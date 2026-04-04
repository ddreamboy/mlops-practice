from __future__ import annotations

import time
from typing import Any

from src.services.loaders import IModelLoader


class SentimentClassifier:
    _pipeline: Any = None

    @classmethod
    def load(cls, loader: IModelLoader) -> None:
        cls._pipeline = loader.load()

    @classmethod
    def predict_sentiment(cls, text: str) -> dict:
        if cls._pipeline is None:
            from src.services.loaders import LocalModelLoader
            from src.settings import ModelSettings

            settings = ModelSettings()
            cls._pipeline = LocalModelLoader(settings.path).load()

        start_time = time.monotonic()

        proba = cls._pipeline.predict_proba([text])[0]
        classes = cls._pipeline.classes_
        predictions = sorted(
            [{"label": str(c), "score": float(s)} for c, s in zip(classes, proba)],
            key=lambda x: x["score"],
            reverse=True,
        )
        end_time = time.monotonic()
        elapsed_time = round(end_time - start_time, 4)
        return {"prediction": predictions, "elapsed_time": elapsed_time}
