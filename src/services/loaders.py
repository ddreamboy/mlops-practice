from abc import ABC, abstractmethod
from typing import Any


class IModelLoader(ABC):
    @abstractmethod
    def load(self) -> Any: ...


class LocalModelLoader(IModelLoader):
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> Any:
        import joblib

        return joblib.load(self.path)
