import os
from pathlib import Path

from clearml import Dataset
from loguru import logger

from settings import ClearMLSettings

settings = ClearMLSettings()

os.environ.setdefault("CLEARML_WEB_HOST", settings.CLEARML_WEB_HOST)
os.environ.setdefault("CLEARML_API_HOST", settings.CLEARML_API_HOST)
os.environ.setdefault("CLEARML_FILES_HOST", settings.CLEARML_FILES_HOST)
os.environ.setdefault("CLEARML_API_ACCESS_KEY", settings.CLEARML_API_ACCESS_KEY)
os.environ.setdefault("CLEARML_API_SECRET_KEY", settings.CLEARML_API_SECRET_KEY)


def upload_dataset() -> str:
    root_dir = Path(__file__).resolve().parent.parent
    train_path = root_dir / "data" / "train.csv"
    test_path = root_dir / "data" / "test.csv"

    if not train_path.exists() or not test_path.exists():
        missing = [str(path) for path in (train_path, test_path) if not path.exists()]
        raise FileNotFoundError(
            f"Нету локальных файлов для загрузки: {', '.join(missing)}"
        )

    dataset = Dataset.create(
        dataset_project="emotion-classification",
        dataset_name="ru_sentiment_dataset",
        dataset_version="1.0",
    )
    dataset.add_files(path=str(train_path))
    dataset.add_files(path=str(test_path))
    dataset.upload()
    dataset.finalize()

    return dataset.id


def delete_dataset(dataset_id: str | None) -> None:
    logger.info(f"Удаление датасета: {dataset_id=}")
    Dataset.delete(dataset_id=dataset_id, force=True)
    logger.info(f"Датасет успешно удален: {dataset_id=}")


def get_dataset(dataset_id: str | None) -> Dataset:
    logger.info(f"Получение датасета: {dataset_id=}")
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Датасет успешно получен: {dataset_id=}")
    return dataset


if __name__ == "__main__":
    dataset_id = upload_dataset()
    logger.info(f"Датасет загружен: {dataset_id=}")

    dataset_id = settings.dataset_id
    dataset = get_dataset(dataset_id)
    logger.info(f"Датасет доступен по пути: {dataset.get_local_copy()}")
