from pathlib import Path

import pandas as pd
from clearml import Dataset
from loguru import logger

from settings import ClearMLSettings

settings = ClearMLSettings()


def get_dataset(dataset_id: str | None) -> Dataset:
    logger.info(f"Получение датасета: {dataset_id=}")
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Датасет успешно получен: {dataset_id=}")
    return dataset


def main():
    dataset_id = settings.dataset_id
    dataset = get_dataset(dataset_id)
    logger.info(f"Датасет доступен по пути: {dataset.get_local_copy()}")

    local_path = Path(dataset.get_local_copy())
    train_path = local_path / "train.csv"
    test_path = local_path / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_counts = train_df["label"].value_counts().reset_index()
    train_counts.columns = ["label", "count"]

    test_counts = test_df["label"].value_counts().reset_index()
    test_counts.columns = ["label", "count"]

    dataset_logger = dataset.get_logger()

    dataset_logger.report_table(
        title="Train label distribution",
        series="labels",
        iteration=0,
        table_plot=train_counts,
    )
    dataset_logger.report_table(
        title="Test label distribution",
        series="labels",
        iteration=0,
        table_plot=test_counts,
    )

    dataset_logger.report_histogram(
        title="Label distribution",
        series="train",
        iteration=0,
        values=train_df["label"].value_counts().values,
        xlabels=train_df["label"].value_counts().index.tolist(),
    )

    dataset_logger.report_histogram(
        title="Label distribution",
        series="test",
        iteration=0,
        values=test_df["label"].value_counts().values,
        xlabels=test_df["label"].value_counts().index.tolist(),
    )


if __name__ == "__main__":
    main()
