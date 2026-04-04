from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset
from loguru import logger

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
DATASET_NAME = "MonoHime/ru_sentiment_dataset"
LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}


def download_dataset() -> DatasetDict:
    """
    Загрузка датасета.

    Returns:
        Объект DatasetDict с train и test сплитами.
    """
    logger.info(f"Загрузка датасета {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    logger.info("Датасет успешно загружен")
    return dataset


def save_split(dataset: DatasetDict, split: str, path: Path) -> pd.DataFrame:
    """
    Сохранение сплита датасета в CSV-файл.

    Args:
        dataset: Объект DatasetDict.
        split (str): Название сплита (train или test).
        path (Path): Путь для сохранения CSV-файла.

    Returns:
        Сохраненный датафрейм.
    """
    df = dataset[split].to_pandas()
    df["label"] = df["sentiment"].map(LABEL_MAP)
    df = df[["text", "label"]]
    df.to_csv(path, index=False)
    logger.info(f"{split}: сохранено {len(df)} строк в {path}")
    return df


def log_distribution(df: pd.DataFrame, split: str) -> None:
    """
    Вывод распределения классов в датафрейме.

    Args:
        df (pd.DataFrame): Датафрейм с колонкой label.
        split (str): Название сплита для вывода в лог.
    """
    counts = df["label"].value_counts()
    for label, count in counts.items():
        logger.info(f"{split} - {label}: {count}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = download_dataset()
    train_df = save_split(dataset, "train", TRAIN_PATH)
    test_df = save_split(dataset, "validation", TEST_PATH)
    log_distribution(train_df, "train")
    log_distribution(test_df, "test")


if __name__ == "__main__":
    main()
