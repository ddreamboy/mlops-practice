from pathlib import Path

import pandas as pd
from clearml import Dataset, OutputModel, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from settings import ClearMLSettings

settings = ClearMLSettings()


def load_data(path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()


def main():
    task: Task = Task.init(
        project_name="emotion-classification", task_name="train_model"
    )
    task_logger = task.get_logger()

    params = task.connect(
        {
            "TFIDF_MAX_FEATURES": 50_000,
            "TFIDF_NGRAM_RANGE": (1, 2),
            "LR_C": 1.0,
            "LR_MAX_ITER": 1000,
            "RANDOM_STATE": 42,
        }
    )

    task.execute_remotely(queue_name=settings.queue_name)

    dataset = Dataset.get(dataset_id=settings.dataset_id)
    dataset_path = Path(dataset.get_local_copy())

    train_path = dataset_path / "train.csv"
    test_path = dataset_path / "test.csv"

    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)

    train_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=params["TFIDF_MAX_FEATURES"],
                    ngram_range=params["TFIDF_NGRAM_RANGE"],
                ),
            ),
            (
                "clf",
                LogisticRegression(C=params["LR_C"], max_iter=params["LR_MAX_ITER"]),
            ),
        ]
    )

    task_logger.report_text("Начинаем обучение модели")
    train_pipeline.fit(x_train, y_train)
    task_logger.report_text("Модель обучена")

    y_pred = train_pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    task_logger.report_text(f"Accuracy: {accuracy:.4f}")
    task_logger.report_text(f"F1 (macro): {f1:.4f}")

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    task_logger.report_matplotlib_figure("Confusion Matrix", cm.figure_)

    OutputModel(task=task).update_weights(train_pipeline)
    task_logger.report_text("Модель сохранена в ClearML")

    task.execute_remotely(queue_name=settings.queue_name)


if __name__ == "__main__":
    main()
