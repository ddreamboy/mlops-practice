from pathlib import Path

import pandas as pd
from clearml import Dataset, OutputModel, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def load_data(path) -> tuple[list[str], list[str]]:
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
            "TFIDF_NGRAM_RANGE_MIN": 1,
            "TFIDF_NGRAM_RANGE_MAX": 2,
            "LR_C": 1.0,
            "LR_MAX_ITER": 1000,
            "RANDOM_STATE": 42,
            "DATASET_ID": "07ed666ce1b344708c952e83729bd0c6",
        }
    )

    task.execute_remotely(queue_name="students")

    dataset = Dataset.get(dataset_id=params["DATASET_ID"])
    dataset_path = Path(dataset.get_local_copy())

    x_train, y_train = load_data(dataset_path / "train.csv")
    x_test, y_test = load_data(dataset_path / "test.csv")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=params["TFIDF_MAX_FEATURES"],
                    ngram_range=(
                        params["TFIDF_NGRAM_RANGE_MIN"],
                        params["TFIDF_NGRAM_RANGE_MAX"],
                    ),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=params["LR_C"],
                    max_iter=params["LR_MAX_ITER"],
                    random_state=params["RANDOM_STATE"],
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    task_logger.report_scalar("metrics", "accuracy", value=accuracy, iteration=0)
    task_logger.report_scalar("metrics", "f1_macro", value=f1, iteration=0)

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    task_logger.report_matplotlib_figure("Confusion Matrix", "val", cm.figure_)

    OutputModel(task=task).update_weights(pipeline)


if __name__ == "__main__":
    main()
