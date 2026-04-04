# Emotion Classification Service

## Dataset

**Датасет:** `MonoHime/ru_sentiment_dataset`
**Источник:** HuggingFace
**Классы:** `positive`, `negative`
**Train:** ~189 891 строк, **Test (validation):** ~21 098 строк
**Структура CSV:** две колонки - `text` (строка), `label` (строка)

Пример строки:

```
text,label
"Фильм оказался просто великолепным",positive
```

Запуск загрузки:

```bash
uv run python data/download_dataset.py
```

После запуска появятся файлы:

```
data/
    train.csv
    test.csv
```
