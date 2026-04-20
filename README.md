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

## Работа с датасетом

Запуск загрузки

```bash
uv run python data/download_dataset.py
```

После запуска появятся файлы:

```
data/
    train.csv
    test.csv
```

Затем загружаем датасет в ClearML
```
uv run python src/upload_dataset
```
> В выводе будет dataset_id, его нужно сохранить в .env

Также можно прикрепить к полученному распределение по меткам
```
uv run python log_dataset_stats.py
```

## Обучение модели

Запуск агента
```
uv run clearml-agent daemon --queue course-queue --foreground
```

Запуск крипта обучения и оценки модели

> Нужно сменить `task.output_uri` и `DATASET_ID` в скрипте на актуальные
```
uv run python src/train.py
```

## Сервинг модели

Создаем serving-сущность
```
uv run clearml-serving create --name emotion-serving
```
> Копируем и сохраняем serve_id из консоли

```
uv run clearml-serving --id {serve_id} model auto-update --engine sklearn --endpoint emotion-classifier --published --project "emotion-classification" --name "emotion-classification-model" --max-versions 2 --preprocess src/preprocess.py
```

Через `docker compose` поднимаем
```yaml
services:
  clearml-serving-inference:                                                                                     
    image: allegroai/clearml-serving-inference                                                                      
    ports:                                                                                                       
      - "8009:8080"                                                                                              
    environment:                                                                                                 
      CLEARML_API_HOST: ${CLEARML_API_HOST}               
      CLEARML_WEB_HOST: ${CLEARML_WEB_HOST}                                                                      
      CLEARML_FILES_HOST: ${CLEARML_FILES_HOST}
      CLEARML_API_ACCESS_KEY: ${CLEARML_API_ACCESS_KEY}                                                          
      CLEARML_API_SECRET_KEY: ${CLEARML_API_SECRET_KEY}   
      CLEARML_SERVING_TASK_ID: ${CLEARML_SERVING_TASK_ID}
      CLEARML_EXTRA_PYTHON_PACKAGES: "scikit-learn==1.8.0"                                                  
```
> полное описание приведено сервисов ClearML в `docker-compose.clearml.yaml`

Пример запроса
```
curl -X 'POST' \
  'http://localhost:8009/serve/emotion-classifier/?version=2' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Это был лучший день!"}'
```

