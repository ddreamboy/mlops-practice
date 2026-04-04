from pydantic import BaseModel, ConfigDict


class TextRequest(BaseModel):
    """
    Запрос на получение анализа текста.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "Сегодня отличный день, я очень рад!"}}
    )

    text: str
