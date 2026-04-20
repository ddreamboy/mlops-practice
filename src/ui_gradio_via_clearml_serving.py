import time

import gradio as gr
import httpx

from settings import GradioSettings

settings = GradioSettings()
CLEARML_SERVING_URL = settings.clearml_serving_url
CLEARML_SERVING_VERSION = settings.clearml_serving_version


async def predict_clearml(text: str) -> str:
    url = f"{CLEARML_SERVING_URL}/?version={CLEARML_SERVING_VERSION}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            t0 = time.monotonic()
            response = await client.post(
                url,
                json={"text": text},
                headers={"accept": "application/json"},
            )
            latency = round(time.monotonic() - t0, 4)
            response.raise_for_status()
            data = response.json()
            label = data.get("label", "—")
            yield f"Метка: {label}\nLatency: {latency} сек"
    except httpx.ConnectError:
        yield f"Ошибка: endpoint недоступен ({url})"
    except httpx.HTTPStatusError as e:
        yield f"Ошибка HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        yield f"Ошибка: {e}"


with gr.Blocks() as demo:
    text = gr.Textbox(
        label="Введите текст",
        lines=5,
        max_lines=10,
        placeholder="Введите несколько строк...",
    )
    out = gr.Textbox(label="Результат")

    btn = gr.Button("Отправить")
    btn.click(predict_clearml, inputs=text, outputs=out)

demo.launch(server_name=settings.host, server_port=settings.port)
