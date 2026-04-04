import asyncio
import json

import gradio as gr
import httpx

from settings import GradioSettings

settings = GradioSettings()
API_BASE_URL = settings.api_base_url
POLL_INTERVAL_SECONDS = settings.poll_interval_seconds
POLL_TIMEOUT_SECONDS = settings.poll_timeout_seconds


async def submit_and_poll(text: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        create_response = await client.post(
            f"{API_BASE_URL}/predict/",
            json={"text": text},
            headers={"accept": "application/json"},
        )
        create_response.raise_for_status()
        created = create_response.json()

        task_id = created.get("task_id")
        if not task_id:
            yield "Не удалось получить task_id"
            return

        yield f"Задача создана: {task_id}\nСтатус: {created.get('status', 'PENDING')}"

        elapsed = 0.0
        while elapsed <= POLL_TIMEOUT_SECONDS:
            status_response = await client.get(
                f"{API_BASE_URL}/predict/{task_id}/status",
                headers={"accept": "application/json"},
            )
            status_response.raise_for_status()
            status_payload = status_response.json()
            status = status_payload.get("status", "UNKNOWN")

            if status == "SUCCESS":
                result_response = await client.get(
                    f"{API_BASE_URL}/predict/{task_id}/result",
                    headers={"accept": "application/json"},
                )
                result_response.raise_for_status()
                result_payload = result_response.json()
                formatted = json.dumps(
                    result_payload.get("result"), ensure_ascii=False, indent=2
                )
                yield f"Статус: {status}\n\nРезультат:\n{formatted}"
                return

            if status in {"FAILURE"}:
                yield f"Задача завершилась с ошибкой. Статус: {status}"
                return

            yield f"Ожидание... task_id={task_id}\nТекущий статус: {status}"
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

        yield f"Таймаут ожидания результата ({POLL_TIMEOUT_SECONDS:.0f} сек). task_id={task_id}"


with gr.Blocks() as demo:
    text = gr.Textbox(
        label="Введите текст",
        lines=5,
        max_lines=10,
        placeholder="Введите несколько строк...",
    )
    out = gr.Textbox(label="Результат")

    btn = gr.Button("Отправить")
    btn.click(submit_and_poll, inputs=text, outputs=out)

demo.launch(server_name=settings.host, server_port=settings.port)
