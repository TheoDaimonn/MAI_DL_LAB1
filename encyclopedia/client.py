from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def create_client() -> OpenAI:
    load_dotenv()
    folder_id = os.environ.get("folder_id")
    api_key = os.environ.get("api_key")
    if not folder_id or not api_key:
        missing = [key for key, value in {"folder_id": folder_id, "api_key": api_key}.items() if not value]
        raise ValueError(f"Missing env vars: {', '.join(missing)}")
    return OpenAI(
        base_url="https://ai.api.cloud.yandex.net/v1",
        api_key=api_key,
        project=folder_id,
    )


def get_model_uri(folder_id: str | None = None) -> str:
    resolved_folder_id = folder_id or os.environ.get("folder_id")
    if not resolved_folder_id:
        raise ValueError("folder_id is required to build model URI")
    return f"gpt://{resolved_folder_id}/yandexgpt/latest"
