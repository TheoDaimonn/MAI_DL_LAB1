from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .constants import INVALID_FILENAME_CHARS


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def sanitize_name(value: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub(" ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned or "Без названия"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) not in {"output_text", "text"}:
                continue
            value = getattr(content, "text", "")
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(c for c in chunks if c).strip()


def safe_resolve_under_root(root_path: Path, relative_path: str) -> Path:
    root = root_path.resolve(strict=False)
    resolved = (root / relative_path).resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError("Path traversal is not allowed")
    return resolved
