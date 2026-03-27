from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from .constants import ALLOWED_FILE_EXTENSIONS
from .models import ArticleRecord, ToolContext
from .utils import normalize_text, safe_resolve_under_root


class ToolBase(BaseModel):
    def process(self, context: ToolContext) -> dict[str, Any]:
        raise NotImplementedError


class Mkdir(ToolBase):
    """Создание директории внутри корневой папки энциклопедии."""

    path: str = Field(description="Относительный или абсолютный путь для создания директории")

    def process(self, context: ToolContext) -> dict[str, Any]:
        clean_path = Path(self.path)
        final_path = clean_path if clean_path.is_absolute() else safe_resolve_under_root(context.state.root_path, self.path)
        final_path.mkdir(parents=True, exist_ok=True)
        context.state.created_topics.add(final_path.name)
        context.memory.add("directory_created", f"Создана директория {final_path}", {"path": str(final_path)})
        return {"status": "ok", "path": str(final_path)}


class Grep(ToolBase):
    """Поиск текста по одному файлу или по всем markdown-файлам в директории."""

    pattern: str = Field(description="Строка для поиска")
    path: str | None = Field(default=None, description="Файл или директория для поиска; если не указано, поиск идёт по всей энциклопедии")

    def process(self, context: ToolContext) -> dict[str, Any]:
        base_path = context.state.root_path if not self.path else Path(self.path)
        target = base_path if base_path.is_absolute() else safe_resolve_under_root(context.state.root_path, str(base_path))
        if not target.exists():
            return {"status": "not_found", "path": str(target), "matches": []}

        files: Iterable[Path]
        files = target.rglob("*.md") if target.is_dir() else [target]

        matches: list[dict[str, Any]] = []
        needle = self.pattern.lower()
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if needle in line.lower():
                    matches.append({"file": str(file_path), "line": line_number, "text": line.strip()})
        if matches:
            context.memory.add(
                "search_found",
                f"Найдено совпадений: {len(matches)}",
                {"pattern": self.pattern, "path": str(target)},
            )
        return {"status": "ok", "count": len(matches), "matches": matches[:20]}


class WriteArticle(ToolBase):
    """Создание и сохранение markdown-статьи."""

    topic: str = Field(description="Название раздела")
    subtopic: str = Field(description="Название статьи")
    content: str = Field(description="Основной markdown-контент статьи")
    filepath: str = Field(description="Относительный или абсолютный путь к markdown-файлу")
    summary: str = Field(description="Краткое описание статьи для памяти", default="")

    def process(self, context: ToolContext) -> dict[str, Any]:
        relative_or_absolute = Path(self.filepath)
        final_path = (
            relative_or_absolute
            if relative_or_absolute.is_absolute()
            else safe_resolve_under_root(context.state.root_path, self.filepath)
        )
        if final_path.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
            final_path = final_path.with_suffix(".md")

        final_path.parent.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = self.content.strip()
        summary = normalize_text(self.summary or body.split(". ")[0])[:180]

        markdown = (
            f"# {self.subtopic}\n\n"
            f"**Раздел:** {self.topic}\n\n"
            f"---\n\n"
            f"{body}\n\n"
            f"---\n\n"
            f"*Энциклопедия для детей*\n\n"
            f"*Дата создания: {created_at}*\n"
        )

        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        temp_path.write_text(markdown, encoding="utf-8")
        temp_path.replace(final_path)

        article = ArticleRecord(
            topic=self.topic,
            subtopic=self.subtopic,
            path=str(final_path),
            summary=summary,
            created_at=created_at,
        )
        context.memory.add_article(article)
        context.state.created_articles.add(f"{self.topic}/{self.subtopic}")

        return {"status": "ok", "path": str(final_path), "topic": self.topic, "subtopic": self.subtopic}
