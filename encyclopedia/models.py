from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class ArticleRecord:
    topic: str
    subtopic: str
    path: str
    summary: str
    created_at: str


@dataclass
class MemoryEntry:
    entry_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)
    articles: list[ArticleRecord] = field(default_factory=list)

    def add(self, entry_type: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.entries.append(MemoryEntry(entry_type=entry_type, content=content, metadata=metadata or {}))

    def add_article(self, article: ArticleRecord) -> None:
        self.articles.append(article)
        self.add(
            entry_type="article_created",
            content=article.subtopic,
            metadata={
                "topic": article.topic,
                "subtopic": article.subtopic,
                "path": article.path,
                "summary": article.summary,
                "created_at": article.created_at,
            },
        )

    def get_all(self) -> str:
        if not self.entries:
            return "Память пуста."
        lines = ["=== История работы ==="]
        for index, entry in enumerate(self.entries, start=1):
            lines.append(f"{index}. [{entry.entry_type}] {entry.content}")
            if entry.metadata:
                lines.append(f"   {json.dumps(entry.metadata, ensure_ascii=False)}")
        return "\n".join(lines)

    def get_related_articles(
        self,
        topic: str,
        exclude_subtopic: str | None = None,
        limit: int = 5,
    ) -> list[ArticleRecord]:
        same_topic = [a for a in self.articles if a.topic == topic and a.subtopic != exclude_subtopic]
        other_topics = [a for a in self.articles if a.topic != topic and a.subtopic != exclude_subtopic]
        return (same_topic + other_topics)[:limit]


@dataclass
class RunState:
    root_path: Path
    target_topics: int
    target_articles_per_topic: int
    created_topics: set[str] = field(default_factory=set)
    created_articles: set[str] = field(default_factory=set)

    @property
    def total_target_articles(self) -> int:
        return self.target_topics * self.target_articles_per_topic

    def is_complete(self) -> bool:
        return len(self.created_articles) >= self.total_target_articles


@dataclass
class ToolContext:
    memory: Memory
    state: RunState


class PlannedTopic(BaseModel):
    topic: str
    subtopics: list[str]


class EncyclopediaPlan(BaseModel):
    topics: list[PlannedTopic]


class GenerationStats(BaseModel):
    created_topics: int
    created_articles: int
    expected_topics: int
    expected_articles: int
    warnings: list[str] = Field(default_factory=list)
