from __future__ import annotations

import json
import re
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from .client import get_model_uri
from .constants import TOPIC_CANDIDATES
from .models import ArticleRecord, EncyclopediaPlan, GenerationStats, Memory, PlannedTopic, RunState, ToolContext
from .tools import Grep, Mkdir, WriteArticle
from .utils import extract_json, normalize_text, response_text, safe_resolve_under_root, sanitize_name


class EncyclopediaAgent:
    def __init__(
        self,
        client: OpenAI,
        root_path: str,
        topics_count: int,
        articles_per_topic: int,
        model: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.client = client
        self.model = model or get_model_uri()
        self.root_path = Path(root_path).resolve(strict=False)
        self.memory = Memory()
        self.state = RunState(
            root_path=self.root_path,
            target_topics=topics_count,
            target_articles_per_topic=articles_per_topic,
        )
        self.context = ToolContext(memory=self.memory, state=self.state)
        self.tools = {"mkdir": Mkdir, "grep": Grep, "write_article": WriteArticle}
        self.verbose = verbose

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def ask_model(self, instructions: str, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, instructions=instructions, input=prompt)
        return response_text(response)

    def build_fallback_plan(self) -> EncyclopediaPlan:
        selected_topics = list(TOPIC_CANDIDATES.items())[: self.state.target_topics]
        topics: list[PlannedTopic] = []
        for topic, examples in selected_topics:
            topics.append(
                PlannedTopic(
                    topic=topic,
                    subtopics=self._fallback_subtopics(topic, examples, self.state.target_articles_per_topic),
                )
            )
        return EncyclopediaPlan(topics=topics)

    @staticmethod
    def _fallback_subtopics(topic: str, examples: list[str], required_count: int) -> list[str]:
        subtopics = list(examples[:required_count])
        index = 1
        while len(subtopics) < required_count:
            candidate = f"{topic}: интересный факт {index}"
            if candidate not in subtopics:
                subtopics.append(candidate)
            index += 1
        return subtopics

    def plan_structure(self) -> EncyclopediaPlan:
        available_topics = {topic: subtopics[:] for topic, subtopics in TOPIC_CANDIDATES.items()}
        instructions = (
            "Ты проектируешь структуру детской энциклопедии. "
            "Верни только JSON без пояснений и markdown. "
            "Строго соблюдай количество тем и подтем."
        )
        prompt = f"""
Создай план детской энциклопедии.

Нужно выбрать ровно {self.state.target_topics} тем.
Для каждой темы нужно предложить ровно {self.state.target_articles_per_topic} разных статей.

Доступные темы и примеры подтем:
{json.dumps(available_topics, ensure_ascii=False, indent=2)}

Формат ответа строго такой:
{{
  "topics": [
    {{
      "topic": "Название темы",
      "subtopics": ["Статья 1", "Статья 2"]
    }}
  ]
}}

Правила:
- используй только темы из списка
- не дублируй темы
- не дублируй названия статей внутри одной темы
- названия статей должны быть понятными детям 8-12 лет
- верни только JSON
""".strip()

        for _ in range(3):
            raw = self.ask_model(instructions, prompt)
            try:
                data = extract_json(raw)
                plan = EncyclopediaPlan.model_validate(data)
                self._validate_plan(plan)
                return plan
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                prompt = (
                    f"Предыдущий ответ не прошёл проверку: {error}. "
                    f"Верни JSON строго нужной структуры.\n\n{prompt}"
                )
        self.log("Не удалось получить корректный план от модели, используется резервный план.")
        return self.build_fallback_plan()

    def _validate_plan(self, plan: EncyclopediaPlan) -> None:
        if len(plan.topics) != self.state.target_topics:
            raise ValueError("Неверное количество тем")

        seen_topics: set[str] = set()
        for topic in plan.topics:
            if topic.topic not in TOPIC_CANDIDATES:
                raise ValueError(f"Недопустимая тема: {topic.topic}")
            if topic.topic in seen_topics:
                raise ValueError(f"Повтор темы: {topic.topic}")
            seen_topics.add(topic.topic)
            if len(topic.subtopics) != self.state.target_articles_per_topic:
                raise ValueError(f"Неверное количество статей для темы {topic.topic}")
            normalized = [normalize_text(item).lower() for item in topic.subtopics]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"Повторяющиеся статьи в теме {topic.topic}")

    def ensure_directory(self, topic: str) -> Path:
        topic_dir = self.root_path / sanitize_name(topic)
        result = Mkdir(path=str(topic_dir)).process(self.context)
        if result.get("status") != "ok":
            raise RuntimeError(f"Не удалось создать директорию для темы {topic}")
        return topic_dir

    def article_path(self, topic: str, subtopic: str) -> Path:
        topic_dir = self.root_path / sanitize_name(topic)
        filename = sanitize_name(subtopic) + ".md"
        return safe_resolve_under_root(self.root_path, str(Path(topic_dir.name) / filename))

    def generate_article_content(self, topic: str, subtopic: str) -> str:
        related_articles = self.memory.get_related_articles(topic=topic, exclude_subtopic=subtopic, limit=5)
        related_block = "\n".join(
            f"- {item.subtopic} ({item.topic}): {item.summary}" for item in related_articles
        ) or "- Пока нет связанных статей"

        instructions = (
            "Ты пишешь статьи для детской энциклопедии. "
            "Пиши понятно, доброжелательно и точно. "
            "Не используй слишком сложные термины без объяснения."
        )
        prompt = f"""
Напиши markdown-контент для статьи детской энциклопедии.

Раздел: {topic}
Статья: {subtopic}

Требования:
- аудитория: дети 8-12 лет
- объём: 350-600 слов
- структура:
  1. короткое вступление
  2. 2-4 подзаголовка второго уровня
  3. простой интересный факт в конце
- стиль: объясняющий, но живой
- не добавляй главный заголовок первого уровня, его создаст программа
- можно ссылаться на уже созданные статьи только если это уместно
- не выдумывай факты, пиши общие проверяемые сведения

Уже созданные статьи:
{related_block}

Верни только markdown-контент статьи.
""".strip()

        try:
            return self.ask_model(instructions, prompt).strip()
        except Exception as error:
            self.log(f"Не удалось сгенерировать статью '{subtopic}' через модель: {error}")
            return self.fallback_article_content(topic, subtopic, related_articles)

    @staticmethod
    def fallback_article_content(topic: str, subtopic: str, related_articles: list[ArticleRecord]) -> str:
        related_sentence = ""
        if related_articles:
            first_related = related_articles[0]
            related_sentence = (
                f"\n\nЭта тема связана со статьёй «{first_related.subtopic}», "
                f"которая тоже находится в разделе «{first_related.topic}»."
            )
        return (
            f"{subtopic} — это важная часть темы «{topic}».\n\n"
            f"## Что это такое\n\n"
            f"Если объяснить просто, {subtopic.lower()} помогает лучше понять окружающий мир. "
            f"Детям интересно изучать такие темы, потому что они показывают, как устроена природа, техника или история.\n\n"
            f"## Почему это важно\n\n"
            f"Когда мы узнаём о {subtopic.lower()}, мы замечаем больше интересных деталей вокруг себя. "
            f"Такое знание развивает любопытство и помогает задавать хорошие вопросы.\n\n"
            f"## Интересный факт\n\n"
            f"Многие большие открытия начинались именно с простого детского вопроса: «Почему это так?»"
            f"{related_sentence}"
        )

    @staticmethod
    def build_summary(content: str) -> str:
        plain = re.sub(r"[#*_`>-]", "", content)
        plain = normalize_text(plain)
        if len(plain) <= 180:
            return plain
        cut = plain[:177].rstrip(" ,.;:")
        return cut + "..."

    def write_article(self, topic: str, subtopic: str, content: str) -> dict[str, str]:
        result = WriteArticle(
            topic=topic,
            subtopic=subtopic,
            content=content,
            filepath=str(self.article_path(topic, subtopic)),
            summary=self.build_summary(content),
        ).process(self.context)
        if result.get("status") != "ok":
            raise RuntimeError(f"Не удалось сохранить статью {topic}/{subtopic}")
        return result

    def run(self) -> GenerationStats:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.memory.add(
            "run_started",
            "Запущено создание детской энциклопедии",
            {
                "root_path": str(self.root_path),
                "topics": self.state.target_topics,
                "articles_per_topic": self.state.target_articles_per_topic,
            },
        )

        plan = self.plan_structure()
        self.memory.add("plan_created", "Сформирован план энциклопедии", plan.model_dump())

        self.log("\n=== ПЛАН ===")
        for topic in plan.topics:
            self.log(f"- {topic.topic}: {', '.join(topic.subtopics)}")

        for planned_topic in plan.topics:
            self.ensure_directory(planned_topic.topic)
            for subtopic in planned_topic.subtopics:
                self.log(f"Создаю статью: {planned_topic.topic} / {subtopic}")
                content = self.generate_article_content(planned_topic.topic, subtopic)
                self.write_article(planned_topic.topic, subtopic, content)

        stats = self.validate_output(plan)
        self.memory.add("run_finished", "Создание энциклопедии завершено", stats.model_dump())
        return stats

    def validate_output(self, plan: EncyclopediaPlan) -> GenerationStats:
        warnings: list[str] = []
        expected_topics = len(plan.topics)
        expected_articles = sum(len(topic.subtopics) for topic in plan.topics)

        actual_topic_dirs = [path for path in self.root_path.iterdir() if path.is_dir()]
        actual_article_files = list(self.root_path.rglob("*.md"))

        if len(actual_topic_dirs) != expected_topics:
            warnings.append(f"Ожидалось {expected_topics} директорий, найдено {len(actual_topic_dirs)}")
        if len(actual_article_files) != expected_articles:
            warnings.append(f"Ожидалось {expected_articles} статей, найдено {len(actual_article_files)}")

        planned_paths = {
            str(self.article_path(topic.topic, subtopic))
            for topic in plan.topics
            for subtopic in topic.subtopics
        }
        existing_paths = {str(path.resolve(strict=False)) for path in actual_article_files}
        missing_paths = sorted(planned_paths - existing_paths)
        if missing_paths:
            warnings.append("Не найдены файлы: " + "; ".join(missing_paths[:5]))

        duplicate_titles = self._find_duplicate_titles(actual_article_files)
        if duplicate_titles:
            warnings.append("Повторяющиеся заголовки: " + "; ".join(duplicate_titles))

        return GenerationStats(
            created_topics=len(actual_topic_dirs),
            created_articles=len(actual_article_files),
            expected_topics=expected_topics,
            expected_articles=expected_articles,
            warnings=warnings,
        )

    @staticmethod
    def _find_duplicate_titles(files: list[Path]) -> list[str]:
        titles: dict[str, int] = {}
        duplicates: list[str] = []
        for file_path in files:
            try:
                first_line = file_path.read_text(encoding="utf-8").splitlines()[0].strip()
            except Exception:
                continue
            titles[first_line] = titles.get(first_line, 0) + 1
        for title, count in titles.items():
            if count > 1:
                duplicates.append(f"{title} ({count})")
        return duplicates
