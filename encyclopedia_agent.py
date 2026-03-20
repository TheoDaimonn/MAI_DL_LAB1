import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


TOPIC_CANDIDATES: Dict[str, List[str]] = {
    "Космос": ["Планеты", "Звёзды", "Кометы", "Астероиды", "Космонавты", "Телескопы"],
    "Животные": ["Млекопитающие", "Птицы", "Насекомые", "Морские животные", "Хищники", "Детёныши животных"],
    "Наука": ["Физика", "Химия", "Биология", "Изобретения", "Электричество", "Микромир"],
    "Природа": ["Леса", "Реки", "Горы", "Вулканы", "Погода", "Времена года"],
    "История": ["Древние цивилизации", "Рыцари", "Пирамиды", "Открытия", "Первые города", "Книгопечатание"],
    "Техника": ["Роботы", "Самолёты", "Поезда", "Компьютеры", "Мосты", "Подводные лодки"],
    "Океан": ["Кораллы", "Киты", "Осьминоги", "Глубины океана", "Течения", "Острова"],
}

ALLOWED_FILE_EXTENSIONS = {".md"}
INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]+')


def create_client() -> OpenAI:
    load_dotenv()
    folder_id = os.environ.get("folder_id")
    api_key = os.environ.get("api_key")
    if not folder_id or not api_key:
        missing = [k for k, v in {"folder_id": folder_id, "api_key": api_key}.items() if not v]
        raise ValueError(f"Missing env vars: {', '.join(missing)}")
    return OpenAI(
        base_url="https://ai.api.cloud.yandex.net/v1",
        api_key=api_key,
        project=folder_id,
    )


def get_model_uri(folder_id: Optional[str] = None) -> str:
    folder_id = folder_id or os.environ.get("folder_id")
    if not folder_id:
        raise ValueError("folder_id is required to build model URI")
    return f"gpt://{folder_id}/yandexgpt/latest"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def sanitize_name(value: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub(" ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned or "Без названия"


def extract_json(text: str) -> Dict[str, Any]:
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

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) in {"output_text", "text"}:
                    value = getattr(content, "text", "")
                    if isinstance(value, str):
                        chunks.append(value)
    return "\n".join(c for c in chunks if c).strip()


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Memory:
    entries: List[MemoryEntry] = field(default_factory=list)
    articles: List[ArticleRecord] = field(default_factory=list)

    def add(self, entry_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
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

    def get_articles_list(self, exclude_subtopic: Optional[str] = None) -> str:
        if not self.articles:
            return "Пока нет готовых статей."

        lines = ["Уже созданные статьи:"]
        for article in self.articles:
            if exclude_subtopic and article.subtopic == exclude_subtopic:
                continue
            lines.append(
                f"- {article.subtopic} (раздел: {article.topic}) — {article.summary}"
            )
        return "\n".join(lines)

    def get_related_articles(self, topic: str, exclude_subtopic: Optional[str] = None, limit: int = 5) -> List[ArticleRecord]:
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


def safe_resolve_under_root(root_path: Path, relative_path: str) -> Path:
    root = root_path.resolve(strict=False)
    resolved = (root / relative_path).resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError("Path traversal is not allowed")
    return resolved


class ToolBase(BaseModel):
    def process(self, context: ToolContext) -> Dict[str, Any]:
        raise NotImplementedError


class Mkdir(ToolBase):
    """Создание директории внутри корневой папки энциклопедии."""

    path: str = Field(description="Относительный или абсолютный путь для создания директории")

    def process(self, context: ToolContext) -> Dict[str, Any]:
        clean_path = Path(self.path)
        final_path = clean_path if clean_path.is_absolute() else safe_resolve_under_root(context.state.root_path, self.path)
        final_path.mkdir(parents=True, exist_ok=True)
        context.state.created_topics.add(final_path.name)
        context.memory.add("directory_created", f"Создана директория {final_path}", {"path": str(final_path)})
        return {"status": "ok", "path": str(final_path)}


class Grep(ToolBase):
    """Поиск текста по одному файлу или по всем markdown-файлам в директории."""

    pattern: str = Field(description="Строка для поиска")
    path: Optional[str] = Field(default=None, description="Файл или директория для поиска; если не указано, поиск идёт по всей энциклопедии")

    def process(self, context: ToolContext) -> Dict[str, Any]:
        base_path = context.state.root_path if not self.path else Path(self.path)
        target = base_path if base_path.is_absolute() else safe_resolve_under_root(context.state.root_path, str(base_path))
        if not target.exists():
            return {"status": "not_found", "path": str(target), "matches": []}

        files: Iterable[Path]
        if target.is_dir():
            files = target.rglob("*.md")
        else:
            files = [target]

        matches: List[Dict[str, Any]] = []
        needle = self.pattern.lower()
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if needle in line.lower():
                    matches.append(
                        {
                            "file": str(file_path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
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

    def process(self, context: ToolContext) -> Dict[str, Any]:
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

        return {
            "status": "ok",
            "path": str(final_path),
            "topic": self.topic,
            "subtopic": self.subtopic,
        }


class PlannedTopic(BaseModel):
    topic: str
    subtopics: List[str]


class EncyclopediaPlan(BaseModel):
    topics: List[PlannedTopic]


class GenerationStats(BaseModel):
    created_topics: int
    created_articles: int
    expected_topics: int
    expected_articles: int
    warnings: List[str] = Field(default_factory=list)


class EncyclopediaAgent:
    def __init__(
        self,
        client: OpenAI,
        root_path: str,
        topics_count: int,
        articles_per_topic: int,
        model: Optional[str] = None,
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
        self.tools = {
            "mkdir": Mkdir,
            "grep": Grep,
            "write_article": WriteArticle,
        }
        self.verbose = verbose

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def ask_model(self, instructions: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )
        return response_text(response)

    def build_fallback_plan(self) -> EncyclopediaPlan:
        selected_topics = list(TOPIC_CANDIDATES.items())[: self.state.target_topics]
        topics = []
        for topic, subtopics in selected_topics:
            topics.append(
                PlannedTopic(topic=topic, subtopics=subtopics[: self.state.target_articles_per_topic])
            )
        return EncyclopediaPlan(topics=topics)

    def plan_structure(self) -> EncyclopediaPlan:
        available_topics = {
            topic: subtopics[:]
            for topic, subtopics in TOPIC_CANDIDATES.items()
        }
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

        for attempt in range(3):
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

        seen_topics = set()
        for topic in plan.topics:
            if topic.topic not in TOPIC_CANDIDATES:
                raise ValueError(f"Недопустимая тема: {topic.topic}")
            if topic.topic in seen_topics:
                raise ValueError(f"Повтор темы: {topic.topic}")
            seen_topics.add(topic.topic)
            if len(topic.subtopics) != self.state.target_articles_per_topic:
                raise ValueError(f"Неверное количество статей для темы {topic.topic}")
            normalized = [normalize_text(s).lower() for s in topic.subtopics]
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
            f"- {item.subtopic} ({item.topic}): {item.summary}"
            for item in related_articles
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
    def fallback_article_content(topic: str, subtopic: str, related_articles: List[ArticleRecord]) -> str:
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

    def write_article(self, topic: str, subtopic: str, content: str) -> Dict[str, Any]:
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
        warnings: List[str] = []
        expected_topics = len(plan.topics)
        expected_articles = sum(len(topic.subtopics) for topic in plan.topics)

        actual_topic_dirs = [p for p in self.root_path.iterdir() if p.is_dir()]
        actual_article_files = list(self.root_path.rglob("*.md"))

        if len(actual_topic_dirs) != expected_topics:
            warnings.append(
                f"Ожидалось {expected_topics} директорий, найдено {len(actual_topic_dirs)}"
            )
        if len(actual_article_files) != expected_articles:
            warnings.append(
                f"Ожидалось {expected_articles} статей, найдено {len(actual_article_files)}"
            )

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
    def _find_duplicate_titles(files: List[Path]) -> List[str]:
        titles: Dict[str, int] = {}
        duplicates: List[str] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Агент для создания детской энциклопедии")
    parser.add_argument("--path", type=str, default="./encyclopedia", help="Путь для сохранения энциклопедии")
    parser.add_argument("--n", type=int, default=3, help="Количество статей в каждой теме")
    parser.add_argument("--m", type=int, default=3, help="Количество тем")
    parser.add_argument("--quiet", action="store_true", help="Отключить подробный вывод")
    args = parser.parse_args()

    if args.n <= 0 or args.m <= 0:
        raise ValueError("Параметры --n и --m должны быть положительными числами")
    if args.m > len(TOPIC_CANDIDATES):
        raise ValueError(
            f"Максимально доступно {len(TOPIC_CANDIDATES)} тем, получено: {args.m}"
        )

    client = create_client()
    agent = EncyclopediaAgent(
        client=client,
        root_path=args.path,
        topics_count=args.m,
        articles_per_topic=args.n,
        verbose=not args.quiet,
    )

    print(f"Создание энциклопедии: {args.path}")
    print(f"Тем: {args.m}, статей в каждой теме: {args.n}\n")

    stats = agent.run()

    print("\n=== ПАМЯТЬ ===")
    print(agent.memory.get_all())

    print("\n=== ИТОГ ===")
    print(
        f"Создано тем: {stats.created_topics}/{stats.expected_topics}; "
        f"статей: {stats.created_articles}/{stats.expected_articles}"
    )
    if stats.warnings:
        print("Предупреждения:")
        for warning in stats.warnings:
            print(f"- {warning}")
    else:
        print("Проверка пройдена без предупреждений.")


if __name__ == "__main__":
    main()