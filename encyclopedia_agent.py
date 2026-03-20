import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, ClassVar, Dict, List

from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv


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
    if not folder_id:
        folder_id = os.environ.get("folder_id")
    if not folder_id:
        raise ValueError("folder_id is required to build model URI")
    return f"gpt://{folder_id}/yandexgpt/latest"


def normalize_path(path: str) -> str:
    path = path.replace("\n", "").replace("\\n", "")
    path = path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    return str(Path(path)).rstrip("/")


@dataclass
class Memory:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, entry_type: str, content: str, metadata: Optional[dict] = None) -> None:
        self.entries.append(
            {
                "type": entry_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_all(self) -> str:
        if not self.entries:
            return "Память пуста."
        lines = ["=== История работы ==="]
        for i, entry in enumerate(self.entries, 1):
            lines.append(f"{i}. [{entry['type']}] {entry['content']}")
            if entry["metadata"]:
                lines.append(f"   {json.dumps(entry['metadata'], ensure_ascii=False)}")
        return "\n".join(lines)

    def get_articles_list(self) -> str:
        """Возвращает список всех созданных статей для цитирования."""
        articles = [e for e in self.entries if e["type"] == "article_created"]
        if not articles:
            return "Нет статей."
        lines = ["Доступные статьи для ссылок:"]
        for a in articles:
            topic = a["metadata"].get("topic", "")
            subtopic = a["metadata"].get("subtopic", "")
            lines.append(f"- [{subtopic}] (раздел: {topic})")
        return "\n".join(lines)


memory = Memory()


class ToolBase(BaseModel):
    memory: ClassVar[Memory] = memory

    @classmethod
    def set_memory(cls, mem: Memory) -> None:
        cls.memory = mem

    def _add_memory(self, entry_type: str, content: str, metadata: Optional[dict] = None) -> None:
        if self.memory:
            self.memory.add(entry_type, content, metadata)


class Mkdir(ToolBase):
    path: str = Field(description="Путь для создания директории")

    def process(self, session_id: str) -> str:
        try:
            clean_path = normalize_path(self.path)
            os.makedirs(clean_path, exist_ok=True)
            self._add_memory(
                "directory_created", f"Создана: {clean_path}", {"path": clean_path}
            )
            return f"OK: {clean_path}"
        except Exception as e:
            return f"ERROR: {e}"


class Grep(ToolBase):
    pattern: str = Field(description="Паттерн для поиска")
    path: str = Field(description="Путь к файлу")

    def process(self, session_id: str) -> str:
        try:
            clean_path = normalize_path(self.path)
            if not os.path.exists(clean_path):
                return f"NOT_FOUND: {clean_path}"
            with open(clean_path, "r", encoding="utf-8") as f:
                content = f.read()
            matches = [
                line
                for line in content.split("\n")
                if self.pattern.lower() in line.lower()
            ]
            if matches:
                self._add_memory(
                    "search_found",
                    f"Найдено: {len(matches)}",
                    {"pattern": self.pattern},
                )
                return f"FOUND {len(matches)}: " + "; ".join(matches[:3])
            return "NOT_FOUND"
        except Exception as e:
            return f"ERROR: {e}"


class WriteArticle(ToolBase):
    topic: str = Field(description="Тема")
    subtopic: str = Field(description="Статья")
    content: str = Field(description="Содержание")
    filepath: str = Field(description="Путь к файлу")

    def process(self, session_id: str) -> str:
        try:
            clean_path = normalize_path(self.filepath)
            if not clean_path.endswith(".md"):
                clean_path = clean_path + ".md"
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)

            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            md = f"""# {self.subtopic}

**Раздел:** {self.topic}

---

{self.content}

---

*Энциклопедия для детей*

*Дата создания: {created_at}*
"""
            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(md)

            self._add_memory(
                "article_created",
                self.subtopic,
                {"topic": self.topic, "path": clean_path},
            )
            return f"OK: {clean_path}"
        except Exception as e:
            return f"ERROR: {e}"


class Agent:
    def __init__(
        self,
        client: OpenAI,
        instruction: str,
        tools: list = None,
        model: str = None,
        tool_choice: str = "required",
        verbose: bool = True,
    ):
        self.client = client
        self.instruction = instruction
        self.model = model or get_model_uri()
        self.tool_choice = tool_choice
        self.verbose = verbose

        self.tool_map: Dict[str, type] = {}
        self.tools_schema: List[Dict[str, Any]] = []

        for tool in tools or []:
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                self.tool_map[tool.__name__] = tool
                self.tools_schema.append(
                    {
                        "type": "function",
                        "name": tool.__name__,
                        "description": tool.__doc__ or "",
                        "parameters": tool.model_json_schema(),
                    }
                )

        self.user_sessions: Dict[str, Dict[str, Any]] = {}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def __call__(self, message: str, session_id: str = "default") -> Any:
        s = self.user_sessions.get(session_id, {"last_reply_id": None, "history": []})
        s["history"].append({"role": "user", "content": message})

        res = self.client.responses.create(
            model=self.model,
            store=True,
            tools=self.tools_schema if self.tools_schema else None,
            tool_choice=self.tool_choice if self.tools_schema else None,
            instructions=self.instruction,
            previous_response_id=s.get("last_reply_id"),
            input=message,
        )

        max_iters = 30
        for iteration in range(max_iters):
            tool_calls = [item for item in res.output if item.type == "function_call"]

            if not tool_calls:
                break

            outputs = []
            for call in tool_calls:
                if call.name not in self.tool_map:
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": f"ERROR: Unknown tool {call.name}",
                        }
                    )
                    continue

                self._log(
                    f" {call.name}({call.arguments[:80] if call.arguments else ''}...)"
                )
                try:
                    fn = self.tool_map[call.name]
                    if call.arguments:
                        payload = json.loads(call.arguments)
                        obj = fn.model_validate(payload)
                    else:
                        obj = fn()
                    result = obj.process(session_id)
                except Exception as e:
                    result = f"ERROR: {e}"
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )

            if outputs:
                # После выполнения - проси модель продолжить
                res = self.client.responses.create(
                    model=self.model,
                    input=outputs
                    + [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "Продолжи создание статей! Вызови WriteArticle для следующих статей!",
                        }
                    ],
                    tools=self.tools_schema,
                    previous_response_id=res.id,
                    store=True,
                )
                continue

            break

        s["last_reply_id"] = res.id
        s["history"].append({"role": "assistant", "content": res.output_text})
        self.user_sessions[session_id] = s

        return res


def main():
    import argparse

    client = create_client()

    instruction = """Ты создаёшь детскую энциклопедию.

ПОСЛЕДОВАТЕЛЬНОСТЬ (ОБЯЗАТЕЛЬНО):
1. Создай ВСЕ папки тем (Mkdir)
2. Создай ВСЕ статьи в каждой папке (WriteArticle)

ПРИМЕР ВЫЗОВОВ:
После Mkdir("./example/Космос/") сразу вызови:
WriteArticle(topic="Космос", subtopic="Планеты", content="...", filepath="./example/Космос/Планеты.md")
WriteArticle(topic="Космос", subtopic="Звезды", content="...", filepath="./example/Космос/Звезды.md")

ПРАВИЛА:
- Продолжай вызывать WriteArticle ПОКА НЕ создашь все статьи
- НЕ останавливайся после директорий"""

    tools = [Mkdir, Grep, WriteArticle]

    for tool in tools:
        if isinstance(tool, type) and issubclass(tool, ToolBase):
            tool.set_memory(memory)

    agent = Agent(
        client=client,
        instruction=instruction,
        tools=tools,
        tool_choice="required",
        verbose=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./encyclopedia")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--m", type=int, default=3)
    args = parser.parse_args()

    print(f"Создание: {args.path}, тем: {args.m}, статей: {args.n}")

    prompt = f"""Создать детскую энциклопедию.

Темы: Космос, Животные, Наука, Природа, История
Выбери {args.m} тем. В каждой {args.n} статей.

ДЕЙСТВИЯ:
1. Создай папку: Mkdir(path="{args.path}/Космос")
2. Создай статьи:
   WriteArticle(topic="Космос", subtopic="Планеты", content="ПОДРОБНЫЙ ТЕКСТ", filepath="{args.path}/Космос/Планеты.md")
   WriteArticle(topic="Космос", subtopic="Звезды", content="ПОДРОБНЫЙ ТЕКСТ", filepath="{args.path}/Космос/Звезды.md")
   ... и так далее

Вызывай WriteArticle ПОКА НЕ создашь все {args.m * args.n} статей!"""

    result = agent(prompt)

    print("\n=== ПАМЯТЬ ===")
    print(memory.get_all())
    print("\nГотово!")


if __name__ == "__main__":
    main()
