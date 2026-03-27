from __future__ import annotations

import argparse

from .agent import EncyclopediaAgent
from .client import create_client
from .constants import TOPIC_CANDIDATES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Агент для создания детской энциклопедии")
    parser.add_argument("--path", type=str, default="./encyclopedia", help="Путь для сохранения энциклопедии")
    parser.add_argument("--n", type=int, default=3, help="Количество статей в каждой теме")
    parser.add_argument("--m", type=int, default=3, help="Количество тем")
    parser.add_argument("--quiet", action="store_true", help="Отключить подробный вывод")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0 or args.m <= 0:
        raise ValueError("Параметры --n и --m должны быть положительными числами")
    if args.m > len(TOPIC_CANDIDATES):
        raise ValueError(f"Максимально доступно {len(TOPIC_CANDIDATES)} тем, получено: {args.m}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

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
