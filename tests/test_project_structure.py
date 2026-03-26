from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from encyclopedia.agent import EncyclopediaAgent
from encyclopedia.models import EncyclopediaPlan, PlannedTopic
from encyclopedia.utils import sanitize_name, safe_resolve_under_root


class AgentHelpersTest(unittest.TestCase):
    def test_fallback_subtopics_fill_required_count(self) -> None:
        result = EncyclopediaAgent._fallback_subtopics("Космос", ["Планеты"], 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Планеты")

    def test_validate_output_detects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            topic_dir = root / sanitize_name("Космос")
            topic_dir.mkdir(parents=True)
            for name in ("Планеты", "Звёзды"):
                (topic_dir / f"{sanitize_name(name)}.md").write_text("# Одинаковый заголовок\n", encoding="utf-8")

            agent = EncyclopediaAgent(
                client=object(),  # type: ignore[arg-type]
                root_path=str(root),
                topics_count=1,
                articles_per_topic=2,
                model="test-model",
            )
            plan = EncyclopediaPlan(topics=[PlannedTopic(topic="Космос", subtopics=["Планеты", "Звёзды"])])
            stats = agent.validate_output(plan)

            self.assertTrue(any("Повторяющиеся заголовки" in warning for warning in stats.warnings))


class UtilsTest(unittest.TestCase):
    def test_safe_resolve_blocks_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                safe_resolve_under_root(Path(temp_dir), "../outside.md")


if __name__ == "__main__":
    unittest.main()
