import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_preprocess.common import (  # noqa: E402
    build_length_stats,
    deduplicate_records,
    sample_records,
    split_records_by_ratio,
)


class DataPreprocessCommonTests(unittest.TestCase):
    def test_deduplicate_records_keeps_first_unique_item(self):
        duplicate = {
            "query": "weather ",
            "tools": '[{"name":"get_weather"}]',
            "answers": '[{"name":"get_weather","arguments":{"city":"Paris"}}]',
        }
        data = [
            duplicate,
            {
                "query": "weather",
                "tools": '[{"name":"get_weather"}]',
                "answers": '[{"name":"get_weather","arguments":{"city":"Paris"}}]',
            },
            {
                "query": "calendar",
                "tools": '[{"name":"get_calendar"}]',
                "answers": '[{"name":"get_calendar","arguments":{"day":"today"}}]',
            },
        ]

        result = deduplicate_records(data)

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], duplicate)

    def test_split_records_by_ratio_matches_expected_sizes(self):
        train, val, test = split_records_by_ratio(
            list(range(20)),
            seed=42,
            train_ratio=0.8,
            val_ratio=0.05,
        )

        self.assertEqual(len(train), 16)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 3)
        self.assertEqual(sorted(train + val + test), list(range(20)))

    def test_sample_records_is_reproducible(self):
        first = sample_records(list(range(10)), size=4, seed=7)
        second = sample_records(list(range(10)), size=4, seed=7)

        self.assertEqual(first, second)

    def test_build_length_stats_uses_linear_percentiles(self):
        stats = build_length_stats([1, 2, 3, 4])

        self.assertEqual(stats["num_samples"], 4)
        self.assertEqual(stats["max"], 4)
        self.assertAlmostEqual(stats["p50"], 2.5)
        self.assertAlmostEqual(stats["p90"], 3.7)


if __name__ == "__main__":
    unittest.main()
