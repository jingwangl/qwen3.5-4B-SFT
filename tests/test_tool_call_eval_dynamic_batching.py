import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.common.tool_call_eval_config import ToolCallEvalConfig
from scripts.utils.tool_call_eval_utils import EvalExample, build_dynamic_batches


def make_args(**overrides):
    args = {
        "base_model_path": "dummy-model",
        "adapter_path": None,
        "data_path": Path("/tmp/test.json"),
        "output_dir": Path("/tmp/output"),
        "num_samples": -1,
        "seed": 42,
        "max_new_tokens": 128,
        "batch_size": 2,
        "max_batch_size": None,
        "bucket_by_length": True,
        "temperature": 0.0,
        "sample_mode": "random",
        "dtype": "auto",
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def make_prepared_items(lengths: list[int]):
    prepared = []
    for index, length in enumerate(lengths):
        prepared.append(
            (
                EvalExample(
                    index=index,
                    query=f"query-{index}",
                    tools=[],
                    gold_calls=[],
                ),
                f"prompt-{index}",
                length,
            )
        )
    return prepared


class ToolCallEvalConfigTests(unittest.TestCase):
    def test_defaults_max_batch_size_to_first_batch_size(self):
        config = ToolCallEvalConfig.from_args(make_args(batch_size=4, max_batch_size=None))

        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.max_batch_size, 4)

    def test_rejects_non_positive_max_batch_size(self):
        with self.assertRaisesRegex(RuntimeError, "max_batch_size 必须大于 0"):
            ToolCallEvalConfig.from_args(make_args(max_batch_size=0))

    def test_rejects_max_batch_size_smaller_than_first_batch(self):
        with self.assertRaisesRegex(RuntimeError, "max_batch_size 不能小于 batch_size"):
            ToolCallEvalConfig.from_args(make_args(batch_size=4, max_batch_size=3))


class BuildDynamicBatchesTests(unittest.TestCase):
    def test_first_batch_sets_budget(self):
        batches, token_budget = build_dynamic_batches(
            prepared_examples=make_prepared_items([10, 9, 8, 7, 6]),
            first_batch_size=2,
            max_batch_size=4,
            max_new_tokens=4,
        )

        self.assertEqual(token_budget, 28)
        self.assertEqual([item[2] for item in batches[0].items], [10, 9])
        self.assertEqual([batch.prompt_token_count for batch in batches], [19, 15, 6])
        self.assertEqual([batch.max_prompt_length for batch in batches], [10, 8, 6])
        self.assertEqual([batch.estimated_token_count for batch in batches], [28, 24, 10])

    def test_shorter_samples_can_expand_later_batches(self):
        batches, token_budget = build_dynamic_batches(
            prepared_examples=make_prepared_items([10, 9, 4, 4, 4, 4]),
            first_batch_size=2,
            max_batch_size=10,
            max_new_tokens=4,
        )

        self.assertEqual(token_budget, 28)
        self.assertEqual(len(batches[1].items), 3)
        self.assertGreater(len(batches[1].items), 2)
        self.assertLessEqual(batches[1].estimated_token_count, token_budget)

    def test_max_batch_size_caps_batch_growth(self):
        batches, token_budget = build_dynamic_batches(
            prepared_examples=make_prepared_items([8, 8, 1, 1, 1, 1, 1]),
            first_batch_size=2,
            max_batch_size=3,
            max_new_tokens=4,
        )

        self.assertEqual(token_budget, 24)
        self.assertEqual([len(batch.items) for batch in batches], [2, 3, 2])
        self.assertEqual([batch.estimated_token_count for batch in batches[1:]], [15, 10])

    def test_preserves_input_order_when_lengths_are_unsorted(self):
        batches, _ = build_dynamic_batches(
            prepared_examples=make_prepared_items([2, 2, 10, 1, 1]),
            first_batch_size=2,
            max_batch_size=4,
            max_new_tokens=4,
        )

        flattened_indices = [
            item[0].index
            for batch in batches
            for item in batch.items
        ]
        self.assertEqual(flattened_indices, [0, 1, 2, 3, 4])

    def test_oversized_sample_runs_alone_with_warning_flag(self):
        batches, token_budget = build_dynamic_batches(
            prepared_examples=make_prepared_items([5, 5, 15, 4, 4]),
            first_batch_size=2,
            max_batch_size=4,
            max_new_tokens=4,
        )

        self.assertEqual(token_budget, 18)
        self.assertEqual([len(batch.items) for batch in batches], [2, 1, 2])
        self.assertFalse(batches[0].exceeds_token_budget)
        self.assertTrue(batches[1].exceeds_token_budget)
        self.assertEqual(batches[1].estimated_token_count, 19)
        self.assertEqual([item[2] for item in batches[2].items], [4, 4])


if __name__ == "__main__":
    unittest.main()
