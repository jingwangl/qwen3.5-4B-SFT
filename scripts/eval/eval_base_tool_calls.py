import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.common.tool_call_eval_runner import run_tool_call_eval  # noqa: E402
from scripts.eval.eval_base_tool_calls_config import EvalBaseToolCallsConfig  # noqa: E402


def main() -> int:
    config = EvalBaseToolCallsConfig.from_cli()
    return run_tool_call_eval(config)


if __name__ == "__main__":
    raise SystemExit(main())
