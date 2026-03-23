import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_base_tool_calls import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--num-samples",
        "10",
        "--sample-mode",
        "random",
        "--seed",
        "42",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
