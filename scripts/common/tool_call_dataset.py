from typing import Any

from scripts.utils.common import load_json_value


def extract_tool_call_sample_fields(item: dict[str, Any]) -> tuple[str, Any, Any]:
    return item["query"], load_json_value(item["tools"]), load_json_value(item["answers"])


def build_prompt_text(tokenizer: Any, query: str, tools: Any) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_full_text(tokenizer: Any, query: str, tools: Any, answers: Any) -> str:
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "tool_calls": answers},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
