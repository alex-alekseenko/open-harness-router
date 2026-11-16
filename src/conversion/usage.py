"""Normalize inclusive OpenAI token usage into disjoint Anthropic buckets."""

from __future__ import annotations

from typing import Any


def normalize_openai_usage(
    payload: dict[str, Any], *, chat: bool = False
) -> dict[str, int]:
    """Split reported input into ordinary input, cache reads and cache writes.

    Missing counters stay absent. Cache writes cannot be inferred from cache
    misses. Bound the breakdown by the inclusive input total so malformed
    upstream details cannot inflate totals or produce negative input counts.
    """
    input_key = "prompt_tokens" if chat else "input_tokens"
    output_key = "completion_tokens" if chat else "output_tokens"
    details_key = "prompt_tokens_details" if chat else "input_tokens_details"
    result = {}
    for source, target in ((input_key, "input_tokens"), (output_key, "output_tokens")):
        value = payload.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            result[target] = max(0, value)
    total = result.get("input_tokens")
    details = payload.get(details_key)
    if total is None or not isinstance(details, dict):
        return result
    remaining = total
    for source, target in (
        ("cached_tokens", "cache_read_input_tokens"),
        ("cache_write_tokens", "cache_creation_input_tokens"),
    ):
        value = details.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[target] = min(value, remaining)
            remaining -= result[target]
    result["input_tokens"] = remaining
    return result
