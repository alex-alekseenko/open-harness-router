"""Cache read/write accounting across OpenAI protocols and response modes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
    convert_responses_streaming_to_claude_with_cancellation,
    convert_responses_to_claude_response,
)
from conversion.usage import normalize_openai_usage
from models.claude import ClaudeMessagesRequest
from services.monitor import UsageScanner
from services.reasoning_cache import ReasoningCache


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_translated_usage_has_disjoint_cache_buckets(chat: bool, stream: bool) -> None:
    """All four conversion paths preserve inclusive input without double counting."""
    request = ClaudeMessagesRequest.model_validate({
        "model": "test", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}],
    })
    usage = {
        "prompt_tokens" if chat else "input_tokens": 100,
        "completion_tokens" if chat else "output_tokens": 10,
        "prompt_tokens_details" if chat else "input_tokens_details": {
            "cached_tokens": 60, "cache_write_tokens": 25,
        },
    }
    response: dict[str, Any] = {
        "usage": usage, "id": "test", "status": "completed",
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
    }
    scanner = UsageScanner()
    if stream:
        event = response if chat else {"type": "response.completed", "response": response}

        async def chunks() -> AsyncIterator[str]:
            yield "data: " + json.dumps(event)

        logger = MagicMock()
        channel = MagicMock()
        channel.is_disconnected = AsyncMock(return_value=False)
        client = MagicMock()
        converted_stream = (
            convert_openai_streaming_to_claude_with_cancellation(
                chunks(), request, logger, channel, client, "test") if chat
            else convert_responses_streaming_to_claude_with_cancellation(
                chunks(), request, logger, channel, client, "test",
                reasoning_cache=ReasoningCache())
        )
        async for chunk in converted_stream:
            scanner.feed_sse(chunk.encode())
    else:
        converted = (convert_openai_to_claude_response(response, request) if chat
                     else convert_responses_to_claude_response(
                         response, request, reasoning_cache=ReasoningCache()))
        scanner.feed_json(json.dumps(converted).encode())
    assert scanner.usage.as_dict() == {
        "input_tokens": 15, "output_tokens": 10,
        "cache_creation_input_tokens": 25, "cache_read_input_tokens": 60,
    }
    assert scanner.usage.total() == 110


@pytest.mark.parametrize("details", [
    {"cached_tokens": 1000, "cache_write_tokens": 1000},
    {"cached_tokens": -1, "cache_write_tokens": True},
    {"cached_tokens": None, "cache_write_tokens": "40"},
])
def test_invalid_cache_details_cannot_inflate_usage(details: dict[str, Any]) -> None:
    usage = normalize_openai_usage({"input_tokens": 100, "input_tokens_details": details})
    assert sum(usage.values()) == 100
    assert all(value >= 0 for value in usage.values())


def test_missing_write_count_is_not_inferred_from_uncached_input() -> None:
    usage = normalize_openai_usage({
        "input_tokens": 100, "input_tokens_details": {"cached_tokens": 60},
    })
    assert usage == {"input_tokens": 40, "cache_read_input_tokens": 60}
