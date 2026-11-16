"""Tool identity and history survive protocol conversion and model switching."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conversion.responses_ingress import ResponsesRequestMapper
from conversion.responses_output import ResponsesOutputMapper
from errors import UpstreamError


def test_custom_tool_namespace_history_and_current_definitions_win() -> None:
    payload = {
        "model": "custom",
        "tools": [
            {
                "type": "namespace",
                "name": "functions",
                "tools": [
                    {"type": "custom", "name": "apply_patch", "description": "Current definition"},
                ],
            }
        ],
        "input": [
            {
                "type": "additional_tools",
                "tools": [
                    {"type": "custom", "name": "apply_patch", "description": "Obsolete definition"}
                ],
            },
            {"role": "user", "content": "Edit file"},
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "namespace": "functions",
                "call_id": "call_1",
                "input": "*** Begin Patch\n*** End Patch",
            },
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Success"},
            {"type": "reasoning", "encrypted_content": "belongs-to-another-model"},
        ],
    }
    mapper = ResponsesRequestMapper(payload)
    request = mapper.request(4096)
    assert len(request["tools"]) == 1
    assert request["tools"][0]["name"] == "functions__apply_patch"
    assert request["tools"][0]["description"].startswith("Current definition")
    tool = request["messages"][1]["content"][0]
    assert tool["type"] == "tool_use"
    assert tool["input"] == {"input": "*** Begin Patch\n*** End Patch"}
    assert request["messages"][2]["content"][0]["tool_use_id"] == "call_1"
    response = ResponsesOutputMapper("custom", mapper).unary(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "functions__apply_patch",
                    "input": {"input": "*** Begin Patch\nnew\n*** End Patch"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
    )
    item = response["output"][0]
    assert item["type"] == "custom_tool_call"
    assert item["namespace"] == "functions"
    assert item["name"] == "apply_patch"
    assert item["call_id"] == "call_2"
    assert item["input"] == "*** Begin Patch\nnew\n*** End Patch"


def test_custom_stream_waits_for_complete_json_and_restores_raw_input() -> None:
    mapper = ResponsesRequestMapper(
        {
            "model": "custom",
            "input": "Hi",
            "tools": [
                {"type": "custom", "name": "exec"},
            ],
        }
    )
    mapper.request(4096)
    output = ResponsesOutputMapper("custom", mapper)
    output.event({"type": "message_start", "message": {"usage": {}}})
    added = output.event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "name": "exec", "id": "call_1", "input": {}},
        }
    )
    assert b'"custom_tool_call"' in added[0]
    for delta in ['{"input":', '"text(1)"}']:
        assert (
            output.event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            == []
        )
    done = output.event({"type": "content_block_stop", "index": 0})
    assert b'"delta": "text(1)"' in done[0]
    assert b'"input": "text(1)"' in done[1]
    complete = output.event({"type": "message_stop"})
    assert b'"custom_tool_call"' in complete[0]


def test_images_developer_messages_and_parallel_calls() -> None:
    mapper = ResponsesRequestMapper(
        {
            "model": "custom",
            "instructions": "System",
            "input": [
                {"role": "developer", "content": "Developer"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
                    ],
                },
                {"type": "function_call", "call_id": "c1", "name": "one", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "two", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "first"},
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": [{"type": "input_text", "text": "second"}],
                },
            ],
        }
    )
    request = mapper.request(100)
    assert request["system"] == "System\n\nDeveloper"
    assert request["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert len(request["messages"][1]["content"]) == 2
    assert len(request["messages"][2]["content"]) == 2


def test_optional_hosted_search_has_explicit_model_instruction() -> None:
    mapper = ResponsesRequestMapper(
        {"model": "custom", "input": "Hi", "tools": [{"type": "web_search"}]}
    )
    request = mapper.request(100)
    assert "tools" not in request
    assert "Hosted web search is unavailable" in request["system"]
    assert mapper.omitted_tools == ["web_search"]


def test_partial_completion_and_cache_usage() -> None:
    mapper = ResponsesRequestMapper({"model": "custom", "input": "Hi"})
    result = ResponsesOutputMapper("custom", mapper).unary(
        {
            "content": [{"type": "text", "text": "Partial"}],
            "stop_reason": "max_tokens",
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 5,
            },
        }
    )
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["usage"]["input_tokens"] == 60
    assert result["usage"]["total_tokens"] == 65


@pytest.mark.parametrize(
    "block",
    [
        {"type": "tool_use", "name": "echo", "id": "c", "input": {}},
        {"type": "text", "text": "Hello"},
    ],
)
def test_empty_tool_arguments_and_complete_initial_blocks(block: dict[str, Any]) -> None:
    mapper = ResponsesRequestMapper({"model": "custom", "input": "Hi"})
    result = ResponsesOutputMapper("custom", mapper).unary({"content": [block], "usage": {}})
    item = result["output"][0]
    if block["type"] == "tool_use":
        assert json.loads(item["arguments"]) == {}
    else:
        assert item["content"][0]["text"] == "Hello"


def test_namespace_collision_is_explicit_error() -> None:
    mapper = ResponsesRequestMapper(
        {
            "model": "custom",
            "input": "Hi",
            "tools": [
                {"type": "function", "name": "ns__echo"},
                {
                    "type": "namespace",
                    "name": "ns",
                    "tools": [{"type": "function", "name": "echo"}],
                },
            ],
        }
    )
    with pytest.raises(UpstreamError, match="collide"):
        mapper.request(100)
