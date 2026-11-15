"""Custom definitions must remain compatible when Codex replays model history."""

from __future__ import annotations

from copy import deepcopy

import pytest

from conversion.responses_tools import CustomToolsAdapter


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("namespaced", [False, True])
def test_custom_definitions_are_adapted_without_a_prior_call(
    history: bool, namespaced: bool,
) -> None:
    definitions = [
        {"type": "custom", "name": "exec", "format": {"type": "text"}},
        {"type": "custom", "name": "apply_patch"},
        {"type": "function", "name": "wait", "parameters": {"type": "object"}},
    ]
    if namespaced:
        definitions = [{"type": "namespace", "name": "functions", "tools": definitions}]
    payload = (
        {"input": [{"type": "additional_tools", "id": "tools_1", "role": "developer",
                    "tools": definitions}]}
        if history else {"tools": definitions, "input": "hi"}
    )
    original = deepcopy(payload)
    adapter = CustomToolsAdapter(payload, ["exec"])
    assert adapter.names == {"exec"}
    rewritten = adapter.request(payload)
    assert payload == original
    if history:
        item = rewritten["input"][0]
        assert {k: v for k, v in item.items() if k != "tools"} == {
            "type": "additional_tools", "id": "tools_1", "role": "developer",
        }
        assert "tools" not in rewritten
        actual = item["tools"]
    else:
        actual = rewritten["tools"]
    if namespaced:
        assert actual[0]["type"] == "namespace"
        assert actual[0]["name"] == "functions"
        actual = actual[0]["tools"]
    assert actual[0]["type"] == "function"
    assert actual[0]["name"] == "exec"
    assert actual[0]["parameters"]["required"] == ["input"]
    assert "format" not in actual[0]
    assert actual[1:] == (definitions[0]["tools"][1:] if namespaced else definitions[1:])


def test_history_definitions_are_deduplicated_without_removing_conversation() -> None:
    payload = {
        "tools": [{"type": "function", "name": "wait", "description": "current"}],
        "input": [
            {"type": "message", "role": "user", "content": "remember this"},
            {"type": "additional_tools", "id": "old", "tools": [
                {"type": "custom", "name": "exec", "description": "old"},
            ]},
            {"type": "custom_tool_call", "id": "ctc", "call_id": "call_1",
             "name": "exec", "input": "read_file()"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "secret"},
            {"type": "additional_tools", "id": "latest", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "tools": [
                    {"type": "custom", "name": "exec", "description": "latest"},
                    {"type": "function", "name": "wait", "description": "historical"},
                    {"type": "custom", "name": "apply_patch"},
                ]},
            ]},
        ],
    }
    original = deepcopy(payload)
    rewritten = CustomToolsAdapter(payload, ["exec"]).request(payload)
    assert payload == original
    assert rewritten["tools"] == payload["tools"]
    items = rewritten["input"]
    assert len(items) == 4
    assert items[0] == payload["input"][0]
    assert items[1]["type"] == "function_call" and items[1]["call_id"] == "call_1"
    assert items[2] == {"type": "function_call_output", "call_id": "call_1", "output": "secret"}
    assert items[3]["id"] == "latest" and items[3]["role"] == "developer"
    tools = items[3]["tools"][0]["tools"]
    assert [tool["name"] for tool in tools] == ["exec", "apply_patch"]
    assert tools[0]["type"] == "function" and tools[0]["description"].startswith("latest")
