"""Adapt selected freeform Responses tools to upstream JSON function tools."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from errors import UpstreamError


def _leaf_tools(definitions: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Visit named tools inside both flat lists and namespace wrappers."""
    for tool in definitions:
        if tool.get("type") == "namespace":
            yield from _leaf_tools(tool.get("tools", []))
        else:
            yield tool


class CustomToolsAdapter:
    """Keep custom tool identity on the client side and JSON arguments upstream."""

    def __init__(self, payload: dict[str, Any], names: list[str]) -> None:
        self.names: set[str] = set()
        definitions = list(payload.get("tools", []))
        items = payload.get("input")
        if isinstance(items, list):
            for item in items:
                if item.get("type") == "additional_tools":
                    definitions.extend(item.get("tools", []))
            self.names.update(
                item["name"]
                for item in items
                if item.get("type") == "custom_tool_call" and item.get("name") in names
            )
        self.names.update(
            tool["name"] for tool in _leaf_tools(definitions)
            if tool.get("type") == "custom" and tool.get("name") in names
        )
        self.item_ids: set[str] = set()

    def _tools(
        self, definitions: list[dict[str, Any]], *, seen: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Adapt definitions wherever Codex carries them, including namespaces."""
        tools = []
        for original_tool in definitions:
            tool = original_tool
            if tool.get("type") == "namespace":
                children = self._tools(tool.get("tools", []), seen=seen)
                if not children:
                    continue
                tool = {**tool, "tools": children}
            elif seen is not None and isinstance(tool.get("name"), str):
                if tool["name"] in seen:
                    continue
                seen.add(tool["name"])
            if tool.get("type") == "custom" and tool.get("name") in self.names:
                tool = {
                    "type": "function",
                    "name": tool["name"],
                    "description": (
                        tool.get("description", "")
                        + "\nReturn the complete raw tool input as the JSON string field 'input'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                }
            tools.append(tool)
        return tools

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Translate definitions, replayed calls and their matching outputs together."""
        payload = dict(payload)
        if "tools" in payload:
            payload["tools"] = self._tools(payload["tools"])
        items = payload.get("input")
        if isinstance(items, list):
            call_ids = {
                item.get("call_id")
                for item in items
                if item.get("type") == "custom_tool_call" and item.get("name") in self.names
            }
            converted = []
            # Some upstreams reject duplicate definitions across current tools
            # and replayed additions. Current tools, then the latest historical
            # definition, win; calls and outputs remain in their original order.
            seen = {
                tool["name"] for tool in _leaf_tools(payload.get("tools", []))
                if isinstance(tool.get("name"), str)
            }
            for original_item in reversed(items):
                item = original_item
                if item.get("type") == "additional_tools":
                    tools = self._tools(item.get("tools", []), seen=seen)
                    if not tools:
                        continue
                    item = {**item, "tools": tools}
                elif item.get("type") == "custom_tool_call" and item.get("name") in self.names:
                    item = dict(item)
                    item["type"] = "function_call"
                    item["arguments"] = json.dumps({"input": item.pop("input")})
                elif (
                    item.get("type") == "custom_tool_call_output"
                    and item.get("call_id") in call_ids
                ):
                    item = {**item, "type": "function_call_output"}
                converted.append(item)
            payload["input"] = list(reversed(converted))
        choice = payload.get("tool_choice")
        if (
            isinstance(choice, dict)
            and choice.get("type") == "custom"
            and choice.get("name") in self.names
        ):
            payload["tool_choice"] = {**choice, "type": "function"}
        return payload

    @staticmethod
    def _input(arguments: str) -> str:
        try:
            value = json.loads(arguments)["input"]
            if isinstance(value, str):
                return value
        except (ValueError, KeyError, TypeError):
            pass
        raise UpstreamError("Upstream returned invalid custom tool arguments", status_code=502)

    def item(self, item: dict[str, Any], *, added: bool = False) -> dict[str, Any]:
        """Restore custom-call semantics without changing call ids or tool output."""
        if item.get("type") != "function_call" or item.get("name") not in self.names:
            return item
        self.item_ids.add(item["id"])
        result = dict(item)
        arguments = result.pop("arguments", "")
        result["type"] = "custom_tool_call"
        result["input"] = "" if added else self._input(arguments)
        return result

    def response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload.get("output"), list):
            payload = {**payload, "output": [self.item(item) for item in payload["output"]]}
        return payload

    def events(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        kind = event.get("type", "")
        if kind in {"response.output_item.added", "response.output_item.done"}:
            event["item"] = self.item(event["item"], added=kind.endswith("added"))
        elif (
            kind
            in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}
            and event.get("item_id") in self.item_ids
        ):
            if kind.endswith("delta"):
                return []  # Decode the complete JSON string at arguments.done.
            value = self._input(event.pop("arguments"))
            prefix = {k: v for k, v in event.items() if k != "type"}
            return [
                {**prefix, "type": "response.custom_tool_call_input.delta", "delta": value},
                {**prefix, "type": "response.custom_tool_call_input.done", "input": value},
            ]
        elif isinstance(event.get("response"), dict):
            event["response"] = self.response(event["response"])
        return [event]

    async def stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Parse complete SSE frames, preserving arbitrary byte/UTF-8 chunk boundaries."""
        pending = b""
        sequence = 0
        async for chunk in chunks:
            pending = (pending + chunk).replace(b"\r\n", b"\n")
            while b"\n\n" in pending:
                frame, pending = pending.split(b"\n\n", 1)
                data = b"\n".join(
                    line[5:].strip() for line in frame.splitlines() if line.startswith(b"data:")
                )
                if not data or data == b"[DONE]":
                    yield frame + b"\n\n"
                    continue
                try:
                    events = self.events(json.loads(data))
                except (ValueError, KeyError, TypeError) as exc:
                    raise UpstreamError(
                        "Upstream returned an invalid tool event", status_code=502
                    ) from exc
                for event in events:
                    event["sequence_number"] = sequence
                    sequence += 1
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        if pending.strip():
            raise UpstreamError("Upstream ended with an incomplete SSE frame", status_code=502)
