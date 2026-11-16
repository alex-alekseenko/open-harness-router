"""Convert Messages JSON and incremental SSE into the Responses lifecycle."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from conversion.responses_ingress import ResponsesRequestMapper
from errors import UpstreamError


async def sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Parse SSE independently of byte, UTF-8 and CRLF chunk boundaries."""
    pending = b""
    async for chunk in chunks:
        pending += chunk
        # Normalize only complete CRLF pairs, keeping a trailing CR until the next read.
        pending = pending.replace(b"\r\n", b"\n")
        while b"\n\n" in pending:
            frame, pending = pending.split(b"\n\n", 1)
            data = b"\n".join(
                line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:")
            )
            if data and data != b"[DONE]":
                try:
                    value = json.loads(data)
                except ValueError as exc:
                    raise UpstreamError("Invalid upstream SSE JSON") from exc
                if not isinstance(value, dict):
                    raise UpstreamError("Invalid upstream SSE event")
                yield value
    if pending.strip():
        raise UpstreamError("Upstream ended with an incomplete SSE frame")


class ResponsesOutputMapper:
    """Maintain stable response/item ids and emit completion only after message_stop."""

    def __init__(self, model: str, request: ResponsesRequestMapper) -> None:
        self.request = request
        self.sequence = 0
        self.response: dict[str, Any] = {
            "id": "resp_" + uuid.uuid4().hex,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
            "output": [],
            "error": None,
            "incomplete_details": None,
        }
        self.blocks: dict[int, dict[str, Any]] = {}
        self.indices: dict[int, int] = {}
        self.arguments: dict[int, str] = {}
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.stopped = False

    def frame(self, kind: str, **fields: Any) -> bytes:
        payload = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    def _usage(self) -> dict[str, Any]:
        cached = self.usage.get("cache_read_input_tokens", 0)
        tokens = (
            self.usage.get("input_tokens", 0)
            + cached
            + self.usage.get("cache_creation_input_tokens", 0)
        )
        output = self.usage.get("output_tokens", 0)
        return {
            "input_tokens": tokens,
            "output_tokens": output,
            "total_tokens": tokens + output,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    def finish(self) -> dict[str, Any]:
        self.response["status"] = "incomplete" if self.stop_reason == "max_tokens" else "completed"
        if self.stop_reason == "max_tokens":
            self.response["incomplete_details"] = {"reason": "max_output_tokens"}
        self.response["usage"] = self._usage()
        return self.response

    def start_block(self, index: int, block: dict[str, Any]) -> list[bytes]:
        kind = block["type"]
        identifier = "item_" + uuid.uuid4().hex
        if kind == "text":
            item = {
                "id": identifier,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        elif kind == "tool_use":
            self.arguments[index] = ""
            item = {
                "id": identifier,
                "type": "function_call",
                "call_id": block["id"],
                "name": block["name"],
                "arguments": "",
                "status": "in_progress",
            }
        elif kind == "thinking":
            item = {
                "id": identifier,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": ""}],
            }
        elif kind == "redacted_thinking":
            return []
        else:
            raise UpstreamError(f"Unsupported upstream content block: {kind}")
        self.blocks[index] = item
        output_index = len(self.response["output"])
        self.indices[index] = output_index
        self.response["output"].append(item)
        added = self.request.restore_item(item, added=True)
        # message items are empty until content_part.added.
        added = {**added, "content": []} if kind == "text" else added
        frames = [self.frame("response.output_item.added", output_index=output_index, item=added)]
        if kind == "text":
            frames.append(
                self.frame(
                    "response.content_part.added",
                    item_id=identifier,
                    output_index=output_index,
                    content_index=0,
                    part=item["content"][0],
                )
            )
            if block.get("text"):
                frames.extend(self.delta(index, {"type": "text_delta", "text": block["text"]}))
        elif kind == "thinking":
            frames.append(
                self.frame(
                    "response.reasoning_summary_part.added",
                    item_id=identifier,
                    output_index=output_index,
                    summary_index=0,
                    part=item["summary"][0],
                )
            )
            if block.get("thinking"):
                frames.extend(
                    self.delta(index, {"type": "thinking_delta", "thinking": block["thinking"]})
                )
        elif block.get("input"):
            frames.extend(
                self.delta(
                    index, {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
                )
            )
        return frames

    def delta(self, index: int, delta: dict[str, Any]) -> list[bytes]:
        if index not in self.blocks:
            return []
        item = self.blocks[index]
        fields = {"item_id": item["id"], "output_index": self.indices[index]}
        kind = delta["type"]
        if kind == "text_delta":
            item["content"][0]["text"] += delta["text"]
            return [
                self.frame(
                    "response.output_text.delta", **fields, content_index=0, delta=delta["text"]
                )
            ]
        if kind == "thinking_delta":
            item["summary"][0]["text"] += delta["thinking"]
            return [
                self.frame(
                    "response.reasoning_summary_text.delta",
                    **fields,
                    summary_index=0,
                    delta=delta["thinking"],
                )
            ]
        if kind == "input_json_delta":
            self.arguments[index] += delta["partial_json"]
            item["arguments"] = self.arguments[index]
            restored = self.request.restore_item(item, added=True)
            if restored["type"] != "custom_tool_call":
                return [
                    self.frame(
                        "response.function_call_arguments.delta",
                        **fields,
                        delta=delta["partial_json"],
                    )
                ]
        elif kind not in {"signature_delta", "citations_delta"}:
            raise UpstreamError(f"Unsupported upstream delta: {kind}")
        return []

    def stop_block(self, index: int) -> list[bytes]:
        if index not in self.blocks:
            return []
        item = self.blocks[index]
        fields = {"item_id": item["id"], "output_index": self.indices[index]}
        frames = []
        if item["type"] == "message":
            item["status"] = "completed"
            frames.extend(
                [
                    self.frame(
                        "response.output_text.done",
                        **fields,
                        content_index=0,
                        text=item["content"][0]["text"],
                    ),
                    self.frame(
                        "response.content_part.done",
                        **fields,
                        content_index=0,
                        part=item["content"][0],
                    ),
                ]
            )
        elif item["type"] == "function_call":
            item["arguments"] = self.arguments[index] or "{}"
            try:
                if not isinstance(json.loads(item["arguments"]), dict):
                    raise ValueError
            except ValueError as exc:
                raise UpstreamError("Upstream returned invalid function arguments") from exc
            item["status"] = "completed"
            item = self.request.restore_item(item)
            if item["type"] == "custom_tool_call":
                frames.append(
                    self.frame(
                        "response.custom_tool_call_input.delta", **fields, delta=item["input"]
                    )
                )
                frames.append(
                    self.frame(
                        "response.custom_tool_call_input.done", **fields, input=item["input"]
                    )
                )
            else:
                frames.append(
                    self.frame(
                        "response.function_call_arguments.done",
                        **fields,
                        arguments=item["arguments"],
                    )
                )
            self.response["output"][self.indices[index]] = item
        else:
            frames.extend(
                [
                    self.frame(
                        "response.reasoning_summary_text.done",
                        **fields,
                        summary_index=0,
                        text=item["summary"][0]["text"],
                    ),
                    self.frame(
                        "response.reasoning_summary_part.done",
                        **fields,
                        summary_index=0,
                        part=item["summary"][0],
                    ),
                ]
            )
        frames.append(
            self.frame("response.output_item.done", output_index=self.indices[index], item=item)
        )
        return frames

    def event(self, event: dict[str, Any]) -> list[bytes]:
        kind = event.get("type")
        if kind == "message_start":
            self.usage.update(event.get("message", {}).get("usage", {}))
            return [
                self.frame("response.created", response=self.response),
                self.frame("response.in_progress", response=self.response),
            ]
        if kind == "content_block_start":
            return self.start_block(event["index"], event["content_block"])
        if kind == "content_block_delta":
            return self.delta(event["index"], event["delta"])
        if kind == "content_block_stop":
            return self.stop_block(event["index"])
        if kind == "message_delta":
            self.usage.update(event.get("usage", {}))
            self.stop_reason = event.get("delta", {}).get("stop_reason", self.stop_reason)
        elif kind == "message_stop":
            self.stopped = True
            response = self.finish()
            return [self.frame("response." + response["status"], response=response)]
        elif kind == "error":
            raise UpstreamError("Upstream failed while streaming the translated response")
        return []

    def unary(self, message: dict[str, Any]) -> dict[str, Any]:
        """Reuse the same block conversion and usage accounting for JSON responses."""
        self.usage.update(message.get("usage", {}))
        self.stop_reason = message.get("stop_reason")
        for index, block in enumerate(message.get("content", [])):
            self.start_block(index, block)
            self.stop_block(index)
        return self.finish()

    async def stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Never report successful completion after an interrupted upstream stream."""
        try:
            async for event in sse_events(chunks):
                for frame in self.event(event):
                    yield frame
                if self.stopped:
                    return
            raise UpstreamError("Upstream ended before message_stop")
        except (UpstreamError, ValueError, KeyError, TypeError):
            yield self.frame(
                "error", code="upstream_error", message="Upstream stream interrupted", param=None
            )
