"""Adapt Responses to existing providers without another upstream transport."""

from __future__ import annotations

import json
import zlib
from collections.abc import AsyncIterator
from typing import Any

from conversion.responses_ingress import ResponsesRequestMapper
from conversion.responses_output import ResponsesOutputMapper
from errors import UpstreamError
from log import get_logger
from providers.base import ClientChannel, Provider, ProviderResult
from routing.schema import RouteLimits


async def _decoded_body(result: ProviderResult) -> AsyncIterator[bytes]:
    """Decode upstream compression before parsing Messages JSON or SSE."""
    encoding = result.headers.get("content-encoding", "identity").lower()
    if encoding not in {"identity", "", "gzip", "deflate"}:
        raise UpstreamError("Unsupported upstream content encoding")
    decoder = (
        zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
        if encoding in {"gzip", "deflate"}
        else None
    )

    async def chunks() -> AsyncIterator[bytes]:
        if isinstance(result.body, bytes):
            yield result.body
        else:
            async for chunk in result.body:
                yield chunk

    try:
        async for chunk in chunks():
            yield decoder.decompress(chunk) if decoder else chunk
        if decoder:
            yield decoder.flush()
            if not decoder.eof:
                raise UpstreamError("Upstream ended with incomplete compressed data")
    except zlib.error as exc:
        raise UpstreamError("Invalid upstream compression") from exc


async def handle_translated_responses(
    payload: dict[str, Any],
    provider: Provider,
    channel: ClientChannel,
    upstream_model: str | None,
    limits: RouteLimits,
) -> ProviderResult:
    """Use provider-owned credentials and retain cleanup before a stream starts."""
    try:
        mapper = ResponsesRequestMapper(payload)
        request = mapper.request(limits.max_tokens_limit or 8192)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise UpstreamError("Invalid translated Responses request", 400) from exc
    if mapper.omitted_tools:
        get_logger(__name__).warning(
            "responses_hosted_tools_unavailable",
            provider=provider.name,
            tools=sorted(set(mapper.omitted_tools)),
        )
    # Never forward a Codex bearer credential, account id, or beta headers to Messages.
    result = await provider.handle_messages(
        json.dumps(request).encode(),
        {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept-encoding": "identity",
        },
        channel,
        upstream_model,
        limits,
    )
    headers = {
        k: v
        for k, v in result.headers.items()
        if k.lower() in {"retry-after", "x-request-id", "request-id"}
    }
    output = ResponsesOutputMapper(payload["model"], mapper)
    if isinstance(result.body, bytes) or result.status_code >= 400:
        try:
            raw = b"".join([chunk async for chunk in _decoded_body(result)])
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise UpstreamError("Upstream returned an invalid Messages response")
            if result.status_code >= 400:
                error = message.get("error", {})
                response = {
                    "error": {
                        "type": error.get("type", "upstream_error"),
                        "code": error.get("type", "upstream_error"),
                        "message": error.get("message", "Upstream request failed"),
                        "param": None,
                    }
                }
            else:
                if not isinstance(message.get("content"), list):
                    raise UpstreamError("Upstream returned an invalid Messages response")
                response = output.unary(message)
            return ProviderResult(
                result.status_code,
                {**headers, "content-type": "application/json"},
                json.dumps(response).encode(),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise UpstreamError("Upstream returned an invalid Messages response") from exc
        finally:
            await result.aclose()

    async def body() -> AsyncIterator[bytes]:
        assert not isinstance(result.body, bytes)
        try:
            async for chunk in output.stream(_decoded_body(result)):
                if await channel.is_disconnected():
                    return
                yield chunk
        finally:
            await result.aclose()

    return ProviderResult(
        result.status_code,
        {**headers, "content-type": "text/event-stream"},
        body(),
        release=result.aclose,
    )
