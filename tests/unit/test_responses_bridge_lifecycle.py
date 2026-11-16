"""Translated Responses release upstream resources even before iteration begins."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from providers.base import ProviderResult
from providers.responses_bridge import handle_translated_responses
from routing.schema import RouteLimits


@pytest.mark.parametrize("started", [False, True])
async def test_result_close_releases_unstarted_or_started_stream(started: bool) -> None:
    released = asyncio.Event()

    async def release() -> None:
        released.set()

    async def upstream() -> AsyncIterator[bytes]:
        yield b'data: {"type":"message_start","message":{"usage":{}}}\n\n'
        await asyncio.Event().wait()

    provider = SimpleNamespace(
        name="test",
        handle_messages=AsyncMock(
            return_value=ProviderResult(
                200,
                {"content-type": "text/event-stream"},
                upstream(),
                release=release,
            )
        ),
    )
    channel = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    result = await handle_translated_responses(
        {"model": "custom", "input": "Hi", "stream": True},
        provider,
        channel,
        None,
        RouteLimits(max_tokens_limit=4096, context_window=None),
    )
    assert not isinstance(result.body, bytes)
    if started:
        await anext(result.body)
    await result.aclose()
    assert released.is_set()


async def test_cancelled_iteration_closes_upstream() -> None:
    released = asyncio.Event()
    entered = asyncio.Event()

    async def upstream() -> AsyncIterator[bytes]:
        yield b'data: {"type":"message_start","message":{"usage":{}}}\n\n'
        entered.set()
        await asyncio.Event().wait()

    async def release() -> None:
        released.set()

    provider = SimpleNamespace(
        name="test",
        handle_messages=AsyncMock(
            return_value=ProviderResult(
                200,
                {},
                upstream(),
                release=release,
            )
        ),
    )
    result = await handle_translated_responses(
        {"model": "custom", "input": "Hi", "stream": True},
        provider,
        SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
        None,
        RouteLimits(max_tokens_limit=4096, context_window=None),
    )

    async def consume() -> None:
        assert not isinstance(result.body, bytes)
        async for _ in result.body:
            pass

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()
