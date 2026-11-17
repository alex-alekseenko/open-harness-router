"""Native OpenAI forwarding, sharing the passthrough transport and cleanup."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import httpx

from const import UPSTREAM_STREAM_INTERRUPTED_MESSAGE
from conversion.responses_tools import CustomToolsAdapter
from errors import ConfigError, UpstreamError
from providers.base import ClientChannel, ProviderResult
from providers.passthrough import PassthroughProvider
from services.codex_catalog import validate_catalog


class NativeOpenAIProvider(PassthroughProvider):
    """Forward native Responses and Images through the same upstream transport.

    The factory disables Anthropic tool folding. Own-key providers inject
    Bearer authentication; the explicitly configured native fallback can
    preserve the client's own OpenAI credentials.
    """

    async def fetch_model_catalog(
        self,
        client_headers: Mapping[str, str],
        *,
        client_version: str | None,
    ) -> dict[str, Any]:
        """Fetch this caller's native catalog without reading or storing login credentials."""
        if not self.cfg.forward_client_auth:
            raise UpstreamError("Native catalog discovery requires client authentication", 400)
        headers = self._build_headers(
            {
                k: v
                for k, v in client_headers.items()
                if k.lower() not in {"if-none-match", "if-modified-since", "accept-encoding"}
            }
        )
        headers.update({"accept": "application/json", "accept-encoding": "identity"})
        params = {"client_version": client_version} if client_version is not None else None
        try:
            async with self._client.stream(
                "GET",
                "models",
                params=params,
                headers=headers,
                timeout=20.0,
            ) as response:
                if response.status_code != 200:
                    status = response.status_code if response.status_code >= 400 else 502
                    raise UpstreamError("Native model discovery failed", status)
                await response.aread()
                try:
                    return validate_catalog(response.json())
                except (ValueError, ConfigError) as exc:
                    raise UpstreamError("Upstream returned an invalid Codex model catalog") from exc
        except httpx.TimeoutException as exc:
            raise UpstreamError("Native model discovery timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError("Native model discovery failed", 502) from exc

    async def handle_images(
        self,
        operation: Literal["generations", "edits"],
        raw_body: bytes,
        client_headers: Mapping[str, str],
        client_channel: ClientChannel,
        *,
        query: str = "",
    ) -> ProviderResult:
        """Forward native Images JSON without applying Responses tool adaptation."""
        path = f"images/{operation}"
        if query:
            path += f"?{query}"
        # Codex's Images client parses the transport body as JSON. Negotiate
        # identity explicitly instead of httpx injecting its gzip defaults.
        headers = {k: v for k, v in client_headers.items() if k.lower() != "accept-encoding"}
        headers["accept-encoding"] = "identity"
        return await self._proxy(path, raw_body, headers, client_channel)

    async def handle_responses(
        self,
        raw_body: bytes,
        client_headers: Mapping[str, str],
        client_channel: ClientChannel,
        *,
        query: str = "",
    ) -> ProviderResult:
        """Append only ``responses`` to the configured API base URL."""
        path = f"responses?{query}" if query else "responses"
        if not self.cfg.responses_custom_tools_as_functions:
            return await self._proxy(path, raw_body, client_headers, client_channel)
        payload = json.loads(raw_body)
        adapter = CustomToolsAdapter(payload, self.cfg.responses_custom_tools_as_functions)
        if not adapter.names:
            return await self._proxy(path, raw_body, client_headers, client_channel)
        raw_body = json.dumps(adapter.request(payload)).encode()
        headers = {k: v for k, v in client_headers.items() if k.lower() != "accept-encoding"}
        headers["accept-encoding"] = "identity"
        result = await self._proxy(path, raw_body, headers, client_channel)
        if result.status_code >= 400:
            return result
        if isinstance(result.body, bytes):
            try:
                body = json.dumps(adapter.response(json.loads(result.body))).encode()
            except (ValueError, KeyError, TypeError) as exc:
                raise UpstreamError(
                    "Upstream returned an invalid tool response", status_code=502
                ) from exc
            return ProviderResult(result.status_code, result.headers, body)

        async def body_iter() -> AsyncIterator[bytes]:
            assert not isinstance(result.body, bytes)
            try:
                async for chunk in adapter.stream(result.body):
                    yield chunk
            except UpstreamError:
                yield self._stream_error_frame()
            finally:
                await result.aclose()

        return ProviderResult(
            result.status_code, result.headers, body_iter(), release=result.aclose
        )

    def _stream_error_frame(self) -> bytes:
        """Report a transport failure in the OpenAI streaming error shape."""
        payload = {
            "type": "error",
            "code": "upstream_error",
            "message": UPSTREAM_STREAM_INTERRUPTED_MESSAGE,
            "param": None,
        }
        return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
