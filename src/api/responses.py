"""Codex model discovery and native ``POST /v1/responses`` forwarding."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from api.adapters import to_fastapi_response
from dependencies import LoggerDep, RegistryDep
from errors import UpstreamError
from services.codex_catalog import merge_catalogs, model_catalog
from services.monitor import Monitor

router = APIRouter()


@router.websocket("/v1/responses")
async def responses_websocket(http_request: WebSocket) -> None:
    """Signal Codex to use HTTP immediately, without retrying a rejected WebSocket.

    Codex treats handshake status 426 as the transport fallback signal.
    Its built-in OpenAI provider probes WebSockets even behind a custom URL.
    """
    await http_request.send_denial_response(Response(status_code=426))


def _error(message: str, status: int = 400, *, code: str = "invalid_request") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code, "param": None}},
    )


@router.get("/codex/models")
async def codex_models(registry: RegistryDep) -> dict[str, Any]:
    """Return the Codex-native catalog used by its model picker."""
    custom = model_catalog(
        registry.rules,
        {name: provider.cfg for name, provider in registry.providers.items()},
        set(registry.responses_providers),
    )
    return merge_catalogs(registry.codex_catalog, custom)


@router.get("/v1/models")
async def list_models(registry: RegistryDep) -> dict[str, Any]:
    """Return the same usable ids in the standard OpenAI model-list shape."""
    catalog = await codex_models(registry)
    return {
        "object": "list",
        "data": [
            {"id": model["slug"], "object": "model", "created": 0, "owned_by": "router"}
            for model in catalog["models"]
        ],
    }


@router.post("/v1/responses")
async def create_response(
    http_request: Request, registry: RegistryDep, logger: LoggerDep
) -> Response:
    """Resolve by model using the shared rules, preserving native Responses bodies."""
    raw_body = await http_request.body()
    try:
        payload = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        return _error("Invalid JSON body")
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model:
        return _error("Missing 'model' field")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        return _error("'stream' must be a boolean")
    decision = registry.resolve(model, default_provider=registry.responses_default)
    native = registry.responses_providers.get(decision.provider.name)
    if native is None:
        return _error(
            "The selected model's provider does not support native Responses; "
            "configure responses_base_url for a Responses-capable upstream.",
            code="unsupported_model",
        )
    if decision.upstream_model is not None:
        payload["model"] = decision.upstream_model
        raw_body = json.dumps(payload).encode()
    logger.info("route", endpoint="responses", model=model, provider=native.name)
    tracker = Monitor.current().start_request(
        model=model, provider=native.name, endpoint="responses", pricing=decision.pricing,
        stream=payload.get("stream", False),
    )
    try:
        result = await native.handle_responses(
            raw_body, http_request.headers, http_request, query=http_request.url.query
        )
    except UpstreamError as exc:
        tracker.fail(exc)
        return _error(exc.message, exc.status_code, code=exc.error_type)
    except BaseException as exc:
        tracker.fail(exc)
        raise
    return to_fastapi_response(tracker.attach(result))
