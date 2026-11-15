"""Native Codex image generation/editing using the explicitly trusted fallback."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import Response

from api.adapters import openai_error, to_fastapi_response
from dependencies import LoggerDep, RegistryDep
from errors import UpstreamError
from services.monitor import Monitor

router = APIRouter()


async def _image_request(
    operation: Literal["generations", "edits"],
    request: Request,
    registry: RegistryDep,
    logger: LoggerDep,
) -> Response:
    raw_body = await request.body()
    try:
        payload = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        return openai_error("Invalid JSON body")
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model:
        return openai_error("Missing 'model' field")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        return openai_error("'stream' must be a boolean")
    fallback = registry.responses_default
    native = registry.responses_providers.get(fallback) if fallback else None
    if native is None or not native.cfg.forward_client_auth:
        return openai_error(
            "Codex Images requires an authenticated native responses_default with "
            "responses_forward_client_auth=true; image requests are not sent to custom models.",
            code="unsupported_image_provider",
        )
    endpoint = f"images/{operation}"
    logger.info("route", endpoint=endpoint, model=model, provider=native.name)
    tracker = Monitor.current().start_request(
        model=model, provider=native.name, endpoint=endpoint, pricing=None,
        stream=payload.get("stream", False),
    )
    try:
        result = await native.handle_images(
            operation, raw_body, request.headers, request, query=request.url.query,
        )
    except UpstreamError as exc:
        tracker.fail(exc)
        return openai_error(exc.message, exc.status_code, code=exc.error_type)
    except BaseException as exc:
        tracker.fail(exc)
        raise
    return to_fastapi_response(tracker.attach(result))


@router.post("/v1/images/generations")
async def generate_image(request: Request, registry: RegistryDep, logger: LoggerDep) -> Response:
    """Preserve Codex's JSON generation request and existing native authentication."""
    return await _image_request("generations", request, registry, logger)


@router.post("/v1/images/edits")
async def edit_image(request: Request, registry: RegistryDep, logger: LoggerDep) -> Response:
    """Preserve Codex's JSON image references without logging their contents."""
    return await _image_request("edits", request, registry, logger)
