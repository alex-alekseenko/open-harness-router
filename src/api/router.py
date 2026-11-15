"""Aggregator for the public API routers.

The ``/v1/messages`` and ``/v1/messages/count_tokens`` paths are mounted at
the root without an ``/api/v1`` prefix -- the contract is fixed by the
Claude Code protocol. Codex uses native Responses, Images and the model catalogs.
``/health``, ``/dashboard`` and ``/metrics`` are separate utility routers.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.count_tokens import router as count_tokens_router
from api.dashboard import router as dashboard_router
from api.health import router as health_router
from api.images import router as images_router
from api.messages import router as messages_router
from api.metrics import router as metrics_router
from api.responses import router as responses_router


def build_root_router() -> APIRouter:
    """Assemble the root router from all sub-routers.

    Returns:
        The aggregated API router.
    """
    root = APIRouter()
    root.include_router(messages_router)
    root.include_router(count_tokens_router)
    root.include_router(responses_router)
    root.include_router(images_router)
    root.include_router(health_router)
    root.include_router(dashboard_router)
    root.include_router(metrics_router)
    return root
