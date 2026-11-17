"""Build Codex model metadata from the same rules that route inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from errors import ConfigError
from routing.matcher import match_model
from routing.schema import (
    CodexModelCfg,
    ProviderCfg,
    RouteLimits,
    RoutingRule,
    advertised_model_ids,
)


def load_catalog(path: Path) -> dict[str, Any]:
    """Read a Codex catalog or models_cache.json without its account/cache metadata."""
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read Codex model catalog {path}: {exc}") from exc
    return {"models": validate_catalog(payload)["models"]}


def validate_catalog(payload: Any) -> dict[str, Any]:
    """Validate native discovery without stripping upstream protocol metadata."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ConfigError("Codex catalog must contain a models array")
    seen: set[str] = set()
    for entry in payload["models"]:
        slug = entry.get("slug") if isinstance(entry, dict) else None
        if not isinstance(slug, str) or not slug or slug in seen:
            raise ConfigError("Codex catalog has a missing or duplicate model slug")
        seen.add(slug)
    return payload


def merge_catalogs(base: dict[str, Any], custom: dict[str, Any]) -> dict[str, Any]:
    """Keep native capabilities intact and append custom entries; explicit rules win collisions."""
    custom_ids = {model["slug"] for model in custom["models"]}
    models = [model for model in base["models"] if model["slug"] not in custom_ids]
    priority = max((model.get("priority", 0) for model in models), default=-1) + 1
    template = next(
        (
            (model.get("model_messages") or {}).get("instructions_template")
            or model.get("base_instructions")
            for model in base["models"]
            if (model.get("model_messages") or {}).get("instructions_template")
            or model.get("base_instructions")
        ),
        "",
    )
    for index, model in enumerate(custom["models"]):
        identity = (
            f"You are Codex, a coding agent using {model['display_name']}. "
            f"The selected inference model ID is {model['slug']}. "
            "Codex is the agent/client name, not the backend model. "
            "When asked which model you use, report this selected model; "
            "earlier GPT identity statements in conversation history do not describe it."
        )
        # Retain the existing coding workflow while replacing its OpenAI
        # self-identification. Missing templates must not trigger GPT defaults.
        opening = "You are Codex, an agent based on GPT-6."
        instructions = (
            template.replace(opening, identity, 1)
            if template.startswith(opening)
            else identity + "\n\n" + template
        )
        models.append(
            {
                **model,
                "priority": priority + index,
                "model_messages": {
                    **(model.get("model_messages") or {}),
                    "instructions_template": instructions,
                },
                "base_instructions": instructions,
            }
        )
    return {**base, "models": models}


def model_catalog(
    rules: list[RoutingRule],
    providers: dict[str, ProviderCfg],
    responses_provider_names: set[str],
) -> dict[str, Any]:
    """List explicit model ids whose first matching rule can serve Responses.

    ``responses_provider_names`` includes native transports and Messages adapters.
    ``providers`` maps names to ``ProviderCfg`` objects. A prefix is never
    guessed into a model id; list its concrete ids in ``client_models``.
    """
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        for model in advertised_model_ids(rule):
            if model in seen:
                continue
            seen.add(model)
            first = next(r for r in rules if match_model(r.match, model))
            if first is not rule or rule.provider not in responses_provider_names:
                continue
            cfg = rule.codex or CodexModelCfg()
            limits = RouteLimits.resolve(providers[rule.provider], rule)
            entry: dict[str, Any] = {
                "slug": model,
                "display_name": cfg.display_name or model,
                "description": f"{model} — {rule.provider} via Open Harness Router",
                "default_reasoning_level": cfg.default_reasoning_effort,
                "supported_reasoning_levels": [
                    {"effort": effort, "description": f"{effort.capitalize()} reasoning"}
                    for effort in cfg.reasoning_efforts
                ],
                "shell_type": "shell_command",
                "apply_patch_tool_type": "freeform",
                "visibility": "list",
                "supported_in_api": True,
                "priority": len(models),
                "base_instructions": "",
                "supports_reasoning_summaries": bool(cfg.reasoning_efforts),
                "supports_reasoning_summary_parameter": bool(cfg.reasoning_efforts),
                "default_reasoning_summary": "none",
                "support_verbosity": False,
                "truncation_policy": {"mode": "bytes", "limit": 10000},
                "experimental_supported_tools": [],
                "input_modalities": cfg.input_modalities,
                "tool_mode": "direct",
            }
            window = cfg.context_window or limits.context_window
            if window is not None:
                entry["context_window"] = window
                entry["max_context_window"] = window
            models.append(entry)
    return {"models": models}
