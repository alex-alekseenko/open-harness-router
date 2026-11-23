"""Generate a dedicated Codex catalog and profile without editing user defaults."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from conversion.responses_ingress import can_translate_responses, is_trusted_openai_responses_url
from errors import ConfigError
from routing.config_loader import load_routing_config
from services.codex_catalog import load_catalog, merge_catalogs, model_catalog
from services.config_files import write_text_atomic
from settings import Settings

PROFILE_NAME = "open-harness-router"


def main(argv: list[str] | None = None) -> int:
    """Write generated files under ``state/codex`` or the explicit output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("state/codex"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8787/v1")
    parser.add_argument("--model", help="default model id (defaults to the first catalog entry)")
    parser.add_argument(
        "--base-catalog",
        type=Path,
        help="merge native Codex models (defaults to routing.codex_catalog_path)",
    )
    parser.add_argument(
        "--static-catalog",
        action="store_true",
        help="pin the generated catalog instead of live native-account discovery",
    )
    args = parser.parse_args(argv)
    try:
        config = load_routing_config(Settings().routing.config_path)
        responses_providers = {
            name
            for name, cfg in config.providers.items()
            if cfg.responses_base_url is not None
            or cfg.api_flavor == "responses"
            or can_translate_responses(cfg)
        }
        responses_default_provider = (
            config.responses_default.provider if config.responses_default is not None else None
        )
        responses_default_cfg = (
            config.providers[responses_default_provider]
            if responses_default_provider is not None
            else None
        )
        trusted_responses_default = (
            responses_default_cfg is not None
            and responses_default_cfg.responses_forward_client_auth
            and responses_default_cfg.tls_verify_hostname
            and is_trusted_openai_responses_url(
                responses_default_cfg.responses_base_url
                or (
                    responses_default_cfg.base_url
                    if responses_default_cfg.api_flavor == "responses"
                    else ""
                )
            )
        )
        custom_providers = {
            name
            for name in responses_providers
            if name != responses_default_provider or not trusted_responses_default
        }
        catalog = model_catalog(config.rules, config.providers, responses_providers)
        source = args.base_catalog or config.codex_catalog_path
        if source is not None:
            if config.responses_default is None:
                raise ConfigError(
                    "merging native models requires responses_default in routing.yaml"
                )
            catalog = merge_catalogs(load_catalog(source), catalog)
        else:
            catalog = merge_catalogs({"models": []}, catalog)
        ids = [model["slug"] for model in catalog["models"]]
        if not ids:
            raise ConfigError(
                "no Responses models to advertise; configure an own-key provider or "
                "responses_base_url, "
                "and client_models (or an exact rule) in routing.yaml"
            )
        model = args.model or ids[0]
        if model not in ids:
            raise ConfigError(f"model '{model}' is not in the generated Codex catalog")
        selected = next(entry for entry in catalog["models"] if entry["slug"] == model)
        effort = selected.get("default_reasoning_level")
        effort_line = f"model_reasoning_effort = {json.dumps(effort)}\n" if effort else ""
        forward_auth = (
            config.responses_default is not None
            and config.providers[config.responses_default.provider].responses_forward_client_auth
        )
        # A combined profile keeps Codex's search defaults for native models.
        search_line = "" if source is not None else 'web_search = "disabled"\n'
        if not args.base_url.startswith(("http://", "https://")):
            raise ConfigError("--base-url must be an HTTP(S) API base URL")
        output_dir = args.output_dir.expanduser().resolve()
        catalog_path = output_dir / f"{PROFILE_NAME}.models.json"
        profile_path = output_dir / f"{PROFILE_NAME}.config.toml"
        # JSON strings are valid TOML basic strings for these paths/URLs.
        # Native-auth profiles keep the built-in identity so existing OpenAI
        # threads can resume. Retain the legacy provider definition for threads
        # created with earlier router profiles (select it explicitly on resume).
        provider_id = "openai" if forward_auth else PROFILE_NAME
        native_url_line = (
            f"openai_base_url = {json.dumps(args.base_url.rstrip('/'))}\n" if forward_auth else ""
        )
        fallback = (
            config.providers[config.responses_default.provider]
            if forward_auth and config.responses_default is not None
            else None
        )
        discovery_url = urlsplit(fallback.responses_base_url or "") if fallback else None
        live_catalog = (
            discovery_url is not None
            and discovery_url.hostname == "chatgpt.com"
            and discovery_url.path.rstrip("/") == "/backend-api/codex"
            and not args.static_catalog
        )
        catalog_line = (
            "" if live_catalog else f"model_catalog_json = {json.dumps(str(catalog_path))}\n"
        )
        # V2 sends nested Code Mode tool arguments as plaintext to the child.
        # Keep the model catalog's choice of multi-agent version.
        has_custom_route = any(rule.provider in custom_providers for rule in config.rules)
        if responses_default_provider is not None:
            has_custom_route = has_custom_route or responses_default_provider in custom_providers
        has_non_gpt_route = any(not model_id.startswith("gpt-") for model_id in ids)
        agent_tools = (
            "\n[features.multi_agent_v2]\nnon_code_mode_only = false\n"
            if has_custom_route or has_non_gpt_route or live_catalog
            else ""
        )
        profile = (
            f"model = {json.dumps(model)}\n"
            f"model_provider = {json.dumps(provider_id)}\n"
            f"{native_url_line}"
            f"{catalog_line}"
            f"{effort_line}"
            f"{search_line}\n"
            f"[model_providers.{PROFILE_NAME}]\n"
            'name = "Open Harness Router"\n'
            f"base_url = {json.dumps(args.base_url.rstrip('/'))}\n"
            'wire_api = "responses"\n'
            f"requires_openai_auth = {str(forward_auth).lower()}\n"
            # The router parses JSON to choose a model, and currently serves
            # HTTP Responses rather than accepting compressed request bodies.
            + ("\n[features]\nenable_request_compression = false\n" if forward_auth else "")
            + agent_tools
        )
        write_text_atomic(catalog_path, json.dumps(catalog, indent=2) + "\n")
        write_text_atomic(profile_path, profile)
    except (ConfigError, ValidationError, OSError) as exc:
        print(f"open-harness-router: {exc}", file=sys.stderr)
        return 1
    print(f"WROTE {catalog_path} ({len(ids)} models)")
    print(f"WROTE {profile_path}")
    print(
        f"Install both files in your Codex home with --output-dir ~/.codex, then run "
        f"'codex -p {PROFILE_NAME}' (Codex CLI 0.169+)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
