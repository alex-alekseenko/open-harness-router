"""Offline Codex configuration generation and native capability validation."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cli.sync_codex_config import main
from errors import ConfigError
from routing.schema import CodexModelCfg, ProviderCfg
from services.codex_catalog import load_catalog, merge_catalogs


@pytest.mark.parametrize("url", ["", "ftp://example.com", "/v1"])
def test_responses_base_url_must_use_http(url: str) -> None:
    with pytest.raises(ValidationError, match="HTTP"):
        ProviderCfg(
            type="passthrough",
            base_url="https://example.com",
            forward_client_auth=False,
            api_key_env="EXAMPLE_KEY",
            responses_base_url=url,
        )


def test_responses_cannot_forward_claude_client_credentials() -> None:
    with pytest.raises(ValidationError, match="requires api_key_env"):
        ProviderCfg(
            type="passthrough",
            base_url="https://example.com",
            responses_base_url="https://example.com/v1",
        )


def test_codex_default_effort_must_be_supported() -> None:
    with pytest.raises(ValidationError, match="included in reasoning_efforts"):
        CodexModelCfg(reasoning_efforts=["low"], default_reasoning_effort="high")


def test_generator_writes_parseable_dedicated_profile_without_needing_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "routing.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "anthropic": {"type": "passthrough", "base_url": "https://api.anthropic.com"},
                    "kimi": {
                        "type": "passthrough",
                        "base_url": "https://api.kimi.com/coding",
                        "forward_client_auth": False,
                        "api_key_env": "KEY_NOT_SET",
                        "responses_base_url": "https://api.kimi.com/coding/v1",
                    },
                },
                "rules": [
                    {"match": {"type": "exact", "value": "kimi-for-coding"}, "provider": "kimi"}
                ],
                "default": {"provider": "anthropic"},
            }
        )
    )
    monkeypatch.setenv("ROUTER_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("KEY_NOT_SET", raising=False)
    output_dir = tmp_path / "codex config"
    assert main(["--output-dir", str(output_dir)]) == 0
    profile = tomllib.loads((output_dir / "open-harness-router.config.toml").read_text())
    assert profile["model"] == "kimi-for-coding"
    provider = profile["model_providers"]["open-harness-router"]
    assert provider["base_url"] == "http://127.0.0.1:8787/v1"
    assert provider["requires_openai_auth"] is False
    assert "env_key" not in provider
    catalog_path = Path(profile["model_catalog_json"])
    assert json.loads(catalog_path.read_text())["models"][0]["slug"] == "kimi-for-coding"
    assert not (output_dir / "config.toml").exists()
    before = catalog_path.read_text()
    assert main(["--output-dir", str(output_dir), "--model", "not-advertised"]) == 1
    assert catalog_path.read_text() == before

    # A combined profile must preserve native capabilities and subscription
    # authentication, rather than merely displaying unusable model names.
    native = {
        "slug": "gpt-native",
        "default_reasoning_level": "high",
        "model_messages": {"instructions_template": "native instructions"},
    }
    native_path = tmp_path / "native-cache.json"
    native_path.write_text(json.dumps({"models": [native], "identity": "not-for-export"}))
    config = yaml.safe_load(config_path.read_text())
    config["providers"]["codex_openai"] = {
        "type": "passthrough",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "responses_base_url": "https://chatgpt.com/backend-api/codex",
        "forward_client_auth": True,
        "responses_forward_client_auth": True,
    }
    config["responses_default"] = {"provider": "codex_openai"}
    config["codex_catalog_path"] = str(native_path)
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--output-dir", str(output_dir)]) == 0
    profile = tomllib.loads((output_dir / "open-harness-router.config.toml").read_text())
    assert profile["model"] == "gpt-native"
    assert profile["model_provider"] == "openai"
    assert profile["openai_base_url"] == "http://127.0.0.1:8787/v1"
    assert profile["features"]["enable_request_compression"] is False
    # Legacy provider remains selectable for threads created by old profiles.
    assert profile["model_providers"]["open-harness-router"]["requires_openai_auth"] is True
    catalog = json.loads(catalog_path.read_text())
    assert catalog["models"][0] == native
    assert [m["slug"] for m in catalog["models"]] == ["gpt-native", "kimi-for-coding"]
    assert "identity" not in catalog


@pytest.mark.parametrize("models", [None, [1], [{"slug": ""}], [{"slug": "x"}, {"slug": "x"}]])
def test_native_catalog_rejects_malformed_entries(tmp_path: Path, models: object) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": models}))
    with pytest.raises(ConfigError):
        load_catalog(path)


def test_own_key_provider_cannot_opt_into_native_client_auth() -> None:
    with pytest.raises(ValidationError, match="responses_forward_client_auth"):
        ProviderCfg(
            type="passthrough",
            base_url="https://api.kimi.com/coding",
            responses_base_url="https://api.kimi.com/coding/v1",
            forward_client_auth=False,
            api_key_env="KIMI_KEY",
            responses_forward_client_auth=True,
        )


@pytest.mark.parametrize(
    "native_messages",
    [
        None,
        {
            "instructions_template": (
                "You are Codex, an agent based on GPT-6. Keep existing coding rules."
            )
        },
    ],
)
def test_custom_identity_does_not_inherit_native_gpt_identity(native_messages: object) -> None:
    base = {"models": [{"slug": "gpt-native", "model_messages": native_messages}]}
    custom = {"models": [{"slug": "k3.1-pro", "display_name": "Kimi K3.1 Pro"}]}
    combined = merge_catalogs(base, custom)
    assert combined["models"][0] == base["models"][0]
    item = combined["models"][1]
    prompt = item["model_messages"]["instructions_template"]
    assert "Kimi K3.1 Pro" in prompt and "k3.1-pro" in prompt
    assert "an agent based on GPT-6" not in prompt
    assert item["base_instructions"] == prompt
    if native_messages is not None:
        assert "Keep existing coding rules." in prompt


@pytest.mark.parametrize(
    "native_url,live",
    [
        ("https://chatgpt.com/backend-api/codex", True),
        ("https://api.openai.com/v1", False),
    ],
)
def test_native_profiles_use_live_discovery_with_explicit_static_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_url: str,
    live: bool,
) -> None:
    native_path = tmp_path / "models.json"
    native_path.write_text(json.dumps({"models": [{"slug": "gpt-native"}]}))
    config_path = tmp_path / "routing.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "anthropic": {"type": "passthrough", "base_url": "https://api.anthropic.com"},
                    "native": {
                        "type": "passthrough",
                        "base_url": native_url,
                        "responses_base_url": native_url,
                        "forward_client_auth": True,
                        "responses_forward_client_auth": True,
                    },
                },
                "rules": [],
                "default": {"provider": "anthropic"},
                "responses_default": {"provider": "native"},
                "codex_catalog_path": str(native_path),
            }
        )
    )
    monkeypatch.setenv("ROUTER_CONFIG_PATH", str(config_path))
    assert main(["--output-dir", str(tmp_path)]) == 0
    profile_path = tmp_path / "open-harness-router.config.toml"
    profile = tomllib.loads(profile_path.read_text())
    assert ("model_catalog_json" not in profile) is live
    assert profile["model_provider"] == "openai"
    assert profile["openai_base_url"] == "http://127.0.0.1:8787/v1"
    assert (tmp_path / "open-harness-router.models.json").exists()
    assert main(["--output-dir", str(tmp_path), "--static-catalog"]) == 0
    assert tomllib.loads(profile_path.read_text())["model_catalog_json"] == str(
        tmp_path / "open-harness-router.models.json"
    )
