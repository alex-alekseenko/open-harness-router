"""Unit tests for the ChatGPT subscription OAuth provider pieces."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from auth.chatgpt_oauth import (
    ChatGPTOAuthProvider,
    ChatGPTTokenStore,
    _b64url,
    _parse_jwt_claims,
    _token_response_to_store,
    build_authorize_url,
)


def _fake_jwt(claims: dict) -> str:
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps(claims).encode())
    return f"{header}.{payload}.{_b64url(b'signature')}"


def _token_response() -> dict:
    return {
        "access_token": _fake_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-123", "chatgpt_plan_type": "pro"}}
        ),
        "refresh_token": "rt-xyz",
        "id_token": _fake_jwt({"email": "user@example.com"}),
        "expires_in": 3600,
    }


class TestTokenStore:
    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ChatGPTTokenStore(tmp_path / "auth.json")
        assert store.load() is None
        store.save({"access_token": "a", "refresh_token": "r"})
        assert store.load() == {"access_token": "a", "refresh_token": "r"}

    def test_permissions(self, tmp_path: Path) -> None:
        store = ChatGPTTokenStore(tmp_path / "auth.json")
        store.save({"access_token": "a", "refresh_token": "r"})
        assert (tmp_path / "auth.json").stat().st_mode & 0o777 == 0o600

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text("not json")
        store = ChatGPTTokenStore(path)
        with pytest.raises(Exception, match="corrupt"):
            store.load()


class TestTokenResponse:
    def test_account_id_from_access_token(self) -> None:
        data = _token_response_to_store(_token_response())
        assert data["account_id"] == "acc-123"
        assert data["plan_type"] == "pro"
        assert data["email"] == "user@example.com"
        assert data["expires_at"] > time.time()


class TestAuthorizeUrl:
    def test_params(self) -> None:
        url = build_authorize_url("state-1", "challenge-1")
        assert "client_id=app_EMoamEEZ73f0CkXaXp7hrann" in url
        assert "code_challenge=challenge-1" in url
        assert "code_challenge_method=S256" in url
        assert "state=state-1" in url


class TestProviderHeaders:
    @pytest.mark.asyncio
    async def test_missing_credential_per_request_error(self, tmp_path: Path) -> None:
        provider = ChatGPTOAuthProvider(ChatGPTTokenStore(tmp_path / "auth.json"))
        with pytest.raises(Exception, match="cli.chatgpt_login"):
            await provider.get_auth_headers()

    @pytest.mark.asyncio
    async def test_headers_from_stored_credential(self, tmp_path: Path) -> None:
        store = ChatGPTTokenStore(tmp_path / "auth.json")
        data = _token_response_to_store(_token_response())
        data["expires_at"] = time.time() + 3600  # fresh, no refresh attempted
        store.save(data)
        provider = ChatGPTOAuthProvider(store)
        headers = await provider.get_auth_headers()
        assert headers["authorization"].startswith("Bearer ")
        assert headers["chatgpt-account-id"] == "acc-123"
        assert headers["originator"] == "codex-tui"


class TestSchemaValidation:
    def test_flag_requires_responses_flavor(self) -> None:
        from routing.schema import ProviderCfg

        with pytest.raises(Exception, match="api_flavor='responses'"):
            ProviderCfg(
                type="openai-translate",
                base_url="https://chatgpt.com/backend-api/codex",
                api_flavor="chat",
                chatgpt_oauth=True,
            )

    def test_flag_excludes_api_key_env(self) -> None:
        from routing.schema import ProviderCfg

        with pytest.raises(Exception, match="api_key_env"):
            ProviderCfg(
                type="openai-translate",
                base_url="https://chatgpt.com/backend-api/codex",
                api_flavor="responses",
                chatgpt_oauth=True,
                api_key_env="KEY",
                max_tokens_limit=128000,
                context_window=1048576,
            )

    def test_valid_flag_shape(self) -> None:
        from routing.schema import ProviderCfg

        cfg = ProviderCfg(
            type="openai-translate",
            base_url="https://chatgpt.com/backend-api/codex",
            api_flavor="responses",
            chatgpt_oauth=True,
            max_tokens_limit=128000,
            context_window=1048576,
        )
        assert cfg.chatgpt_oauth is True