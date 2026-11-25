"""ChatGPT subscription OAuth for the openai-translate provider.

Implements the same OAuth2 PKCE flow the Codex CLI uses against
auth.openai.com, so the router holds its own credential for a ChatGPT
subscription: login once (``python -m cli.chatgpt_login``), the tokens are
stored under ``state/chatgpt_auth.json`` (0600) and refreshed
automatically. The router never reads the Codex CLI's own ``auth.json`` --
this is a separate, independently issued credential for the same account.

Request contract for chatgpt.com/backend-api/codex (verified against the
Codex CLI binary and the CLIProxyAPI reference implementation):

* ``authorization: Bearer <access_token>``
* ``chatgpt-account-id: <chatgpt_account_id from the id_token claims>``
* ``originator: codex-tui``
* Codex CLI User-Agent.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

AUTH_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
# The public Codex CLI client id, shared by every Codex-family client.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REDIRECT_URI = "http://localhost:1455/auth/callback"
REDIRECT_PORT = 1455
SCOPES = "openid email profile offline_access"

ORIGINATOR = "codex-tui"
USER_AGENT = "codex-tui/0.171.0 (Mac OS 26.5.2; arm64) iTerm.app/3.6.11 (codex-tui; 0.171.0)"

# Refresh 5 minutes before expiry; tokens live ~1 hour.
_REFRESH_MARGIN_S = 300


class ChatGPTAuthError(RuntimeError):
    """Login, refresh or credential-store failure."""


def _b64url(raw: bytes) -> str:
    """Base64url without padding, as the PKCE and JWT specs require."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _parse_jwt_claims(id_token: str) -> dict[str, Any]:
    """Decode the payload of a JWT without verifying its signature.

    The token is used only as a self-assertion of the account id; the
    access token itself is what authenticates requests upstream.
    """
    try:
        payload = id_token.split(".")[1]
        padding = "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload + padding))
    except (IndexError, ValueError) as exc:
        raise ChatGPTAuthError("id_token is not a valid JWT") from exc


class ChatGPTTokenStore:
    """JSON credential store with atomic writes and 0600 permissions.

    Layout matches the CLIProxyAPI token file so the two tools can read
    each other's state if ever needed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text())
        except ValueError:
            raise ChatGPTAuthError(f"credential store {self.path} is corrupt")
        if data.get("access_token") and data.get("refresh_token"):
            return data
        return None

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


def _token_response_to_store(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize a token endpoint response into the stored layout.

    The account id and plan live in the ACCESS token's ``.../auth`` claim
    (``chatgpt_account_id`` / ``chatgpt_plan_type``); the id_token's claim
    only carries the platform org id. The access token is what
    authenticates requests, so that is where the header value comes from.
    """
    access_claims = _parse_jwt_claims(response["access_token"])
    auth = access_claims.get("https://api.openai.com/auth", {})
    id_claims = _parse_jwt_claims(response["id_token"])
    return {
        "access_token": response["access_token"],
        "refresh_token": response["refresh_token"],
        "id_token": response["id_token"],
        "account_id": auth.get("chatgpt_account_id", ""),
        "email": id_claims.get("email", ""),
        "plan_type": auth.get("chatgpt_plan_type", ""),
        "expires_at": time.time() + int(response.get("expires_in", 3600)),
    }


def build_authorize_url(state: str, code_challenge: str) -> str:
    """PKCE authorization URL for the browser step of the login."""
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def exchange_code(http: httpx.Client, code: str, code_verifier: str) -> dict[str, Any]:
    """Exchange the authorization code for tokens; returns the stored layout."""
    response = http.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
        },
        headers={"Accept": "application/json"},
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ChatGPTAuthError(f"token exchange failed: {response.status_code} {response.text}")
    return _token_response_to_store(response.json())


def refresh_tokens(http: httpx.Client, refresh_token: str) -> dict[str, Any]:
    """Refresh the access token; returns the stored layout (new refresh token)."""
    response = http.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
            "scope": "openid profile email",
        },
        headers={"Accept": "application/json"},
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ChatGPTAuthError(f"token refresh failed: {response.status_code} {response.text}")
    data = response.json()
    stored = _token_response_to_store(data)
    # Some refresh responses omit the refresh token; keep the old one.
    if not stored.get("refresh_token"):
        stored["refresh_token"] = refresh_token
    return stored


class ChatGPTOAuthProvider:
    """Live credential holder: fresh access token + headers for chatgpt.com.

    Created at provider build time; ``get_auth_headers`` is called before
    every upstream request, so a rotation mid-flight lands on the next
    request. Refresh is single-flight guarded.
    """

    def __init__(self, store: ChatGPTTokenStore) -> None:
        self._store = store
        self._http = httpx.Client()
        self._data: dict[str, Any] | None = store.load()
        self._lock = __import__("asyncio").Lock()

    def _require_data(self) -> dict[str, Any]:
        """The current credential, or a clear error pointing at the login CLI.

        Lazy (instead of failing at router startup) so the rest of the
        router keeps working before the one-time login has been run; the
        error surfaces per request, not as a dead router.
        """
        data = self._data if self._data is not None else self._store.load()
        if data is None:
            raise ChatGPTAuthError(
                f"no ChatGPT subscription credential at {self._store.path}; "
                "run 'PYTHONPATH=src .venv/bin/python -m cli.chatgpt_login' first"
            )
        self._data = data
        return data

    @property
    def account_id(self) -> str:
        return self._data.get("account_id", "")

    def _fresh_data(self) -> dict[str, Any] | None:
        """Prefer the on-disk credential: another login may have replaced it."""
        data = self._store.load()
        if data is not None:
            return data
        return self._data

    async def get_auth_headers(self) -> dict[str, str]:
        """Return Bearer + account headers, refreshing first if expired."""
        data = self._require_data()
        if data.get("expires_at", 0) <= time.time() + _REFRESH_MARGIN_S:
            async with self._lock:
                data = self._require_data()
                if data.get("expires_at", 0) <= time.time() + _REFRESH_MARGIN_S:
                    new = await __import__("asyncio").to_thread(
                        refresh_tokens, self._http, data["refresh_token"]
                    )
                    self._store.save(new)
                    self._data = new
                    data = new
        return {
            "authorization": f"Bearer {data['access_token']}",
            "chatgpt-account-id": data.get("account_id", ""),
            "originator": ORIGINATOR,
            "user-agent": USER_AGENT,
        }

    def close(self) -> None:
        self._http.close()
