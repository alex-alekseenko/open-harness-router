"""ChatGPT subscription OAuth for the openai-translate provider.

Implements the same OAuth2 PKCE flow the Codex CLI uses against
auth.openai.com, so the router holds its own credential for a ChatGPT
subscription: login once (``python -m cli.chatgpt_login``), the tokens are
stored under ``state/chatgpt_auth.json`` (0600) and refreshed
automatically. The router never reads the Codex CLI's own ``auth.json`` --
this is a separate, independently issued credential for the same account.

Request contract for the subscription backend (verified against the Codex
CLI binary and the CLIProxyAPI reference implementation):

* endpoint: ``https://chatgpt.com/backend-api/codex`` (Responses API)
* ``authorization: Bearer <access_token>``
* ``chatgpt-account-id: <chatgpt_account_id from the access token claims>``
* ``originator: codex-tui`` and the Codex CLI User-Agent

The subscription credential is only ever attached to that exact HTTPS
endpoint (``SUBSCRIPTION_ENDPOINT``): a provider configured with any other
``base_url`` is a startup error, and the auth hook re-checks the request
destination before attaching the bearer.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
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
REDIRECT_PATH = "/auth/callback"
SCOPES = "openid email profile offline_access"

ORIGINATOR = "codex-tui"
USER_AGENT = "codex-tui/0.171.0 (Mac OS 26.5.2; arm64) iTerm.app/3.6.11 (codex-tui; 0.171.0)"

# The only upstream the subscription credential may ever reach. The schema
# rejects a chatgpt_oauth provider with any other base_url, and the auth
# hook below refuses to attach the bearer to a different destination.
SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex"

# Refresh 5 minutes before expiry; tokens live ~1 hour.
_REFRESH_MARGIN_S = 300

_REQUIRED_STORE_FIELDS = ("access_token", "refresh_token", "account_id", "expires_at")


class ChatGPTAuthError(RuntimeError):
    """Login, refresh or credential-store failure."""


def _b64url(raw: bytes) -> str:
    """Base64url without padding, as the PKCE and JWT specs require."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _parse_jwt_claims(token: str) -> dict[str, Any]:
    """Decode the payload of a JWT without verifying its signature.

    The token is used only as a self-assertion of the account id; the
    access token itself is what authenticates requests upstream.
    """
    try:
        payload = token.split(".")[1]
        padding = "=" * (-len(payload) % 4)
        claims: dict[str, Any] = json.loads(base64.urlsafe_b64decode(payload + padding))
        return claims
    except (IndexError, ValueError) as exc:
        raise ChatGPTAuthError("token is not a valid JWT") from exc


def _extract_auth_fields(access_token: str, id_token: str | None) -> dict[str, str]:
    """Account id and plan from the ACCESS token's ``.../auth`` claim.

    The id_token only carries the platform org id and email; the ChatGPT
    account id required by the ``chatgpt-account-id`` header lives in the
    access token.
    """
    claims = _parse_jwt_claims(access_token)
    auth = claims.get("https://api.openai.com/auth", {})
    fields = {
        "account_id": str(auth.get("chatgpt_account_id", "")),
        "plan_type": str(auth.get("chatgpt_plan_type", "")),
    }
    if id_token:
        try:
            fields["email"] = str(_parse_jwt_claims(id_token).get("email", ""))
        except ChatGPTAuthError:
            fields["email"] = ""
    return fields


def _merge_token_response(previous: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Merge a token endpoint response into the stored credential layout.

    A refresh response may omit ``refresh_token`` and ``id_token`` (the old
    values stay valid); only ``access_token`` and ``expires_in`` are
    always present. Merging BEFORE normalization means no optional field is
    ever an unconditional lookup.
    """
    merged = dict(previous)
    merged["access_token"] = response["access_token"]
    if response.get("refresh_token"):
        merged["refresh_token"] = response["refresh_token"]
    if response.get("id_token"):
        merged["id_token"] = response["id_token"]
    merged.update(_extract_auth_fields(merged["access_token"], merged.get("id_token")))
    merged["expires_at"] = time.time() + int(response.get("expires_in", 3600))
    return merged


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
    body = response.json()
    if not body.get("access_token") or not body.get("refresh_token"):
        raise ChatGPTAuthError("token exchange response missing access_token/refresh_token")
    return _merge_token_response(
        {"refresh_token": "", "id_token": "", "account_id": "", "email": "", "plan_type": ""},
        body,
    )


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
    body = response.json()
    if not body.get("access_token"):
        raise ChatGPTAuthError("token refresh response missing access_token")
    # The refresh token itself carries the previous credential forward (an
    # omitted replacement keeps the old one valid).
    return _merge_token_response(
        {"refresh_token": refresh_token, "id_token": "", "account_id": "", "email": ""},
        body,
    )


class ChatGPTTokenStore:
    """JSON credential store with atomic writes and 0600 permissions.

    The temporary file is created with mode 0600 BEFORE any credential
    bytes are written, with a unique name per save, so no other process
    ever reads broader permissions on the file's contents, and concurrent
    saves cannot interfere.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._save_count = 0

    def load(self) -> dict[str, Any] | None:
        """Read and shape-validate the stored credential.

        Returns None when there is no usable credential; a file that
        exists but does not hold the required fields is treated as absent
        rather than aborting the registry.
        """
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text())
        except ValueError:
            return None
        if not isinstance(data, dict) or any(not data.get(f) for f in _REQUIRED_STORE_FIELDS):
            return None
        return data

    def save(self, data: dict[str, Any]) -> None:
        """Atomically replace the stored credential with 0600 permissions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._save_count += 1
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{self._save_count}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps(data, indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


class ChatGPTOAuthProvider:
    """Live credential holder: fresh access token + headers for the backend.

    Refresh is coordinated PER STORE (a module-level registry keyed by
    resolved path), so two providers sharing one token file cannot refresh
    concurrently, a re-login is picked up from disk on the next request,
    and a refresh started before a new login merges into -- never
    overwrites -- the newer on-disk generation.
    """

    _store_locks: dict[Path, asyncio.Lock] = {}

    def __init__(self, store: ChatGPTTokenStore) -> None:
        self._store = store
        self._http: httpx.Client | None = None
        self._cached: dict[str, Any] | None = None
        key = store.path.resolve()
        if key not in ChatGPTOAuthProvider._store_locks:
            ChatGPTOAuthProvider._store_locks[key] = asyncio.Lock()
        self._refresh_lock = ChatGPTOAuthProvider._store_locks[key]

    def _http_client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client()
        return self._http

    def _require_data(self) -> dict[str, Any]:
        """The current credential, preferring the on-disk generation.

        Lazy (instead of failing at router startup) so the rest of the
        router keeps working before the one-time login has been run; a
        missing/unusable credential surfaces per request on the OAuth
        route, and a re-login replaces the cached copy transparently.
        """
        data = self._store.load()
        if data is None and self._cached is not None:
            data = self._cached
        if data is None:
            raise ChatGPTAuthError(
                f"no ChatGPT subscription credential at {self._store.path}; "
                "run 'PYTHONPATH=src .venv/bin/python -m cli.chatgpt_login' first"
            )
        self._cached = data
        return data

    def _is_fresh(self, data: dict[str, Any]) -> bool:
        return float(data.get("expires_at", 0)) > time.time() + _REFRESH_MARGIN_S

    async def get_auth_headers(self) -> dict[str, str]:
        """Return Bearer + account headers, refreshing first if expired.

        The store lock is per credential file: concurrent providers and
        concurrent requests serialize their refresh, always re-reading
        the newest on-disk generation inside the critical section.
        """
        data = self._require_data()
        if self._is_fresh(data):
            return self._headers(data)

        async with self._refresh_lock:
            data = self._require_data()
            if self._is_fresh(data):
                return self._headers(data)
            previous = dict(data)
            new = await asyncio.to_thread(
                refresh_tokens, self._http_client(), previous["refresh_token"]
            )
            # Merge into the CURRENT disk generation so a re-login that
            # happened while we waited never gets overwritten: keep the
            # newer refresh token when ours was replaced meanwhile.
            current = self._store.load()
            if (
                current is not None
                and current is not data
                and current.get("refresh_token") != previous["refresh_token"]
            ):
                new["refresh_token"] = current["refresh_token"]
            self._store.save(new)
            self._cached = new
            return self._headers(new)

    def _headers(self, data: dict[str, Any]) -> dict[str, str]:
        return {
            "authorization": f"Bearer {data['access_token']}",
            "chatgpt-account-id": data.get("account_id", ""),
            "originator": ORIGINATOR,
            "user-agent": USER_AGENT,
        }

    def close(self) -> None:
        """Close the credential holder's HTTP client (provider shutdown)."""
        if self._http is not None:
            self._http.close()
            self._http = None
