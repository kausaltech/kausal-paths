"""
OIDC sign-in against Kausal SSO for the devtool client.

Authorization code + PKCE as the public ``kausal-paths-devtool`` client, with
the redirect caught on a loopback port. The ID token is what the backend
accepts as a bearer (``kausal_common.auth.tokens.authenticate_devtool_id_token``).
It is short-lived, and the server additionally bounds ``iat`` freshness, so
tokens are refreshed silently and often.
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import time
import webbrowser
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import parse_qs, urlencode, urlparse

import httpx2

from devtool.errors import AuthError

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_OIDC_ENDPOINT = 'https://keycloak.kausal.tech/realms/kausal'
DEFAULT_CLIENT_ID = 'kausal-paths-devtool'
DEFAULT_CALLBACK_PORT = 8765
DEFAULT_SCOPES = 'openid email profile'
LOGIN_TIMEOUT_S = 300
# The server enforces ID-token freshness (social_core ID_TOKEN_MAX_AGE, 600 s)
# in addition to ``exp``; refresh well before either bound.
ID_TOKEN_MAX_AGE_S = 600
FRESHNESS_SLACK_S = 60


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload *without* verification: for client-side expiry checks and display only."""
    try:
        payload_b64 = token.split('.')[1]
        payload_b64 += '=' * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
    except IndexError, ValueError:
        return {}
    return claims if isinstance(claims, dict) else {}


def id_token_is_fresh(id_token: str, now: float | None = None) -> bool:
    """Whether the backend would still accept this token, with slack for the round trip."""
    claims = decode_jwt_claims(id_token)
    now = time.time() if now is None else now
    exp = claims.get('exp', 0)
    iat = claims.get('iat', 0)
    return now < exp - FRESHNESS_SLACK_S and now - iat < ID_TOKEN_MAX_AGE_S - FRESHNESS_SLACK_S


@dataclass(frozen=True)
class SsoConfig:
    oidc_endpoint: str = DEFAULT_OIDC_ENDPOINT
    client_id: str = DEFAULT_CLIENT_ID
    callback_port: int = DEFAULT_CALLBACK_PORT
    scopes: str = DEFAULT_SCOPES

    @property
    def redirect_uri(self) -> str:
        # Must be registered on the Keycloak client as-is.
        return f'http://127.0.0.1:{self.callback_port}/callback'

    @property
    def discovery_url(self) -> str:
        return f'{self.oidc_endpoint.rstrip("/")}/.well-known/openid-configuration'


@dataclass(frozen=True)
class Tokens:
    id_token: str
    refresh_token: str | None = None

    @classmethod
    def from_token_response(cls, body: dict[str, Any]) -> Tokens:
        id_token = body.get('id_token')
        if not id_token:
            raise AuthError('Token response contained no id_token (is the "openid" scope allowed for the client?)')
        return cls(id_token=id_token, refresh_token=body.get('refresh_token'))

    @property
    def claims(self) -> dict[str, Any]:
        return decode_jwt_claims(self.id_token)

    @property
    def subject_label(self) -> str:
        claims = self.claims
        return str(claims.get('email') or claims.get('sub') or '<unknown>')


class TokenCache:
    """
    File-backed token store holding one entry, keyed by (OIDC endpoint, client id).

    Plaintext in a 0600 file under ``XDG_CONFIG_HOME``: the same posture as
    ``gh`` without a keyring backend. Moving to the OS keyring is an open
    follow-up in the plan.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        return base / 'kausal-paths-devtool' / 'tokens.json'

    def load(self, config: SsoConfig) -> Tokens | None:
        try:
            data = json.loads(self.path.read_text())
        except OSError, ValueError:
            return None
        if not isinstance(data, dict):
            return None
        if data.get('oidc_endpoint') != config.oidc_endpoint or data.get('client_id') != config.client_id:
            return None
        id_token = data.get('id_token')
        if not id_token:
            return None
        return Tokens(id_token=id_token, refresh_token=data.get('refresh_token'))

    def save(self, config: SsoConfig, tokens: Tokens) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'oidc_endpoint': config.oidc_endpoint,
            'client_id': config.client_id,
            'id_token': tokens.id_token,
            'refresh_token': tokens.refresh_token,
        }
        self.path.touch(mode=0o600, exist_ok=True)
        self.path.chmod(0o600)
        self.path.write_text(json.dumps(data, indent=2))

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class AuthProvider(Protocol):
    """Anything that can put an ``Authorization`` header on a backend request."""

    def authorization_header(self) -> str | None: ...


class _CallbackServer(http.server.HTTPServer):
    result: dict[str, list[str]] | None = None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self) -> None:
        self.server.result = parse_qs(urlparse(self.path).query)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(b'<html><body><p>Signed in. You can close this tab.</p></body></html>')

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _notify_stderr(message: str) -> None:
    print(message, file=sys.stderr)


class SsoAuth:
    """
    Obtains fresh ID tokens: from cache, by silent refresh, or by browser login, in that order.

    ``notify`` receives human-readable progress messages (stderr by default).
    ``open_browser`` is injectable so a TUI can present the URL through its
    own UI, and tests can drive the flow without a browser.
    """

    def __init__(
        self,
        config: SsoConfig | None = None,
        *,
        http: httpx2.Client | None = None,
        cache: TokenCache | None = None,
        notify: Callable[[str], None] | None = None,
        open_browser: Callable[[str], bool] = webbrowser.open,
    ) -> None:
        self.config = config if config is not None else SsoConfig()
        self.http = http if http is not None else httpx2.Client(timeout=10)
        self.cache = cache if cache is not None else TokenCache()
        self.notify = notify if notify is not None else _notify_stderr
        self.open_browser = open_browser

    def authorization_header(self) -> str:
        """:class:`AuthProvider` implementation."""
        return f'Bearer {self.get_id_token()}'

    @cached_property
    def oidc_config(self) -> dict[str, Any]:
        try:
            resp = self.http.get(self.config.discovery_url)
            resp.raise_for_status()
        except httpx2.HTTPError as e:
            raise AuthError(f'OIDC discovery failed for {self.config.oidc_endpoint}: {e}') from e
        return resp.json()

    def get_id_token(self, *, force_login: bool = False) -> str:
        cached = None if force_login else self.cache.load(self.config)
        if cached is not None:
            if id_token_is_fresh(cached.id_token):
                return cached.id_token
            if cached.refresh_token:
                refreshed = self._refresh(cached.refresh_token)
                if refreshed is not None:
                    self.cache.save(self.config, refreshed)
                    return refreshed.id_token
                self.notify('Silent refresh failed; falling back to browser login.')
        return self.login().id_token

    def login(self) -> Tokens:
        """Run the browser flow unconditionally and cache the result."""
        tokens = self._run_login_flow()
        self.cache.save(self.config, tokens)
        self.notify(f'Signed in as {tokens.subject_label}.')
        return tokens

    def logout(self) -> None:
        self.cache.clear()

    def _refresh(self, refresh_token: str) -> Tokens | None:
        resp = self.http.post(
            self.oidc_config['token_endpoint'],
            data={
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.config.client_id,
                'scope': self.config.scopes,
            },
        )
        if resp.status_code != 200:
            return None
        try:
            return Tokens.from_token_response(resp.json())
        except ValueError, AuthError:
            return None

    def _run_login_flow(self) -> Tokens:
        config = self.config
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        state = secrets.token_urlsafe(16)
        auth_url = (
            self.oidc_config['authorization_endpoint']
            + '?'
            + urlencode({
                'response_type': 'code',
                'client_id': config.client_id,
                'redirect_uri': config.redirect_uri,
                'scope': config.scopes,
                'state': state,
                'code_challenge': challenge,
                'code_challenge_method': 'S256',
                'nonce': secrets.token_urlsafe(16),
            })
        )

        params = self._await_callback(auth_url)
        if 'error' in params:
            desc = params.get('error_description', ['no description'])[0]
            raise AuthError(f'Login failed: {params["error"][0]}: {desc}')
        if params.get('state', [None])[0] != state:
            raise AuthError('Login failed: state mismatch in callback.')
        code = params.get('code', [None])[0]
        if not code:
            raise AuthError('Login failed: no authorization code in callback.')

        resp = self.http.post(
            self.oidc_config['token_endpoint'],
            data={
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': config.redirect_uri,
                'client_id': config.client_id,
                'code_verifier': verifier,
            },
        )
        if resp.status_code != 200:
            raise AuthError(f'Token exchange failed ({resp.status_code}): {resp.text}')
        return Tokens.from_token_response(resp.json())

    def _await_callback(self, auth_url: str) -> dict[str, list[str]]:
        """Send the user to ``auth_url`` and return the query parameters Keycloak redirects back with."""
        config = self.config
        try:
            server = _CallbackServer(('127.0.0.1', config.callback_port), _CallbackHandler)
        except OSError as e:
            raise AuthError(
                f'Cannot listen on {config.redirect_uri} ({e.strerror}); pick another port with --port',
            ) from e
        server.timeout = 1
        self.notify(f'Opening browser for sign-in (waiting on {config.redirect_uri}) ...')
        if not self.open_browser(auth_url):
            self.notify(f'Could not open a browser; visit this URL manually:\n{auth_url}')

        deadline = time.monotonic() + LOGIN_TIMEOUT_S
        try:
            while server.result is None:
                if time.monotonic() > deadline:
                    raise AuthError('Timed out waiting for the login callback.')
                server.handle_request()
        finally:
            server.server_close()
        return server.result
