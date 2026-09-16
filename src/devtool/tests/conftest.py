import base64
import json
import time
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from devtool.auth import SsoConfig

if TYPE_CHECKING:
    from collections.abc import Callable

OIDC_ENDPOINT = 'https://sso.example/realms/test'
CLIENT_ID = 'test-devtool'


def _b64url(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b'=').decode()


def make_id_token(**claims: Any) -> str:
    """Build an unsigned JWT-shaped token; the client only ever peeks at the payload."""
    now = int(time.time())
    payload: dict[str, Any] = {'iss': OIDC_ENDPOINT, 'sub': 'sub-1', 'email': 'dev@kausal.tech', 'iat': now, 'exp': now + 300}
    payload.update(claims)
    return f'{_b64url({"alg": "RS256", "kid": "k"})}.{_b64url(payload)}.signature'


@pytest.fixture
def sso_config() -> SsoConfig:
    return SsoConfig(oidc_endpoint=OIDC_ENDPOINT, client_id=CLIENT_ID, callback_port=0)


def oidc_discovery_document() -> dict[str, str]:
    return {
        'issuer': OIDC_ENDPOINT,
        'authorization_endpoint': f'{OIDC_ENDPOINT}/protocol/openid-connect/auth',
        'token_endpoint': f'{OIDC_ENDPOINT}/protocol/openid-connect/token',
    }


class RecordingHandler:
    """A MockTransport handler that records requests and answers by URL path suffix."""

    def __init__(self, routes: dict[str, Callable[[httpx2.Request], httpx2.Response]]) -> None:
        self.routes = routes
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        for suffix, handler in self.routes.items():
            if request.url.path.endswith(suffix):
                return handler(request)
        return httpx2.Response(404, text=f'no route for {request.url}')


@pytest.fixture
def mock_http() -> Callable[[dict[str, Callable[[httpx2.Request], httpx2.Response]]], tuple[httpx2.Client, RecordingHandler]]:
    def build(routes: dict[str, Callable[[httpx2.Request], httpx2.Response]]) -> tuple[httpx2.Client, RecordingHandler]:
        handler = RecordingHandler(routes)
        return httpx2.Client(transport=httpx2.MockTransport(handler)), handler

    return build
