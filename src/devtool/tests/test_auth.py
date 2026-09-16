import stat
import time
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest

from devtool.auth import SsoAuth, SsoConfig, TokenCache, Tokens, id_token_is_fresh
from devtool.errors import AuthError
from devtool.tests.conftest import CLIENT_ID, make_id_token, oidc_discovery_document

if TYPE_CHECKING:
    from pathlib import Path


def test_freshness_respects_both_exp_and_iat() -> None:
    now = int(time.time())
    assert id_token_is_fresh(make_id_token(iat=now, exp=now + 300))
    assert not id_token_is_fresh(make_id_token(iat=now - 3600, exp=now + 300)), 'server-side iat bound'
    assert not id_token_is_fresh(make_id_token(iat=now, exp=now + 30)), 'expires within the slack window'
    assert not id_token_is_fresh('not-a-jwt')


def test_token_cache_round_trip_is_private_and_keyed(tmp_path: Path, sso_config: SsoConfig) -> None:
    cache = TokenCache(tmp_path / 'tokens.json')
    assert cache.load(sso_config) is None

    tokens = Tokens(id_token=make_id_token(), refresh_token='r1')
    cache.save(sso_config, tokens)

    assert cache.load(sso_config) == tokens
    assert stat.S_IMODE(cache.path.stat().st_mode) == 0o600
    other_client = SsoConfig(oidc_endpoint=sso_config.oidc_endpoint, client_id='someone-else')
    assert cache.load(other_client) is None, 'tokens for one client must not leak to another'

    cache.clear()
    assert cache.load(sso_config) is None


def test_fresh_cached_token_needs_no_network(tmp_path: Path, sso_config: SsoConfig, mock_http) -> None:
    http, handler = mock_http({})
    cache = TokenCache(tmp_path / 'tokens.json')
    cache.save(sso_config, Tokens(id_token=make_id_token()))
    auth = SsoAuth(sso_config, http=http, cache=cache)

    token = auth.get_id_token()

    assert token == cache.load(sso_config).id_token  # type: ignore[union-attr]
    assert handler.requests == []
    assert auth.authorization_header() == f'Bearer {token}'


def test_stale_token_is_refreshed_silently(tmp_path: Path, sso_config: SsoConfig, mock_http) -> None:
    now = int(time.time())
    fresh = make_id_token()
    http, handler = mock_http({
        '/openid-configuration': lambda _r: httpx2.Response(200, json=oidc_discovery_document()),
        '/token': lambda _r: httpx2.Response(200, json={'id_token': fresh, 'refresh_token': 'r2'}),
    })
    cache = TokenCache(tmp_path / 'tokens.json')
    cache.save(sso_config, Tokens(id_token=make_id_token(iat=now - 3000, exp=now + 300), refresh_token='r1'))
    auth = SsoAuth(sso_config, http=http, cache=cache)

    assert auth.get_id_token() == fresh

    refresh_request = next(r for r in handler.requests if r.url.path.endswith('/token'))
    form = parse_qs(refresh_request.content.decode())
    assert form['grant_type'] == ['refresh_token']
    assert form['refresh_token'] == ['r1']
    assert form['client_id'] == [CLIENT_ID]
    assert cache.load(sso_config) == Tokens(id_token=fresh, refresh_token='r2')


def test_failed_refresh_falls_back_to_login(tmp_path: Path, sso_config: SsoConfig, mock_http, monkeypatch) -> None:
    now = int(time.time())
    http, _handler = mock_http({
        '/openid-configuration': lambda _r: httpx2.Response(200, json=oidc_discovery_document()),
        '/token': lambda _r: httpx2.Response(400, json={'error': 'invalid_grant'}),
    })
    cache = TokenCache(tmp_path / 'tokens.json')
    cache.save(sso_config, Tokens(id_token=make_id_token(iat=now - 3000), refresh_token='expired'))
    messages: list[str] = []
    auth = SsoAuth(sso_config, http=http, cache=cache, notify=messages.append)
    from_login = Tokens(id_token=make_id_token(), refresh_token='r-new')
    monkeypatch.setattr(auth, '_run_login_flow', lambda: from_login)

    assert auth.get_id_token() == from_login.id_token

    assert any('falling back to browser login' in m for m in messages)
    assert cache.load(sso_config) == from_login


def test_login_flow_uses_pkce_and_exchanges_the_code(tmp_path: Path, sso_config: SsoConfig, mock_http, monkeypatch) -> None:
    id_token = make_id_token()
    http, handler = mock_http({
        '/openid-configuration': lambda _r: httpx2.Response(200, json=oidc_discovery_document()),
        '/token': lambda _r: httpx2.Response(200, json={'id_token': id_token, 'refresh_token': 'r1'}),
    })
    auth = SsoAuth(sso_config, http=http, cache=TokenCache(tmp_path / 'tokens.json'), notify=lambda _m: None)
    seen_auth_urls: list[str] = []

    def fake_callback(auth_url: str) -> dict[str, list[str]]:
        seen_auth_urls.append(auth_url)
        state = parse_qs(urlparse(auth_url).query)['state'][0]
        return {'code': ['the-code'], 'state': [state]}

    monkeypatch.setattr(auth, '_await_callback', fake_callback)

    tokens = auth.login()

    assert tokens == Tokens(id_token=id_token, refresh_token='r1')
    auth_params = parse_qs(urlparse(seen_auth_urls[0]).query)
    assert auth_params['code_challenge_method'] == ['S256']
    assert auth_params['redirect_uri'] == [sso_config.redirect_uri]
    assert auth_params['client_id'] == [CLIENT_ID]
    exchange = parse_qs(handler.requests[-1].content.decode())
    assert exchange['grant_type'] == ['authorization_code']
    assert exchange['code'] == ['the-code']
    assert 'code_verifier' in exchange


def test_state_mismatch_is_refused(tmp_path: Path, sso_config: SsoConfig, mock_http, monkeypatch) -> None:
    http, handler = mock_http({
        '/openid-configuration': lambda _r: httpx2.Response(200, json=oidc_discovery_document()),
    })
    auth = SsoAuth(sso_config, http=http, cache=TokenCache(tmp_path / 'tokens.json'), notify=lambda _m: None)
    monkeypatch.setattr(auth, '_await_callback', lambda _url: {'code': ['c'], 'state': ['forged']})

    with pytest.raises(AuthError, match='state mismatch'):
        auth.login()
    assert not any(r.url.path.endswith('/token') for r in handler.requests), 'no code exchange after a bad state'
