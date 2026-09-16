import json

import httpx2
import pytest

from devtool.client import HttpTransport, PathsClient
from devtool.errors import GraphQLError, TransportError


class StaticAuth:
    def __init__(self, header: str | None) -> None:
        self.header = header

    def authorization_header(self) -> str | None:
        return self.header


def test_execute_sends_bearer_and_returns_data(mock_http) -> None:
    http, handler = mock_http({
        '/v1/graphql/': lambda _r: httpx2.Response(200, json={'data': {'me': {'email': 'dev@kausal.tech'}}}),
    })
    client = PathsClient(HttpTransport('https://paths-backend.test/', http=http), auth=StaticAuth('Bearer tok'))

    assert client.me() == {'email': 'dev@kausal.tech'}

    request = handler.requests[0]
    assert str(request.url) == 'https://paths-backend.test/v1/graphql/'
    assert request.headers['Authorization'] == 'Bearer tok'
    assert json.loads(request.content) == {'query': '{ me { email } }'}


def test_anonymous_client_sends_no_authorization_header(mock_http) -> None:
    http, handler = mock_http({'/v1/graphql/': lambda _r: httpx2.Response(200, json={'data': {'me': None}})})
    client = PathsClient(HttpTransport('http://127.0.0.1:8000', http=http))

    assert client.me() is None
    assert 'Authorization' not in handler.requests[0].headers


def test_variables_are_forwarded(mock_http) -> None:
    http, handler = mock_http({'/v1/graphql/': lambda _r: httpx2.Response(200, json={'data': {'ok': True}})})
    client = PathsClient(HttpTransport('http://127.0.0.1:8000', http=http))

    client.execute('query Q($id: ID!) { ok }', {'id': 'x'})

    assert json.loads(handler.requests[0].content)['variables'] == {'id': 'x'}


def test_graphql_errors_raise_with_partial_data(mock_http) -> None:
    body = {'data': {'me': None}, 'errors': [{'message': 'Not allowed'}]}
    http, _handler = mock_http({'/v1/graphql/': lambda _r: httpx2.Response(200, json=body)})
    client = PathsClient(HttpTransport('http://127.0.0.1:8000', http=http))

    with pytest.raises(GraphQLError, match='Not allowed') as excinfo:
        client.execute('{ me { email } }')
    assert excinfo.value.data == {'me': None}
    assert client.execute_raw('{ me { email } }') == body


def test_non_200_raises_transport_error(mock_http) -> None:
    http, _handler = mock_http({'/v1/graphql/': lambda _r: httpx2.Response(502, text='bad gateway')})
    client = PathsClient(HttpTransport('http://127.0.0.1:8000', http=http))

    with pytest.raises(TransportError) as excinfo:
        client.execute('{ me { email } }')
    assert excinfo.value.status_code == 502
    assert excinfo.value.body == 'bad gateway'
