"""GraphQL client for a Paths backend, with the transport pluggable."""

import json
from typing import TYPE_CHECKING, Any, Protocol

import httpx2

from devtool.errors import GraphQLError, TransportError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from devtool.auth import AuthProvider

GRAPHQL_PATH = '/v1/graphql/'

INSTANCE_EXPORT_QUERY = """
query DevtoolInstanceExport($id: ID!) @instance(identifier: $id) {
  instance {
    identifier
    export
  }
}
"""


class GraphQLTransport(Protocol):
    """
    Executes one GraphQL operation and returns the parsed response body.

    Two implementations are planned (Phase B of the devtool plan): this HTTP
    one for remote backends, and an in-process one going through
    ``django.test.Client`` against the same path, so both see the same
    middleware and the same ``Authorization`` handling.
    """

    def execute(
        self,
        query: str,
        variables: Mapping[str, Any] | None,
        headers: Mapping[str, str],
    ) -> dict[str, Any]: ...


class HttpTransport:
    """POSTs operations to ``<api_url>/v1/graphql/``."""

    def __init__(self, api_url: str, *, http: httpx2.Client | None = None, timeout: float = 30) -> None:
        self.endpoint = api_url.rstrip('/') + GRAPHQL_PATH
        self.http = http if http is not None else httpx2.Client()
        self.timeout = timeout

    def execute(
        self,
        query: str,
        variables: Mapping[str, Any] | None,
        headers: Mapping[str, str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {'query': query}
        if variables:
            payload['variables'] = dict(variables)
        try:
            resp = self.http.post(self.endpoint, json=payload, headers=dict(headers), timeout=self.timeout)
        except httpx2.HTTPError as e:
            raise TransportError(f'Request to {self.endpoint} failed: {e}') from e
        if resp.status_code != 200:
            raise TransportError(
                f'{self.endpoint} answered HTTP {resp.status_code}',
                status_code=resp.status_code,
                body=resp.text,
            )
        try:
            body = resp.json()
        except ValueError as e:
            raise TransportError(f'{self.endpoint} did not return JSON', status_code=200, body=resp.text) from e
        if not isinstance(body, dict):
            raise TransportError(f'{self.endpoint} returned a non-object body', status_code=200, body=resp.text)
        return body


class PathsClient:
    """
    A Paths backend as the devtool sees it: authenticated GraphQL operations.

    ``auth`` supplies the bearer for each request. ``None`` sends anonymous
    requests, which is what an in-process transport with the development
    escape hatch (``DANGEROUSLY_FORCE_AUTHENTICATED_USER``) will want.
    """

    def __init__(self, transport: GraphQLTransport, auth: AuthProvider | None = None) -> None:
        self.transport = transport
        self.auth = auth

    def execute_raw(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the full response body, ``data`` and ``errors`` both, for callers that want partial results."""
        headers: dict[str, str] = {}
        if self.auth is not None:
            authorization = self.auth.authorization_header()
            if authorization:
                headers['Authorization'] = authorization
        return self.transport.execute(query, variables, headers)

    def execute(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return ``data`` of a successful operation; raise :class:`GraphQLError` when the response carries errors."""
        body = self.execute_raw(query, variables)
        errors = body.get('errors')
        if errors:
            raise GraphQLError(errors, data=body.get('data'))
        data = body.get('data')
        if not isinstance(data, dict):
            raise TransportError('GraphQL response carries neither data nor errors', body=json.dumps(body))
        return data

    def me(self) -> dict[str, Any] | None:
        """Return the authenticated user as the backend sees it, or ``None`` if the request counted as anonymous."""
        return self.execute('{ me { email } }').get('me')

    def export_instance(self, identifier: str) -> dict[str, Any]:
        """Return the instance's InstanceExport document, exactly as the backend serialized it."""
        data = self.execute(INSTANCE_EXPORT_QUERY, {'id': identifier})
        document = (data.get('instance') or {}).get('export')
        if not isinstance(document, dict):
            raise TransportError(f'Backend returned no export document for {identifier!r}', body=json.dumps(data))
        return document
