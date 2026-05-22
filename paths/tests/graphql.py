"""Typed GraphQL test client for Paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from strawberry.django.test import GraphQLTestClient

from paths.const import INSTANCE_IDENTIFIER_HEADER

if TYPE_CHECKING:
    from django.test import Client
    from graphql import GraphQLFormattedError
    from strawberry.test.client import Response

    from nodes.models import InstanceConfig


class PathsTestClient(GraphQLTestClient):
    """
    Sync Strawberry test client with correct return types and instance support.

    Usage::

        client = PathsTestClient(django_client)
        client.set_instance(my_instance_config)
        data = client.query_data('{ instance { id } }')
    """

    _instance_config: InstanceConfig | None

    def __init__(self, client: Client) -> None:
        super().__init__(client, url='/v1/graphql/')
        self._instance_config = None

    def set_instance(self, ic: InstanceConfig | None) -> None:
        self._instance_config = ic

    # -- Override to fix return type (sync only) and inject instance header --

    def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        headers: dict[str, object] | None = None,
        files: dict[str, object] | None = None,
        assert_no_errors: bool | None = True,
    ) -> Response:
        resp = cast(
            'Response',
            super().query(
                query,
                variables=variables,
                headers=headers,
                files=files,
                assert_no_errors=False,
            ),
        )
        if assert_no_errors:
            if resp.errors is not None:
                from rich import print

                print(resp.errors)
            assert resp.errors is None, 'Expected no errors in response, got %s' % resp.errors
        return resp

    def query_data(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        headers: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Execute a query and return the data dict. Asserts no errors and non-null data."""
        resp = self.query(query, variables=variables, headers=headers, assert_no_errors=True)
        assert resp.data is not None, 'Expected data in response, got None'
        return resp.data

    def query_errors(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        headers: dict[str, object] | None = None,
        assert_error_message: str | None = None,
    ) -> list[GraphQLFormattedError]:
        """Execute a query that is expected to fail. Returns the errors list."""
        resp = self.query(query, variables=variables, headers=headers, assert_no_errors=False)
        assert resp.errors is not None, 'Expected errors in response, got None'
        if assert_error_message:
            assert 'message' in resp.errors[0]
            assert assert_error_message in resp.errors[0]['message'], f'Expected error message not found: {assert_error_message}'
        return resp.errors

    def request(
        self,
        body: dict[str, object],
        headers: dict[str, object] | None = None,
        files: dict[str, object] | None = None,
    ) -> Any:
        if self._instance_config is not None:
            headers = dict(headers or {})
            headers.setdefault(INSTANCE_IDENTIFIER_HEADER, self._instance_config.identifier)
        return super().request(body, headers=headers, files=files)
