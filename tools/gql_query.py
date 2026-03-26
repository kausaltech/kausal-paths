#!/usr/bin/env python3
"""
Execute GraphQL queries in-process against the Strawberry schema.

Runs queries through Django's test Client, so all middleware (including
@locale and @instance directives) is exercised. Useful for debugging
schema issues without needing a running server.

Usage:
    # Run a query from a query-store JSON file
    python tools/gql_query.py query-store/0009-gronlogik-outcomenode.json

    # Run a raw query string
    python tools/gql_query.py -i gronlogik -l sv -q '{ instance { id name } }'

    # Pipe output through jq
    python tools/gql_query.py query-store/0009-gronlogik-outcomenode.json | jq '.data.node.name'
"""

# ruff: noqa: E402
from __future__ import annotations

from kausal_common.development.django import init_django

init_django()

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from django.test import Client


GQL_URL = '/v1/graphql/'


def execute_query(
    query: str,
    variables: dict[str, Any] | None = None,
    operation_name: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a GraphQL query in-process via Django test client."""
    client = Client()
    body: dict[str, Any] = {'query': query}
    if variables:
        body['variables'] = variables
    if operation_name:
        body['operationName'] = operation_name

    kwargs: dict[str, Any] = {}
    if headers:
        # Django test client expects headers as HTTP_<UPPERCASED_HEADER_WITH_UNDERSCORES>
        for key, val in headers.items():
            django_key = 'HTTP_' + key.upper().replace('-', '_')
            kwargs[django_key] = val

    resp = client.post(GQL_URL, json.dumps(body), content_type='application/json', **kwargs)
    return json.loads(resp.content)


def execute_from_file(path: Path) -> dict[str, Any]:
    """Execute a query from a query-store JSON file."""
    with path.open() as f:
        data = json.load(f)
    return execute_query(
        query=data['query'],
        variables=data.get('variables'),
        operation_name=data.get('operation_name'),
        headers=data.get('headers'),
    )


def main():
    parser = argparse.ArgumentParser(description='Execute GraphQL queries in-process')
    parser.add_argument('file', nargs='?', help='Path to query-store JSON file')
    parser.add_argument('-i', '--instance', help='Instance identifier')
    parser.add_argument('-l', '--locale', default='en', help='Locale for @locale directive')
    parser.add_argument('-q', '--query', help='Raw GraphQL query string')
    parser.add_argument('--compact', action='store_true', help='Compact JSON output')

    args = parser.parse_args()

    if args.file:
        result = execute_from_file(Path(args.file))
    elif args.query:
        # Wrap in @locale/@instance if instance specified
        query = args.query
        variables: dict[str, Any] = {}
        headers: dict[str, str] = {}

        if args.instance:
            headers['x-paths-instance-identifier'] = args.instance
            variables['_locale'] = args.locale
            variables['_identifier'] = args.instance
            variables['_hostname'] = f'{args.instance}.localhost'
            # If query doesn't already have directives, add variable declarations
            if '@locale' not in query:
                query = (
                    f'query Q($_locale: String!, $_identifier: ID!, $_hostname: String!) '
                    f'@locale(lang: $_locale) @instance(identifier: $_identifier, hostname: $_hostname) '
                    f'{query}'
                )

        result = execute_query(query=query, variables=variables, headers=headers)
    else:
        parser.error('Provide either a query-store file or -q/--query')
        return

    indent = None if args.compact else 2
    json.dump(result, sys.stdout, indent=indent, ensure_ascii=False)
    print()


if __name__ == '__main__':
    main()
