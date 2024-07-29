import os
from pathlib import Path
from gql.transport.aiohttp import AIOHTTPTransport
from gql import Client, gql


def get_client(base_url: str | None = None):
    if base_url is None:
        base_url = 'http://127.0.0.1:8000/v1/graphql/'
    transport = AIOHTTPTransport(url=base_url, timeout=40)
    return Client(transport=transport, fetch_schema_from_transport=False)


def load_gql_query(query_name: str):
    p = Path(os.path.dirname(__file__)) / 'queries' / f"{query_name}.gql"
    data = p.read_text()
    q = gql(data)
    return q
