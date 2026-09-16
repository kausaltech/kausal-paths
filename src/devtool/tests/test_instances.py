import json
from typing import TYPE_CHECKING

import httpx2
import pytest

from devtool import cli
from devtool.auth import SsoAuth
from devtool.client import HttpTransport, PathsClient
from devtool.errors import DevtoolError, TransportError
from devtool.instances import default_export_path, describe_export, load_export_document, save_export_document

if TYPE_CHECKING:
    from pathlib import Path

DOCUMENT = {
    'schema_version': 11,
    'instance': {'schema_version': 11, 'metadata': {'identifier': 'bisko', 'name': 'BISKO'}, 'nodes': [{'identifier': 'a'}]},
    'datasets': [{}, {}],
    'pages': [],
    'exported_at': '2026-09-16T12:00:00Z',
    'exported_from': 'https://paths-backend.test/',
    'draft_head_token': None,
}


def test_export_query_sends_instance_directive_and_returns_document(mock_http) -> None:
    http, handler = mock_http({
        '/v1/graphql/': lambda _r: httpx2.Response(200, json={'data': {'instance': {'identifier': 'bisko', 'export': DOCUMENT}}}),
    })
    client = PathsClient(HttpTransport('https://paths-backend.test', http=http))

    assert client.export_instance('bisko') == DOCUMENT

    body = json.loads(handler.requests[0].content)
    assert '@instance(identifier: $id)' in body['query']
    assert body['variables'] == {'id': 'bisko'}


def test_export_without_document_is_a_transport_error(mock_http) -> None:
    http, _ = mock_http({'/v1/graphql/': lambda _r: httpx2.Response(200, json={'data': {'instance': None}})})
    client = PathsClient(HttpTransport('https://paths-backend.test', http=http))

    with pytest.raises(TransportError, match='no export document'):
        client.export_instance('bisko')


def test_document_round_trips_through_disk_verbatim(tmp_path: Path) -> None:
    path = default_export_path(DOCUMENT, tmp_path)
    assert path.name == 'bisko.instance-export.json'

    save_export_document(DOCUMENT, path)

    assert load_export_document(path) == DOCUMENT
    assert json.loads(path.read_text()) == DOCUMENT
    assert 'nodes:           1' in describe_export(DOCUMENT)
    assert 'datasets:        2' in describe_export(DOCUMENT)


def test_loading_rejects_files_that_are_not_exports(tmp_path: Path) -> None:
    path = tmp_path / 'x.json'
    path.write_text('{"nope": 1}')
    with pytest.raises(DevtoolError, match='not an InstanceExport'):
        load_export_document(path)
    with pytest.raises(DevtoolError, match='Cannot read'):
        load_export_document(tmp_path / 'missing.json')


def test_export_command_saves_file(tmp_path: Path, mock_http, monkeypatch, capsys) -> None:
    http, handler = mock_http({
        '/v1/graphql/': lambda _r: httpx2.Response(200, json={'data': {'instance': {'identifier': 'bisko', 'export': DOCUMENT}}}),
    })
    monkeypatch.setattr(httpx2, 'Client', lambda **_kw: http)
    monkeypatch.setattr(SsoAuth, 'get_id_token', lambda _self, **_kw: 'tok')
    out = tmp_path / 'out.json'

    rc = cli.main(['export', 'bisko', '--api-url', 'https://paths-backend.test', '-o', str(out)])

    assert rc == 0
    assert json.loads(out.read_text()) == DOCUMENT
    assert handler.requests[0].headers['Authorization'] == 'Bearer tok'
    assert 'saved to' in capsys.readouterr().out
