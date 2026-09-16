"""InstanceExport documents on disk: naming, saving, loading, summarizing. No Django here."""

import json
from pathlib import Path
from typing import Any

from devtool.errors import DevtoolError

EXPORT_SUFFIX = '.instance-export.json'


def export_identifier(document: dict[str, Any]) -> str:
    identifier = ((document.get('instance') or {}).get('metadata') or {}).get('identifier')
    if not identifier:
        raise DevtoolError('Export document carries no instance identifier')
    return str(identifier)


def default_export_path(document: dict[str, Any], directory: Path | None = None) -> Path:
    return (directory or Path.cwd()) / f'{export_identifier(document)}{EXPORT_SUFFIX}'


def save_export_document(document: dict[str, Any], path: Path) -> None:
    """Write the document verbatim, as the backend serialized it, so the server can load it back unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


def load_export_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except OSError as e:
        raise DevtoolError(f'Cannot read {path}: {e.strerror}') from e
    except ValueError as e:
        raise DevtoolError(f'{path} is not valid JSON: {e}') from e
    if not isinstance(document, dict) or 'instance' not in document:
        raise DevtoolError(f'{path} is not an InstanceExport document')
    return document


def describe_export(document: dict[str, Any]) -> str:
    instance = document.get('instance') or {}
    lines = [
        f'instance:        {export_identifier(document)}',
        f'schema version:  {document.get("schema_version")}',
        f'nodes:           {len(instance.get("nodes") or [])}',
        f'datasets:        {len(document.get("datasets") or [])}',
        f'pages:           {len(document.get("pages") or [])}',
        f'exported at:     {document.get("exported_at") or "-"}',
        f'exported from:   {document.get("exported_from") or "-"}',
        f'draft head:      {document.get("draft_head_token") or "-"}',
    ]
    return '\n'.join(lines)
