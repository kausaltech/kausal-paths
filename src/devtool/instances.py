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


def format_instance_table(instances: list[dict[str, Any]]) -> str:
    """Render the ``editableInstances`` rows as a fixed-width table."""
    if not instances:
        return 'No editable instances.'
    rows: list[tuple[str, ...]] = []
    for inst in sorted(instances, key=lambda i: str(i.get('identifier'))):
        editor = inst.get('editor') or {}
        flags = []
        if inst.get('isLocked'):
            flags.append('locked')
        if editor.get('hasUnpublishedChanges'):
            flags.append('draft')
        published = editor.get('lastPublishedAt') or '-'
        rows.append((
            str(inst.get('identifier', '')),
            str(editor.get('configSource') or '?'),
            ' '.join(flags),
            str(published)[:10],
            str(inst.get('name', '')),
        ))
    header = ('identifier', 'source', 'state', 'published', 'name')
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header) - 1)]
    lines = ['  '.join(col.ljust(widths[i]) for i, col in enumerate(header[:-1])) + '  ' + header[-1]]
    lines.extend('  '.join(col.ljust(widths[i]) for i, col in enumerate(r[:-1])) + '  ' + r[-1] for r in rows)
    return '\n'.join(lines)
