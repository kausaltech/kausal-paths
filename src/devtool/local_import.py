"""
Import an InstanceExport file into the *local* database.

This is the one place the devtool touches Django: it boots the Paths settings
with ``init_django()`` and hands the document to ``nodes.instance_import``.
Importing into a remote backend is a later addition and will go through
``PathsClient`` instead.
"""

import sys
from importlib.util import find_spec
from pathlib import Path

from devtool.errors import DevtoolError
from devtool.instances import load_export_document


def _ensure_repo_root_on_path() -> None:
    """
    Make ``kausal_common`` importable when running as the ``paths-devtool`` script.

    The package lives at the repository root, not under ``src/``, so it is on
    the path only when Python's working directory is the root (``manage.py``,
    ``python -m ...``). A console script has no such luck; the editable
    install means this file sits inside the checkout, so the root is two
    levels up from the package.
    """
    if find_spec('kausal_common') is not None:
        return
    root = Path(__file__).resolve().parents[2]
    if not (root / 'kausal_common' / '__init__.py').exists():
        raise DevtoolError(
            'Local import needs the kausal_common submodule importable; run from the repository root '
            'or initialize the submodule.',
        )
    sys.path.insert(0, str(root))


def import_export_file(
    path: Path,
    *,
    identifier: str | None = None,
    organization: str | None = None,
    name: str | None = None,
    dry_run: bool = False,
) -> str:
    """Load ``path`` into the local database and return a one-line summary."""
    _ensure_repo_root_on_path()
    from kausal_common.development.django import init_django

    init_django()

    from django.db import transaction

    from nodes.instance_import import InstanceImportError, import_instance_export
    from nodes.instance_serialization import InstanceExport

    document = load_export_document(path)
    try:
        export = InstanceExport.from_serialized_data(document)
    except ValueError as e:
        raise DevtoolError(f'{path} does not validate as an InstanceExport:\n{e}') from e

    try:
        with transaction.atomic():
            ic = import_instance_export(export, identifier=identifier, organization=organization, name=name)
            summary = (
                f'{"Would import" if dry_run else "Imported"} {len(export.instance.nodes)} nodes and '
                f'{len(export.datasets)} datasets into instance {ic.identifier!r} (uuid {ic.uuid}).'
            )
            if dry_run:
                transaction.set_rollback(True)
    except InstanceImportError as e:
        raise DevtoolError(str(e)) from e
    return summary
