"""
Copy a model instance (spec, nodes, datasets) and its Wagtail content.

Produces a self-contained, DB-backed copy under a new identifier:

    python manage.py copy_instance zuerich zuerich-copy \
        --site-url https://zuerich-copy.paths.kausal.dev/

Two copy representations, selected with ``--mode`` (default ``auto`` follows
the source's ``config_source``):

  db   — ``export_instance`` → ``import_instance`` into a fresh
         ``config_source='database'`` InstanceConfig, copying spec, nodes,
         edges, dimensions, datasets and datapoints. Self-contained snapshot
         of the source's *current DB state*; the right choice for instances
         that are already database-backed.
  yaml — copy ``configs/<src>.yaml`` → ``configs/<dst>.yaml`` (rewriting only
         the instance id / name), create a ``config_source='yaml'``
         InstanceConfig, then materialise its NodeConfig rows — plus the editor
         graph (NodeEdge/DatasetPort) — from the source's DB snapshot so
         admin-authored fields are carried (falling back to ``sync_nodes()``
         when the source has no DB spec). Preserves full YAML fidelity for
         instances whose features are not (yet) fully expressible in the DB
         spec. The model definition is read from the file, so DVC datasets ride
         along by reference; only DB-resident (admin-edited) datasets are copied
         into the DB. The DB rows come from the source's mirror (last
         ``sync_instance_to_db``), which can lag the YAML.

Both modes then deep-copy the source's Wagtail page subtree (including draft
revisions) and InstanceSiteContent, create an InstanceHostname route, and repoint every node
reference — ``OutcomePage.outcome_node`` plus ``NodeChooserBlock`` PKs inside
StreamField bodies — from the source's NodeConfig rows to the copy's, on both
the live page rows and every copied revision.

Notes:
  * In db mode, datasets from a DVC ``dataset_repo`` are exported as external
    placeholders, so the copy references the same repo rather than duplicating
    the parquet.
  * In yaml mode the new file is written to ``configs/`` (a filesystem side
    effect outside the DB transaction); it is removed again on dry-run or
    failure. Framework-backed instances cannot be yaml-copied (their YAML is
    the shared framework file) — use ``--mode db``.
  * Translated page subtrees in non-primary locales are NOT copied (Paths only
    authors primary-language pages today).

"""

import contextlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from wagtail.blocks import ListBlock, StreamBlock, StructBlock
from wagtail.fields import StreamField
from wagtail.models import Page

from loguru import logger

from nodes.blocks import NodeChooserBlock
from nodes.instance_serialization import (
    export_instance,
    import_instance,
    import_instance_datasets,
    import_instance_edges_and_ports,
    import_instance_nodes,
)
from nodes.models import InstanceConfig, InstanceHostname, NodeConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Iterator


class DryRunRollbackError(Exception):
    """Raised to roll back the transaction at the end of a dry run."""


def rewrite_instance_yaml(data: Any, *, dest_id: str, name: str | None = None, name_suffix: str = ' (copy)') -> Any:
    """
    Rewrite a loaded instance-YAML mapping in place for a copy.

    Only the top-level instance identity is changed: ``id`` and the
    ``name``/``name_*`` fields. When ``name`` is given it overwrites every
    name field (so the runtime/model name matches the DB row); otherwise
    ``name_suffix`` is appended to each. Dataset paths (``default_path``,
    ``emission_dataset``, ``zuerich/...`` dataset ids) and code references
    (``ch.zuerich....``) are deliberately left alone — they point at the
    shared DVC data and region code, not at the instance id.
    """
    data['id'] = dest_id
    data.pop('site_url', None)
    for key in list(data.keys()):
        if key == 'name' or key.startswith('name_'):
            data[key] = name if name is not None else f'{data[key]}{name_suffix}'
    return data


def _read_yaml_identity(yaml_path: Path) -> str | None:
    """
    Read the primary ``name`` from a committed instance YAML.

    Used by ``--use-existing-yaml`` to default the DB row from the
    already-deployed file when the CLI omits ``--name``.

    Most configs carry the name only as per-language ``name_<lang>`` fields (no
    bare ``name``), and ``default_language`` may be a regional code (``de-CH``)
    while the suffix is the short form (``name_de``). So the name is resolved as:
    ``name`` → ``name_<default_language>`` → ``name_<short default_language>`` →
    the first ``name_*`` in file order.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    with yaml_path.open() as f:
        data = yaml.load(f)

    name = data.get('name')
    if not name:
        candidates: list[str] = []
        default_lang = data.get('default_language')
        if default_lang:
            candidates.append(f'name_{default_lang}')
            candidates.append(f'name_{str(default_lang).split("-")[0]}')
        candidates.extend(k for k in data if k.startswith('name_'))
        for key in candidates:
            if data.get(key):
                name = data[key]
                break

    return str(name) if name else None


def rewrite_include_paths(data: Any, *, src_id: str, dest_id: str) -> list[tuple[str, str]]:
    """
    Repoint per-instance ``include`` files from ``<src_id>/…`` to ``<dest_id>/…``.

    Instances keep their node-group fragments in a ``configs/<id>/`` subdirectory
    (e.g. ``include: - file: zuerich/buildings.yaml``). Those are editable model
    *source*, so a copy gets its own duplicates rather than sharing the source's.
    Mutates ``data['include']`` in place and returns the list of
    ``(old_relpath, new_relpath)`` fragment files the caller must copy. Includes
    that don't live under ``<src_id>/`` (shared library fragments) are left
    untouched and stay shared.
    """
    copies: list[tuple[str, str]] = []
    prefix = f'{src_id}/'
    for entry in data.get('include') or []:
        if not isinstance(entry, dict):
            continue
        file = entry.get('file')
        if isinstance(file, str) and file.startswith(prefix):
            new_file = f'{dest_id}/{file[len(prefix) :]}'
            entry['file'] = new_file
            copies.append((file, new_file))
    return copies


def _remap_raw(block: Any, raw: Any, node_map: dict[int, int]) -> Any:  # noqa: C901, PLR0912
    """
    Walk a block's serialized (raw) value, remapping NodeConfig PKs.

    ``node_map`` maps source NodeConfig pk to copy NodeConfig pk; unknown
    pks are left untouched.
    """
    if isinstance(block, NodeChooserBlock):
        if raw is None or raw == '':
            return raw
        try:
            old_pk = int(raw)
        except TypeError, ValueError:
            return raw
        return node_map.get(old_pk, raw)

    if isinstance(block, StructBlock):
        if not isinstance(raw, dict):
            return raw
        struct_out = dict(raw)
        for name, child in block.child_blocks.items():
            if name in struct_out:
                struct_out[name] = _remap_raw(child, struct_out[name], node_map)
        return struct_out

    if isinstance(block, StreamBlock):
        if not isinstance(raw, list):
            return raw
        stream_out: list[Any] = []
        for item in raw:
            new_item = dict(item)
            child = block.child_blocks.get(item.get('type'))
            if child is not None:
                new_item['value'] = _remap_raw(child, item.get('value'), node_map)
            stream_out.append(new_item)
        return stream_out

    if isinstance(block, ListBlock):
        if not isinstance(raw, list):
            return raw
        list_out: list[Any] = []
        for item in raw:
            # New-format list items are {'type': 'item', 'value': ..., 'id': ...}.
            if isinstance(item, dict) and 'value' in item:
                new_item = dict(item)
                new_item['value'] = _remap_raw(block.child_block, item['value'], node_map)
                list_out.append(new_item)
            else:
                list_out.append(_remap_raw(block.child_block, item, node_map))
        return list_out

    return raw


def _stream_fields(page: Page) -> list[StreamField]:
    return [f for f in page._meta.get_fields() if isinstance(f, StreamField)]


def _parse_revision_stream(raw_json: Any, *, page: Page, revision_pk: Any, field_name: str) -> Any:
    """
    Parse a StreamField value from revision content, tolerating bad data.

    Revision content stores StreamFields as JSON strings. A legacy/corrupt
    revision with invalid JSON is skipped (returns ``None``) with a warning
    rather than aborting the whole copy.
    """
    if not isinstance(raw_json, str):
        return raw_json
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning(
            'copy_instance: skipping malformed StreamField {!r} in revision {} of page {}',
            field_name,
            revision_pk,
            page.pk,
        )
        return None


def remap_page_live(page: Page, node_map: dict[int, int]) -> bool:
    """Remap node references on a (specific) page's live row. Returns True if changed."""
    from pages.models import OutcomePage

    changed_fields: list[str] = []

    if isinstance(page, OutcomePage) and page.outcome_node_id in node_map:
        page.outcome_node_id = node_map[page.outcome_node_id]
        changed_fields.append('outcome_node')

    for field in _stream_fields(page):
        sv = getattr(page, field.name)
        # ``get_prep_value()`` yields a plain list (the same form stored in
        # revision content), whereas ``raw_data`` is a RawDataView that
        # ``_remap_raw`` would not recurse into.
        raw = sv.get_prep_value()
        new_raw = _remap_raw(field.stream_block, raw, node_map)
        if new_raw != raw:
            setattr(page, field.name, field.stream_block.to_python(new_raw))
            changed_fields.append(field.name)

    if not changed_fields:
        return False
    page.save(update_fields=changed_fields)
    return True


def remap_page_revisions(page: Page, node_map: dict[int, int]) -> int:
    """
    Remap node references inside every revision of a (specific) page.

    Revision content stores ``outcome_node`` as an int pk and each StreamField
    as a JSON string. We rewrite both in place so drafts, previews, and
    rollbacks can never point back at the source instance's nodes.
    """
    stream_field_names = {f.name: f.stream_block for f in _stream_fields(page)}
    n_changed = 0
    for revision in page.revisions.all():
        content = revision.content
        changed = False

        old_pk = content.get('outcome_node')
        if isinstance(old_pk, int) and old_pk in node_map:
            content['outcome_node'] = node_map[old_pk]
            changed = True

        for name, stream_block in stream_field_names.items():
            raw_json = content.get(name)
            if not raw_json:
                continue
            raw = _parse_revision_stream(raw_json, page=page, revision_pk=revision.pk, field_name=name)
            if raw is None:
                continue
            new_raw = _remap_raw(stream_block, raw, node_map)
            if new_raw != raw:
                content[name] = json.dumps(new_raw) if isinstance(raw_json, str) else new_raw
                changed = True

        if changed:
            revision.content = content
            revision.save(update_fields=['content'])
            n_changed += 1
    return n_changed


def _iter_node_pks(block: Any, raw: Any) -> Iterator[int]:  # noqa: C901
    """Yield NodeConfig pks referenced by NodeChooser leaves in a raw value."""
    if isinstance(block, NodeChooserBlock):
        if raw not in (None, ''):
            with contextlib.suppress(TypeError, ValueError):
                yield int(raw)
    elif isinstance(block, StructBlock) and isinstance(raw, dict):
        for name, child in block.child_blocks.items():
            if name in raw:
                yield from _iter_node_pks(child, raw[name])
    elif isinstance(block, StreamBlock) and isinstance(raw, list):
        for item in raw:
            child = block.child_blocks.get(item.get('type'))
            if child is not None:
                yield from _iter_node_pks(child, item.get('value'))
    elif isinstance(block, ListBlock) and isinstance(raw, list):
        for item in raw:
            value = item['value'] if isinstance(item, dict) and 'value' in item else item
            yield from _iter_node_pks(block.child_block, value)


def find_source_node_refs(page: Page, source_pks: set[int]) -> list[str]:
    """
    Return human-readable locations where a (specific) page still references source nodes.

    Scans the live row and every revision — ``outcome_node`` and StreamField
    bodies. Used to fail loudly if remapping left a dangling reference (e.g. an
    unmapped node, or a stale page pointing at a removed source node).
    """
    from pages.models import OutcomePage

    hits: list[str] = []
    if isinstance(page, OutcomePage) and page.outcome_node_id in source_pks:
        hits.append(f'{page.slug}: outcome_node')

    stream_fields = {f.name: f.stream_block for f in _stream_fields(page)}
    for name, stream_block in stream_fields.items():
        raw = getattr(page, name).get_prep_value()
        if any(pk in source_pks for pk in _iter_node_pks(stream_block, raw)):
            hits.append(f'{page.slug}: {name}')

    for revision in page.revisions.all():
        content = revision.content
        if isinstance(content.get('outcome_node'), int) and content['outcome_node'] in source_pks:
            hits.append(f'{page.slug} rev {revision.pk}: outcome_node')
        for name, stream_block in stream_fields.items():
            raw_json = content.get(name)
            if not raw_json:
                continue
            raw = _parse_revision_stream(raw_json, page=page, revision_pk=revision.pk, field_name=name)
            if raw is None:
                continue
            if any(pk in source_pks for pk in _iter_node_pks(stream_block, raw)):
                hits.append(f'{page.slug} rev {revision.pk}: {name}')
    return hits


class Command(BaseCommand):
    help = 'Copy an instance (spec, nodes, datasets) and its Wagtail content under a new identifier'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('source', type=str, help='Identifier of the instance to copy')
        parser.add_argument('dest', type=str, help="Identifier for the copy (e.g. 'zuerich-copy')")
        parser.add_argument('--name', type=str, help='Display name for the copy (default: "<source name> (copy)")')
        parser.add_argument(
            '--site-url',
            type=str,
            help='Public URL for the copy. Used to create an InstanceHostname route.',
        )
        parser.add_argument(
            '--mode',
            choices=['auto', 'db', 'yaml'],
            default='auto',
            help=(
                "Copy representation. 'auto' (default) follows the source's config_source; "
                + "'db' is a self-contained export/import; 'yaml' copies configs/<src>.yaml."
            ),
        )
        parser.add_argument(
            '--no-pages',
            action='store_true',
            help='Skip copying all Wagtail content (the page tree and instance site content)',
        )
        parser.add_argument(
            '--allow-dangling-refs',
            action='store_true',
            help='Do not fail if copied pages/revisions still reference source-instance nodes after remapping.',
        )
        parser.add_argument(
            '--sync-source',
            action='store_true',
            help="(db mode) (Re)sync the source's DB mirror from YAML before exporting. Mutates it (not reverted).",
        )
        parser.add_argument('--dry-run', action='store_true', help='Do everything, then roll back without saving')

        # Split the yaml-mode operation across environments: write the config in
        # one place (commit it), apply the DB side in another (e.g. production).
        yaml_stage = parser.add_mutually_exclusive_group()
        yaml_stage.add_argument(
            '--write-config-only',
            action='store_true',
            help=(
                '(yaml mode) Only write configs/<dst>.yaml and exit; make no DB changes. '
                + 'Commit + deploy the file, then run with --use-existing-yaml.'
            ),
        )
        yaml_stage.add_argument(
            '--use-existing-yaml',
            action='store_true',
            help='(yaml mode) Do not write the config; apply the DB side from an already-committed configs/<dst>.yaml.',
        )

    def handle(self, *args, **options) -> None:
        source: str = options['source']
        dest: str = options['dest']
        dry_run: bool = options['dry_run']

        if InstanceConfig.objects.filter(identifier=dest).exists():
            raise CommandError(f"An instance with identifier '{dest}' already exists.")
        try:
            ic_src = InstanceConfig.objects.get(identifier=source)
        except InstanceConfig.DoesNotExist:
            raise CommandError(f"Source instance '{source}' not found.") from None

        self._created_files: list[Path] = []
        try:
            with transaction.atomic():
                self._copy(ic_src, dest, options)
                if dry_run:
                    raise DryRunRollbackError  # noqa: TRY301
        except DryRunRollbackError:
            self._discard_created_files()
            self.stdout.write(self.style.WARNING('Dry run complete — rolled back, nothing saved.'))
            return
        except Exception:
            self._discard_created_files()
            raise

        if options['write_config_only']:
            self.stdout.write(
                self.style.SUCCESS(f"Wrote config for '{dest}'. Commit + deploy it, then run with --use-existing-yaml."),
            )
        else:
            self.stdout.write(self.style.SUCCESS(f"Copied '{source}' → '{dest}'."))

    def _discard_created_files(self) -> None:
        parents: set[Path] = set()
        for path in self._created_files:
            path.unlink(missing_ok=True)
            parents.add(path.parent)
        # Remove any now-empty directory we created (e.g. the copied configs/<dest>/).
        for parent in parents:
            with contextlib.suppress(OSError):
                parent.rmdir()

    def _copy(self, ic_src: InstanceConfig, dest: str, options: dict[str, Any]) -> None:
        mode = options['mode']
        if mode == 'auto':
            mode = 'yaml' if ic_src.config_source == 'yaml' else 'db'

        if (options['write_config_only'] or options['use_existing_yaml']) and mode != 'yaml':
            raise CommandError('--write-config-only / --use-existing-yaml are only valid in yaml mode (use --mode yaml).')
        if options['write_config_only'] and options['dry_run']:
            raise CommandError('--write-config-only writes a file to commit; it cannot be combined with --dry-run.')

        self.stdout.write(f'Copy mode: {mode}')
        if mode == 'yaml':
            self._copy_yaml(ic_src, dest, options)
        else:
            self._copy_db(ic_src, dest, options)

    def _copy_db(self, ic_src: InstanceConfig, dest: str, options: dict[str, Any]) -> None:
        # 1. The export reads the source's DB mirror; make sure one exists.
        if options['sync_source']:
            from nodes.spec_export import sync_instance_to_db

            self.stdout.write(self.style.WARNING(f"Syncing source '{ic_src.identifier}' DB mirror from YAML…"))
            sync_instance_to_db(ic_src.identifier)
            ic_src.refresh_from_db()
        elif not ic_src.nodes.exists():
            sid = ic_src.identifier
            msg = (
                f"Source '{sid}' has no DB-synced model (0 node rows). "
                + f'Run `python manage.py sync_instance_to_db {sid}` first, '
                + "or pass --sync-source (which mutates the source's DB mirror)."
            )
            raise CommandError(msg)

        # 2. Export + import into a fresh InstanceConfig.
        self.stdout.write('Exporting source…')
        export = export_instance(ic_src)

        name = options['name'] or f'{ic_src.get_name()} (copy)'
        ic_copy = InstanceConfig(
            identifier=dest,
            name=name,
            organization=ic_src.organization,
            primary_language=ic_src.primary_language,
            other_languages=list(ic_src.other_languages or []),
            config_source='database',
        )
        ic_copy.save()
        self.stdout.write(f'Created InstanceConfig {dest!r}; importing model…')
        import_instance(ic_copy, export)
        ic_copy.refresh_from_db()
        self.stdout.write(f'  imported {ic_copy.nodes.count()} nodes, {len(export.datasets)} datasets.')

        # 3. Point the copy and its nodes at the source via copy_of.
        self._set_copy_of(ic_src, ic_copy)

        # 4. Copy the Wagtail content.
        if not options['no_pages']:
            self._copy_pages(ic_src, ic_copy, options['site_url'], allow_dangling=options['allow_dangling_refs'])
            self._copy_site_content(ic_src, ic_copy)

    def _copy_yaml(self, ic_src: InstanceConfig, dest: str, options: dict[str, Any]) -> None:
        from django.conf import settings

        # Framework-backed instances have no standalone YAML (it's the shared framework
        # file), so they can't be yaml-copied — reject before either staged path, not
        # just on the write path.
        if ic_src.has_framework_config():
            raise CommandError(
                f"Source '{ic_src.identifier}' is framework-backed (its YAML is the shared framework file); " + 'use --mode db.',
            )

        dst_yaml = Path(settings.BASE_DIR, 'configs', f'{dest}.yaml')

        # Stage 1 — produce configs/<dst>.yaml, unless we are reusing a committed one.
        if options['use_existing_yaml']:
            if not dst_yaml.exists():
                raise CommandError(
                    f"--use-existing-yaml given but '{dst_yaml}' was not found. Commit and deploy it first.",
                )
            self.stdout.write(f'  using existing {dst_yaml.relative_to(settings.BASE_DIR)}')
        else:
            self._write_copy_config(ic_src, dest, dst_yaml, options)
            if options['write_config_only']:
                # The DB side runs later (e.g. in production) via --use-existing-yaml.
                return

        # Stage 2 — DB side: InstanceConfig + datasets + nodes + Wagtail content.
        # When applying an already-committed config (--use-existing-yaml), fall back
        # to the file's own name for any value the CLI omits.
        site_url = options['site_url']
        name = options['name']
        if options['use_existing_yaml']:
            name = name or _read_yaml_identity(dst_yaml)
        name = name or f'{ic_src.get_name()} (copy)'
        ic_copy = InstanceConfig(
            identifier=dest,
            name=name,
            organization=ic_src.organization,
            primary_language=ic_src.primary_language,
            other_languages=list(ic_src.other_languages or []),
            config_source='yaml',
        )
        ic_copy.save()

        # Materialise the model from the snapshot rather than rebuilding it from
        # the YAML, so admin-authored node fields (description, goal, …) the YAML
        # can't express are carried over. The snapshot is identifier-keyed, so
        # node references need no pk remapping. config_source stays 'yaml': the
        # runtime still loads the copied YAML; these rows are the admin/page
        # mirror (node spec is dormant for yaml-backed instances). DB-resident
        # (admin-edited) datasets come from the snapshot too; DVC datasets ride
        # along by reference in the YAML and are not copied.
        if ic_src.spec is not None:
            export = export_instance(ic_src)
            nodes_by_id = import_instance_nodes(ic_copy, export)
            db_datasets = [d for d in export.datasets if not d.is_external_placeholder and d.data is not None]
            datasets_by_id: dict[str, Any] = {}
            if db_datasets:
                imported = import_instance_datasets(ic_copy, db_datasets, create_missing_dimensions=True)
                datasets_by_id = {ds.identifier: ds for ds in imported if ds.identifier is not None}
                self.stdout.write(f'  copied {len(db_datasets)} DB-resident dataset(s).')
            # Recreate the editor graph (NodeEdge/DatasetPort) so the copy's DB
            # mirror matches the source's — not just its node rows. Dormant for
            # config_source='yaml' (the runtime loads the YAML), but the Trailhead
            # editor reads them.
            import_instance_edges_and_ports(ic_copy, export, nodes_by_id, datasets_by_id)
            self.stdout.write(f'  imported {ic_copy.nodes.count()} node rows (+ edges/ports) from snapshot.')
            self.stdout.write(
                self.style.WARNING(
                    f'  note: node rows/edges/ports come from the {ic_src.identifier} DB mirror, which '
                    + 'reflects its last sync_instance_to_db and can lag the YAML. The copied YAML governs '
                    + 'the runtime; re-sync the source first if its mirror is stale.',
                ),
            )
        else:
            # No DB mirror to snapshot — fall back to building rows from the YAML.
            ic_copy.sync_nodes()
            self.stdout.write(
                self.style.WARNING(f'  source has no DB spec; synced {ic_copy.nodes.count()} node rows from YAML.'),
            )

        # Point the copy and its nodes at the source via copy_of.
        self._set_copy_of(ic_src, ic_copy)

        if not options['no_pages']:
            self._copy_pages(ic_src, ic_copy, site_url, allow_dangling=options['allow_dangling_refs'])
            self._copy_site_content(ic_src, ic_copy)

    def _write_copy_config(self, ic_src: InstanceConfig, dest: str, dst_yaml: Path, options: dict[str, Any]) -> None:
        """Write the rewritten configs/<dst>.yaml (comment-preserving)."""
        from django.conf import settings

        from ruamel.yaml import YAML

        configs_dir = Path(settings.BASE_DIR, 'configs')
        src_yaml = configs_dir / f'{ic_src.identifier}.yaml'
        if not src_yaml.exists():
            raise CommandError(f"No standalone YAML config at '{src_yaml}'. Use --mode db for this instance.")
        if dst_yaml.exists():
            raise CommandError(f"Config '{dst_yaml}' already exists. (Use --use-existing-yaml to apply it.)")

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.width = 4096
        with src_yaml.open() as f:
            data = yaml.load(f)
        # With --name, the DB row and every YAML name field are forced to that single
        # value, so the DB/page title and the runtime/model name match exactly.
        # Without it, each name independently gets " (copy)" appended — pass the same
        # --name to both the --write-config-only and --use-existing-yaml runs to keep
        # the two stages consistent.
        rewrite_instance_yaml(data, dest_id=dest, name=options['name'])
        include_copies = rewrite_include_paths(data, src_id=ic_src.identifier, dest_id=dest)
        with dst_yaml.open('w') as f:
            yaml.dump(data, f)
        # Always track the files we create. On success nothing discards them (they are
        # the deliverable, including in --write-config-only); a failure or dry-run runs
        # _discard_created_files(), so a half-written config tree never survives.
        self._created_files.append(dst_yaml)
        self.stdout.write(f'  wrote {dst_yaml.relative_to(settings.BASE_DIR)}')

        # Copy the per-instance include fragments into configs/<dest>/ so the copy
        # owns its node-group source rather than sharing the source's files.
        for old_rel, new_rel in include_copies:
            src_frag = configs_dir / old_rel
            dst_frag = configs_dir / new_rel
            if not src_frag.exists():
                raise CommandError(f"Include fragment '{src_frag}' not found.")
            if dst_frag.exists():
                raise CommandError(f"Include fragment '{dst_frag}' already exists.")
            dst_frag.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_frag, dst_frag)
            self._created_files.append(dst_frag)
        if include_copies:
            self.stdout.write(f'  copied {len(include_copies)} include fragment(s) into configs/{dest}/.')

        # Surface identity-like fields the rewrite deliberately leaves shared with the
        # source. Fine for a sandbox; review them for a customer-facing clone.
        shared = [k for k in data if k == 'theme_identifier' or k.startswith('owner')]
        if shared:
            shared_list = ', '.join(sorted(shared))
            warn = (
                f'  kept source identity fields unchanged: {shared_list}. '
                + 'Absolute URLs embedded in rich-text/CTA page content are also not rewritten — '
                + 'review these for a customer-facing clone.'
            )
            self.stdout.write(self.style.WARNING(warn))

    def _set_copy_of(self, ic_src: InstanceConfig, ic_copy: InstanceConfig) -> None:
        """Point the copy (and each of its nodes) at the source via ``copy_of``."""
        InstanceConfig.objects.filter(pk=ic_copy.pk).update(copy_of=ic_src)
        src_by_id = {nc.identifier: nc.pk for nc in ic_src.nodes.all()}
        for nc in ic_copy.nodes.all():
            src_pk = src_by_id.get(nc.identifier)
            if src_pk is not None:
                NodeConfig.objects.filter(pk=nc.pk).update(copy_of_id=src_pk)

    def _node_pk_map(self, ic_src: InstanceConfig, ic_copy: InstanceConfig) -> dict[int, int]:
        """Map source NodeConfig pk -> copy NodeConfig pk, keyed by identifier."""
        src_by_id = {nc.identifier: nc.pk for nc in ic_src.nodes.all()}
        copy_by_id = {nc.identifier: nc.pk for nc in ic_copy.nodes.all()}
        node_map: dict[int, int] = {}
        for identifier, src_pk in src_by_id.items():
            copy_pk = copy_by_id.get(identifier)
            if copy_pk is not None:
                node_map[src_pk] = copy_pk
        return node_map

    def _copy_pages(
        self, ic_src: InstanceConfig, ic_copy: InstanceConfig, site_url: str | None, *, allow_dangling: bool = False
    ) -> None:
        if ic_src.root_page is None:
            self.stdout.write(self.style.WARNING('Source has no root page; skipping page copy.'))
            return

        node_map = self._node_pk_map(ic_src, ic_copy)
        source_pks = set(ic_src.nodes.values_list('pk', flat=True))
        src_home = ic_src.root_page.specific
        root_node = Page.get_first_root_node()
        assert root_node is not None

        self.stdout.write('Copying Wagtail page tree (with revisions)…')
        new_home = src_home.copy(
            recursive=True,
            to=root_node,
            update_attrs={'slug': ic_copy.identifier, 'title': ic_copy.get_name()},
            keep_live=True,
        )

        # Repoint node references on every copied page: both the live row and
        # every copied revision (drafts included).
        n_pages = 0
        n_revisions = 0
        copied_pages = list(new_home.get_descendants(inclusive=True).specific())
        for page in copied_pages:
            changed = remap_page_live(page, node_map)
            revs = remap_page_revisions(page, node_map)
            n_revisions += revs
            if changed or revs:
                n_pages += 1

        self.stdout.write(f'  remapped node refs on {n_pages} page(s), {n_revisions} revision(s).')

        # Fail loudly if any copied page/revision still points at a source node
        # (e.g. an unmapped node, or a stale page referencing a removed node).
        dangling = [hit for page in copied_pages for hit in find_source_node_refs(page, source_pks)]
        if dangling:
            sample = '; '.join(dangling[:10]) + (' …' if len(dangling) > 10 else '')
            if not allow_dangling:
                msg = (
                    f'{len(dangling)} copied page reference(s) still point at source-instance nodes: {sample}. '
                    + 'This usually means a node was not materialised in the copy. '
                    + 'Pass --allow-dangling-refs to override.'
                )
                raise CommandError(msg)
            self.stdout.write(self.style.WARNING(f'  {len(dangling)} dangling node ref(s) left (allowed): {sample}'))

        ic_copy.root_page = new_home
        ic_copy.save(update_fields=['root_page'])

        # Create an explicit hostname route for the copy so it is reachable.
        if site_url is None:
            self.stdout.write(self.style.WARNING('No --site-url given; pages copied but no InstanceHostname route created.'))
            return
        parsed = urlparse(site_url)
        if not parsed.hostname:
            raise CommandError(f'Could not parse a hostname from --site-url {site_url!r}.')

        # A non-root --site-url path (e.g. https://host/foo) maps onto base_path so routing matches the URL.
        base_path = parsed.path.rstrip('/')
        if InstanceHostname.objects.filter(hostname=parsed.hostname, base_path=base_path).exists():
            self.stdout.write(
                self.style.WARNING(f"  InstanceHostname '{parsed.hostname}' already exists; not creating a routing row."),
            )
        else:
            InstanceHostname.objects.create(instance=ic_copy, hostname=parsed.hostname, base_path=base_path)
            label = f'{parsed.hostname}{base_path}'
            self.stdout.write(f"  created InstanceHostname '{label}' → {ic_copy.identifier}.")

    def _copy_site_content(self, ic_src: InstanceConfig, ic_copy: InstanceConfig) -> None:
        """Copy instance-level Wagtail content (intro_content) not in the page tree."""
        from pages.models import InstanceSiteContent

        src_sc = InstanceSiteContent.objects.filter(instance=ic_src).first()
        if src_sc is None:
            return
        # A post_save signal already created a blank InstanceSiteContent for the
        # copy; update it. intro_content holds only RichText blocks (no node
        # references), so no remap is needed.
        InstanceSiteContent.objects.update_or_create(
            instance=ic_copy,
            defaults={'intro_content': src_sc.intro_content},
        )
        self.stdout.write('  copied InstanceSiteContent.')
