#!/usr/bin/env python3
"""
Oracle for the loader inversion: parse output must equal export output.

Verifies that parsing YAML directly into specs produces exactly what the
runtime-export path (``nodes/spec_export.py``) does. For each instance:
  1. Run the current runtime-based ``sync_instance_to_db`` inside a rolled-back
     transaction (forecast-default promotion disabled) and capture the
     resulting ``InstanceSnapshot`` from the DB.
  2. Run ``parse_instance_snapshot`` on the same merged YAML dicts.
  3. Deep-diff the two snapshots.

The export-side snapshots are cached (they need full runtime init, which is
slow); use ``--refresh`` after changing the export path or the YAML.

Usage:
    python tools/parse_oracle.py                 # all non-framework instances
    python tools/parse_oracle.py -i espoo        # one instance
    python tools/parse_oracle.py -i espoo --refresh
"""

# ruff: noqa: E402
from __future__ import annotations

from kausal_common.development.django import init_django

init_django()

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from django.db import transaction

from deepdiff import DeepDiff

from nodes.instance_loader import InstanceYAMLConfig
from nodes.instance_parser import parse_instance_snapshot
from nodes.instance_serialization import InstanceSnapshot, build_instance_snapshot
from nodes.models import InstanceConfig

CACHE_DIR = Path('.parse-oracle-cache')


class _RollbackError(Exception):
    """Control-flow signal: unwind the oracle's transaction without committing."""


def export_side_snapshot(instance_id: str, *, refresh: bool) -> tuple[InstanceSnapshot, UUID, dict[str, UUID], dict[str, Any]]:
    """
    Snapshot from the runtime-export path, via a rolled-back sync. Cached.

    Also captures the dataset schema info (inside the transaction, where
    sync-created placeholder datasets still exist) for the parse side's
    binding resolution.
    """
    from nodes.spec_export import sync_instance_to_db
    from nodes.spec_sync import collect_dataset_schema_info

    cache_fn = CACHE_DIR / f'{instance_id}.json'
    if not refresh and cache_fn.exists():
        cached = json.loads(cache_fn.read_text())
        if 'schemas' in cached:
            snapshot = InstanceSnapshot.from_serialized_data(cached['snapshot'])
            node_uuids = {k: UUID(v) for k, v in cached['node_uuids'].items()}
            return snapshot, UUID(cached['instance_uuid']), node_uuids, cached['schemas']

    result: list[tuple[InstanceSnapshot, UUID, dict[str, UUID], dict[str, Any]]] = []
    try:
        with transaction.atomic():
            sync_instance_to_db(instance_id, promote_forecast_defaults=False)
            ic = InstanceConfig.objects.get(identifier=instance_id)
            snapshot = build_instance_snapshot(ic)
            node_uuids = {nc.identifier: nc.uuid for nc in ic.nodes.all().defer('spec')}
            schema_info = collect_dataset_schema_info(ic)
            schemas = {
                k: {'metric_keys': v.metric_keys, 'metric_names': v.metric_names, 'forecast_from': v.forecast_from}
                for k, v in schema_info.items()
            }
            result.append((snapshot, ic.uuid, node_uuids, schemas))
            raise _RollbackError()  # noqa: TRY301
    except _RollbackError:
        pass

    snapshot, instance_uuid, node_uuids, schemas = result[0]
    CACHE_DIR.mkdir(exist_ok=True)
    cache_fn.write_text(
        json.dumps({
            'snapshot': snapshot.model_dump(mode='json'),
            'instance_uuid': str(instance_uuid),
            'node_uuids': {k: str(v) for k, v in node_uuids.items()},
            'schemas': schemas,
        })
    )
    return snapshot, instance_uuid, node_uuids, schemas


def parse_side_snapshot(
    instance_id: str, instance_uuid: UUID, node_uuids: dict[str, UUID], schemas: dict[str, Any]
) -> InstanceSnapshot:
    from nodes.spec_sync import DatasetSchemaInfo, resolve_dataset_port_snapshots

    config_path = Path(f'configs/{instance_id}.yaml').resolve()
    yaml_conf = InstanceYAMLConfig.load_for_entrypoint(config_path)
    data = yaml_conf.data
    assert data is not None
    snapshot = parse_instance_snapshot(data, instance_uuid=instance_uuid, node_uuids=node_uuids)
    # The sync write-half forces this feature on; mirror it so the comparison
    # targets parse fidelity, not write-half policy.
    snapshot.spec.features.use_datasets_from_db = True
    # Resolve dataset-port metrics against the captured schemas — the write half's job.
    schema_info = {
        k: DatasetSchemaInfo(metric_keys=v['metric_keys'], metric_names=v['metric_names'], forecast_from=v.get('forecast_from'))
        for k, v in schemas.items()
    }
    snapshot.dataset_ports = resolve_dataset_port_snapshots(snapshot, schema_info)
    # Materialize lazy translation promises under the instance's languages,
    # the same way SchemaField storage did for the DB side.
    from kausal_common.i18n.pydantic import set_i18n_context

    with set_i18n_context(data['default_language'], data.get('supported_languages', [])):
        return InstanceSnapshot.model_validate(snapshot.model_dump(mode='json'))


def full_sync_side_snapshot(instance_id: str) -> InstanceSnapshot:
    """
    Snapshot from the parse-only sync's full DB effects, via a rolled-back run.

    This exercises the entire write half (metadata columns, node rows,
    dimensions, placeholders, edges, dataset ports), not just the parse.
    """
    from nodes.spec_sync import sync_parsed_instance_to_db

    result: list[InstanceSnapshot] = []
    try:
        with transaction.atomic():
            sync_parsed_instance_to_db(instance_id, promote_forecast_defaults=False)
            ic = InstanceConfig.objects.get(identifier=instance_id)
            result.append(build_instance_snapshot(ic))
            raise _RollbackError()  # noqa: TRY301
    except _RollbackError:
        pass
    return result[0]


_LANG_KEY_RE = re.compile(r'^[a-z]{2,3}(-[A-Z]{2})?$')


def _strip_empty_langs(value: Any) -> Any:
    """
    Drop empty-string language entries from TranslatedString dicts.

    The SchemaField write path materializes missing context languages as ''
    on the DB side; the in-memory parse side doesn't. A parse-only sync
    writes through the same field, so the difference never reaches the DB —
    normalize it away here.
    """
    if isinstance(value, dict):
        stripped = {
            k: _strip_empty_langs(v) for k, v in value.items() if not (v == '' and isinstance(k, str) and _LANG_KEY_RE.match(k))
        }
        # A language-less plain string gets tagged with whichever language was
        # active when its module was imported (class-level parameter labels).
        # Collapse single-language string dicts to the bare string so the
        # comparison is about content, not import order.
        if len(stripped) == 1:
            ((k, v),) = stripped.items()
            if isinstance(v, str) and isinstance(k, str) and _LANG_KEY_RE.match(k):
                return v
        return stripped
    if isinstance(value, list):
        return [_strip_empty_langs(v) for v in value]
    return value


def _dump(obj: Any) -> Any:
    return _strip_empty_langs(obj.model_dump(mode='json'))


def _node_metadata_parts(node: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = {'uuid': str(node.uuid), 'identifier': node.identifier}
    display = _strip_empty_langs(
        node.model_dump(
            mode='json',
            include={'name', 'short_name', 'short_description', 'color', 'order', 'is_visible'},
        )
    )
    return identity, display


def compare_snapshots(  # noqa: C901, PLR0915
    db_snap: InstanceSnapshot,
    parse_snap: InstanceSnapshot,
    schemas: dict[str, Any] | None = None,
    persisted_node_identifiers: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []

    def diff(label: str, a: Any, b: Any, *, into: list[str] | None = None) -> None:
        dd = DeepDiff(a, b, ignore_order=False, verbose_level=2)
        if dd:
            (problems if into is None else into).append(f'{label}: {dd.pretty()}')

    # Metadata name/owner can carry stale pre-sync admin edits in the DB row's
    # i18n dict (sync merges rather than replaces); warn, don't fail.
    diff('metadata', _dump(db_snap.metadata), _dump(parse_snap.metadata), into=warnings)
    diff('spec', _dump(db_snap.spec), _dump(parse_snap.spec))

    assert all(n.identifier is not None for n in db_snap.nodes)
    assert all(n.identifier is not None for n in parse_snap.nodes)
    db_nodes = {cast('str', n.identifier): n for n in db_snap.nodes}
    parse_nodes = {cast('str', n.identifier): n for n in parse_snap.nodes}
    for ident in sorted(db_nodes.keys() | parse_nodes.keys()):
        a, b = db_nodes.get(ident), parse_nodes.get(ident)
        if a is None or b is None:
            problems.append(f'node {ident}: only in {"parse" if a is None else "db"} side')
            continue
        a_spec, b_spec = a.spec, b.spec
        assert a_spec is not None
        assert b_spec is not None
        identity_a, display_a = _node_metadata_parts(a)
        identity_b, display_b = _node_metadata_parts(b)
        dump_a, dump_b = _dump(a_spec), _dump(b_spec)
        if persisted_node_identifiers is not None and ident not in persisted_node_identifiers:
            # Neither side's uuid persisted (both rolled-back runs invented
            # one for a row missing from the live DB); ignore it.
            identity_a.pop('uuid')
            identity_b.pop('uuid')
        diff(f'node {ident} identity', identity_a, identity_b)
        # Existing DB-authored display metadata intentionally wins over YAML
        # during sync; report drift without making parse fidelity fail.
        diff(f'node {ident} display metadata', display_a, display_b, into=warnings)
        diff(f'node {ident} spec', dump_a, dump_b)

    def edge_key(e: Any) -> tuple[Any, ...]:
        return (e.from_node, e.to_node, str(e.from_port), str(e.to_port))

    db_edges = {edge_key(e): _dump(e) for e in db_snap.edges}
    parse_edges = {edge_key(e): _dump(e) for e in parse_snap.edges}
    for key in sorted(db_edges.keys() | parse_edges.keys()):
        a, b = db_edges.get(key), parse_edges.get(key)
        if a is None or b is None:
            problems.append(f'edge {key}: only in {"parse" if a is None else "db"} side')
            continue
        diff(f'edge {key}', a, b)

    def port_key(p: Any) -> tuple[Any, ...]:
        return (p.node, p.dataset, str(p.port_id))

    db_ports = {port_key(p): p for p in db_snap.dataset_ports}
    parse_ports = {port_key(p): p for p in parse_snap.dataset_ports}
    for key in sorted(db_ports.keys() | parse_ports.keys()):
        pa, pb = db_ports.get(key), parse_ports.get(key)
        if pa is None or pb is None:
            problems.append(f'dataset port {key}: only in {"parse" if pa is None else "db"} side')
            continue
        if pa.dataset_index != pb.dataset_index:
            problems.append(f'dataset port {key}: dataset_index {pa.dataset_index} != {pb.dataset_index}')
        if pa.metric != pb.metric:
            problems.append(f'dataset port {key}: metric {pa.metric!r} != {pb.metric!r}')
        spec_a, spec_b = pa.spec, pb.spec
        # The runtime export echoes a dataset-level forecast default back into
        # the binding pipeline (DBDataset.from_def). Parse doesn't; after
        # promotion the DB end states are identical. Normalize the echo away.
        ds_default = (schemas or {}).get(pa.dataset, {}).get('forecast_from')
        if ds_default is not None and spec_a.forecast_from == ds_default and spec_b.forecast_from is None:
            spec_a = spec_a.without_forecast_from()
        diff(f'dataset port {key} spec', _dump(spec_a), _dump(spec_b))

    return problems, warnings


def instance_ids_from_db() -> list[str]:
    ids = InstanceConfig.objects.filter(framework_config__isnull=True).values_list('identifier', flat=True)
    return [i for i in ids if Path(f'configs/{i}.yaml').exists()]


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--instance', action='append', help='Instance id(s); default: all non-framework')
    parser.add_argument('--refresh', action='store_true', help='Re-run the runtime export instead of using the cache')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument(
        '--full-sync',
        action='store_true',
        help='Compare full DB effects of the parse-only sync (rolled back) instead of the in-memory parse',
    )
    args = parser.parse_args()

    instance_ids = args.instance or instance_ids_from_db()
    failures = 0
    for instance_id in instance_ids:
        try:
            export_result = export_side_snapshot(instance_id, refresh=args.refresh)
        except Exception as e:
            print(f'[EXPORT-ERROR] {instance_id}: {type(e).__name__}: {e}')
            failures += 1
            if args.fail_fast:
                raise
            continue
        db_snap, instance_uuid, node_uuids, schemas = export_result
        persisted_node_identifiers: set[str] | None = None
        try:
            if args.full_sync:
                ic = InstanceConfig.objects.filter(identifier=instance_id).first()
                persisted_node_identifiers = set(ic.nodes.values_list('identifier', flat=True)) if ic is not None else set()
                parse_snap = full_sync_side_snapshot(instance_id)
            else:
                parse_snap = parse_side_snapshot(instance_id, instance_uuid, node_uuids, schemas)
        except Exception as e:
            print(f'[PARSE-ERROR] {instance_id}: {type(e).__name__}: {e}')
            failures += 1
            if args.fail_fast:
                raise
            continue
        problems, warnings = compare_snapshots(db_snap, parse_snap, schemas, persisted_node_identifiers)
        for w in warnings:
            print(f'    [warn] {instance_id}: {w}')
        if problems:
            failures += 1
            print(f'[DIFF] {instance_id}: {len(problems)} problem(s)')
            for p in problems:
                print(f'    {p}')
            if args.fail_fast:
                break
        else:
            print(f'[OK] {instance_id}')
    print(f'\n{len(instance_ids) - failures}/{len(instance_ids)} instances OK')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
