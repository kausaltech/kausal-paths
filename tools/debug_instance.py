#!/usr/bin/env python3
"""
Quick debug tool for investigating DB-backed vs YAML-backed model instances.

Usage examples:
    # Sync YAML → DB, then compute from DB
    python tools/debug_instance.py -i espoo --sync --node net_emissions --filter 2020-2024,T

    # Compare YAML vs DB output for a node
    python tools/debug_instance.py -i espoo --source yaml --node net_emissions --filter 2020-2024,T
    python tools/debug_instance.py -i espoo --source db --node net_emissions --filter 2020-2024,T

    # Eval Python with instance/ctx/node in scope
    python tools/debug_instance.py -i espoo --source db -c "
        for n in ctx.nodes.values():
            if not n.input_dataset_instances:
                continue
            print(f'{n.id}: {[ds.id for ds in n.input_dataset_instances]}')
    "

    # Check a specific node's edges and inputs
    python tools/debug_instance.py -i espoo --source db --node building_heating_emissions -c "
        for e in node.edges:
            if e.output_node.id == node.id:
                print(f'{e.input_node.id} -> tags={e.tags}')
    "

    # Diff a node's config dict between YAML and DB
    python tools/debug_instance.py -i espoo --diff-node building_type_index
"""

# ruff: noqa: E402
from __future__ import annotations

from kausal_common.development.django import init_django

from nodes.exceptions import NodeError

init_django()

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from common.cache import CacheKind
from nodes.instance_loader import InstanceLoader
from nodes.models import InstanceConfig


def _load_from_yaml(instance_id: str) -> InstanceLoader:
    config_path = Path(f'configs/{instance_id}.yaml').resolve()
    if not config_path.exists():
        # Try other patterns
        for p in Path('configs').glob(f'{instance_id}*.yaml'):
            config_path = p.resolve()
            break
    if not config_path.exists():
        print(f'YAML config not found for {instance_id}', file=sys.stderr)
        sys.exit(1)
    return InstanceLoader.from_yaml(config_path)


def _load_from_db(instance_id: str, skip_validation: bool = True) -> InstanceLoader:
    from nodes.instance_from_db import serialize_instance_to_dict

    ic = InstanceConfig.objects.get(identifier=instance_id)
    if ic.config_source != 'database':
        print(f'Warning: {instance_id} config_source is "{ic.config_source}", not "database"', file=sys.stderr)

    config = serialize_instance_to_dict(ic)

    if skip_validation:
        orig = InstanceLoader.setup_validations

        def _nop(self) -> None:  # pyright: ignore[reportUnusedParameter]
            pass

        InstanceLoader.setup_validations = _nop  # type: ignore[method-assign]
        try:
            loader = InstanceLoader(config=config)
        finally:
            InstanceLoader.setup_validations = orig  # type: ignore[method-assign]
    else:
        loader = InstanceLoader(config=config)
    return loader


def _get_config_dict(instance_id: str, source: str) -> dict[str, Any]:
    """Get the raw config dict for diffing."""
    if source == 'db':
        from nodes.instance_from_db import serialize_instance_to_dict

        ic = InstanceConfig.objects.get(identifier=instance_id)
        return serialize_instance_to_dict(ic)

    loader = _load_from_yaml(instance_id)
    return dict(loader.config)


def _find_node_in_config(config: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for n in config.get('nodes', []) + config.get('actions', []):
        if n.get('id') == node_id:
            return dict(n)
    return None


def _diff_node(instance_id: str, node_id: str) -> None:
    yaml_config = _get_config_dict(instance_id, 'yaml')
    db_config = _get_config_dict(instance_id, 'db')

    yaml_node = _find_node_in_config(yaml_config, node_id)
    db_node = _find_node_in_config(db_config, node_id)

    if yaml_node is None:
        print(f'Node {node_id} not found in YAML config')
        return
    if db_node is None:
        print(f'Node {node_id} not found in DB config')
        return

    def to_json(d: dict[str, Any]) -> str:
        return json.dumps(d, indent=2, sort_keys=True, default=str)

    yaml_json = to_json(yaml_node)
    db_json = to_json(db_node)

    if yaml_json == db_json:
        print(f'Node {node_id}: YAML and DB configs are identical')
        return

    import difflib

    diff = difflib.unified_diff(
        yaml_json.splitlines(keepends=True),
        db_json.splitlines(keepends=True),
        fromfile='yaml',
        tofile='db',
    )
    sys.stdout.writelines(diff)


def _run_instance(args: argparse.Namespace) -> None:
    """Load an instance and run the requested operation (eval, compute, or summary)."""
    skip_validation = args.no_validation and not args.with_validation

    if args.source == 'yaml':
        loader = _load_from_yaml(args.instance)
    else:
        loader = _load_from_db(args.instance, skip_validation=skip_validation)

    instance = loader.instance
    ctx = loader.context

    if args.no_cache:
        ctx.cache.set_allowed_cache_kinds({CacheKind.RUN, CacheKind.LOCAL})
    if args.flush_cache:
        ctx.cache.clear()

    node = None
    if args.node:
        node = ctx.get_node(args.node)

    if args.code:
        code = textwrap.dedent(args.code)
        ns = {
            'instance': instance,
            'ctx': ctx,
            'node': node,
            'loader': loader,
            'json': json,
            'print': print,
        }
        with ctx.run():
            exec(compile(code, '<debug>', 'exec'), ns)  # noqa: S102
        return

    if node is not None:
        filters: list[str] = []
        if args.filter:
            filters = args.filter.split(',')
        try:
            with ctx.run():
                node.print_output(filters=filters or None)
        except NodeError as e:
            if e.event_chain:
                print('Error in computing node %s\nEvent chain: %s' % (e.event_chain[0].node.id, e.get_event_chain()))
            if e.__cause__:
                raise e.__cause__ from None
            raise
    else:
        print(f'Instance: {instance.id}')
        print(f'Source: {args.source}')
        print(f'Nodes: {len(ctx.nodes)}')
        print(f'Scenarios: {list(ctx.scenarios.keys())}')
        print(f'Global params: {list(ctx.global_parameters.keys())}')
        print(f'Dimensions: {list(ctx.dimensions.keys())}')


def main():
    parser = argparse.ArgumentParser(description='Debug model instances (YAML vs DB)')
    parser.add_argument('-i', '--instance', required=True, help='Instance identifier')
    parser.add_argument('--source', choices=['yaml', 'db'], default='db', help='Config source (default: db)')
    parser.add_argument('--node', help='Node identifier to inspect/compute')
    parser.add_argument('--filter', help='Output filter (e.g. 2020-2024,T)')
    parser.add_argument('--no-cache', action='store_true', help='Disable computation cache')
    parser.add_argument('--flush-cache', action='store_true', help='Flush external cache')
    parser.add_argument('--no-validation', action='store_true', default=True, help='Skip setup_validations (default: True)')
    parser.add_argument('--with-validation', action='store_true', help='Enable setup_validations')
    parser.add_argument('--sync', action='store_true', help='Sync YAML → DB before loading (implies --source db)')
    parser.add_argument('-c', '--code', help='Python code to eval (instance, ctx, node in scope)')
    parser.add_argument('--diff-node', help='Diff a node config between YAML and DB')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress log output')

    args = parser.parse_args()

    if args.quiet:
        from loguru import logger

        from kausal_common.logging.handler import loguru_logfmt_sink

        logger.remove()
        logger.add(loguru_logfmt_sink, format='{message}', level='WARNING')

    if args.diff_node:
        _diff_node(args.instance, args.diff_node)
        return

    if args.sync:
        from nodes.spec_export import sync_instance_to_db

        sync_instance_to_db(args.instance)
        args.source = 'db'

    _run_instance(args)


if __name__ == '__main__':
    main()
