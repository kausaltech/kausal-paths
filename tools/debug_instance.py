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

from kausal_common.logging.init import is_pretty_terminal

from nodes.exceptions import NodeError

init_django()

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from common.cache import CacheKind
from nodes.instance_loader import InstanceLoader
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from nodes.instance import Instance


def _load_from_yaml(instance_id: str) -> Instance:
    config_path = Path(f'configs/{instance_id}.yaml').resolve()
    if not config_path.exists():
        # Try other patterns
        for p in Path('configs').glob(f'{instance_id}*.yaml'):
            config_path = p.resolve()
            break
    if not config_path.exists():
        print(f'YAML config not found for {instance_id}', file=sys.stderr)
        sys.exit(1)
    loader = InstanceLoader.from_yaml(config_path)
    return loader.instance


def _load_from_db(instance_id: str) -> Instance:
    ic = InstanceConfig.objects.get(identifier=instance_id)
    if ic.config_source != 'database':
        print(f'Warning: {instance_id} config_source is "{ic.config_source}", not "database"', file=sys.stderr)
        ic.config_source = 'database'
    return ic.get_instance()


def _diff_node(instance_id: str, node_id: str) -> None:
    """
    Diff a node's *spec* between a fresh YAML parse and the stored DB spec.

    The YAML side is what ``parse_instance_snapshot`` produces from the current
    config file; the DB side is the stored ``NodeConfig.spec``. A diff means
    the DB mirror is stale (re-sync) or the parse changed.
    """
    from deepdiff import DeepDiff

    from nodes.instance_loader import InstanceYAMLConfig
    from nodes.instance_parser import parse_instance_snapshot
    from nodes.yaml_port_refs import build_yaml_port_reference_catalog

    ic = InstanceConfig.objects.get(identifier=instance_id)
    config_path = Path(f'configs/{instance_id}.yaml').resolve()
    yaml_conf = InstanceYAMLConfig.load_for_entrypoint(config_path)
    assert yaml_conf.data is not None
    node_uuids = {nc.identifier: nc.uuid for nc in ic.nodes.all().defer('spec')}
    snapshot = parse_instance_snapshot(
        yaml_conf.data,
        instance_uuid=ic.uuid,
        node_uuids=node_uuids,
        port_references=build_yaml_port_reference_catalog(ic),
    )

    yaml_spec = next((n.spec for n in snapshot.nodes if n.identifier == node_id), None)
    nc = ic.nodes.filter(identifier=node_id).first()
    db_spec = nc.spec if nc is not None else None

    if yaml_spec is None:
        print(f'Node {node_id} not found in YAML parse')
        return
    if db_spec is None:
        print(f'Node {node_id} has no spec in the DB')
        return

    diff = DeepDiff(db_spec.model_dump(mode='json'), yaml_spec.model_dump(mode='json'), verbose_level=2)
    if not diff:
        print(f'Node {node_id}: YAML-parsed and DB specs are identical')
        return

    print(f'Node {node_id}: spec differences (db → yaml parse)')
    if not is_pretty_terminal():
        print(diff.pretty())
    else:
        from rich.pretty import pprint

        pprint(diff)


def _run_instance(args: argparse.Namespace) -> None:  # noqa: C901
    """Load an instance and run the requested operation (eval, compute, or summary)."""
    if args.source == 'yaml':
        instance = _load_from_yaml(args.instance)
    else:
        instance = _load_from_db(args.instance)

    if args.save:
        ic = InstanceConfig.objects.get(identifier=instance.id)
        ic.config_source = 'database' if args.source == 'db' else 'yaml'
        ic.full_clean()
        ic.save()

    ctx = instance.context

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
    parser.add_argument('--save', action='store_true', help='Save the new config source into DB')
    parser.add_argument('--node', help='Node identifier to inspect/compute')
    parser.add_argument('--filter', help='Output filter (e.g. 2020-2024,T)')
    parser.add_argument('--no-cache', action='store_true', help='Disable computation cache')
    parser.add_argument('--flush-cache', action='store_true', help='Flush external cache')
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
        from nodes.spec_sync import sync_parsed_instance_to_db

        sync_parsed_instance_to_db(args.instance)
        args.source = 'db'

    _run_instance(args)


if __name__ == '__main__':
    main()
