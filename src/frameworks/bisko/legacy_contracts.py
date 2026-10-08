"""
One-off repair: input contracts stored before shapes took over their per-combination lists.

An input contract used to list its required combinations itself (`combinations`). Those lists
are now required groups of a shape the contract refers to, and the field is gone, so stored
node rows and the template's published revisions that still carry it no longer parse. The value
contracts were only ever deployed to the data-studio staging backend, so instead of keeping the
old field readable, `setup_bisko` rewrites what is stored:

- a list naming no categories (a value in each year) becomes `required: true`;
- a per-combination list is dropped and reported: the template's next sync declares it as the
  shape's required groups, and its dependants are upgraded off the old revisions in the same run.

Remove this module once staging has run it.
"""

import json
from typing import Any, cast

from django.contrib.contenttypes.models import ContentType
from django.db import connection
from wagtail.models import Revision

from nodes.instance_serialization import InstanceSnapshot
from nodes.models import InstanceConfig, NodeConfig
from nodes.template_graph import snapshot_content_hash


def _upgrade_contract(contract: dict[str, Any]) -> int:
    """Rewrite one stored contract in place; return the number of per-combination requirements dropped."""
    combinations = contract.pop('combinations', None)
    if not combinations:
        return 0
    if all(not combination.get('categories') for combination in combinations):
        contract['required'] = True
        qualifiers = combinations[0].get('qualifiers') or {}
        if qualifiers:
            contract['qualifiers'] = qualifiers
        return 0
    return len(combinations)


def _upgrade_node_spec(spec: dict[str, Any]) -> tuple[bool, int]:
    changed, dropped = False, 0
    for port in spec.get('input_ports') or []:
        contract = port.get('validation')
        if isinstance(contract, dict) and 'combinations' in contract:
            dropped += _upgrade_contract(contract)
            changed = True
    return changed, dropped


def retire_legacy_value_contracts() -> list[str]:
    """Rewrite stored contracts that still list combinations; return what was done, one line per row."""
    return [*_upgrade_node_rows(), *_upgrade_revisions()]


def _upgrade_node_rows() -> list[str]:
    report: list[str] = []
    table = NodeConfig._meta.db_table
    with connection.cursor() as cursor:
        # The rows do not parse any more, so they are read and written as raw JSON.
        cursor.execute(f'SELECT id, identifier, spec::text FROM {table} WHERE spec::text LIKE \'%%"combinations"%%\'')  # noqa: S608
        for pk, identifier, raw in cursor.fetchall():
            spec = json.loads(raw)
            changed, dropped = _upgrade_node_spec(spec)
            if changed:
                cursor.execute(f'UPDATE {table} SET spec = %s::jsonb WHERE id = %s', [json.dumps(spec), pk])  # noqa: S608
                report.append(f'node {identifier}: contract rewritten, {dropped} per-combination requirements dropped')
    return report


def _upgrade_revisions() -> list[str]:
    report: list[str] = []
    content_type = ContentType.objects.get_for_model(InstanceConfig)
    revisions = Revision.objects.filter(content_type=content_type).order_by('pk')
    new_hashes: dict[int, str] = {}
    for revision in revisions:
        structured = (revision.content.get('model_snapshot') or {}).get('structured')
        if not structured:
            continue
        changed, dropped = False, 0
        for node in structured.get('nodes', []):
            node_changed, node_dropped = _upgrade_node_spec(node.get('spec') or {})
            changed |= node_changed
            dropped += node_dropped
        if changed:
            revision.save(update_fields=['content'])
            new_hashes[revision.pk] = snapshot_content_hash(InstanceSnapshot.from_serialized_data(structured, compose=False))
            report.append(f'revision {revision.pk} ({revision.object_str}): {dropped} per-combination requirements dropped')
    if new_hashes:
        # A dependant's revision records the content hash of the template revision it composed.
        for revision in revisions:
            structured = (revision.content.get('model_snapshot') or {}).get('structured')
            if not structured:
                continue
            pinned = cast('int | None', structured.get('template_revision_id'))
            if pinned in new_hashes and structured.get('template_content_hash') != new_hashes[pinned]:
                structured['template_content_hash'] = new_hashes[pinned]
                revision.save(update_fields=['content'])
                report.append(f'revision {revision.pk} ({revision.object_str}): template content hash restamped')
    return report
