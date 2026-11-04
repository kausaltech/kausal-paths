"""
List identifiers in the database that the runtime model rejects. Read-only.

The runtime accepts only ``^[a-z0-9_]+$`` for node, dimension and category identifiers
(``paths.identifiers.Identifier``). Until the model editor checked this, a category saved as
``Fernwärme``, ``Energy`` or ``district-heating`` was stored without complaint, and the whole
instance then failed to load on its next start. This command finds such identifiers so they can be
repaired before anyone trips over them.

It looks in four places:

    category    DimensionCategory.identifier
    dimension   DimensionScope.identifier, the dimension's identifier in one scope
    node        NodeConfig.identifier
    spec        InstanceConfig.spec.dimensions, the copy a database-sourced instance loads from

The fourth matters because the editor mirrors every category edit into the instance spec, so a bad
identifier can sit there as well as in the row.

    python manage.py check_identifiers
    python manage.py check_identifiers potsdam-gpc muenchen-bisko

Output is one tab-separated line per finding, so it survives a narrow terminal and can be piped.
Nothing is written. For a database-sourced instance, ``rename_dimension_category <instance> --auto``
repairs the categories, row and spec mirror together; the command lines are printed at the end. A
yaml-sourced instance is fixed in its YAML.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandParser

from kausal_common.datasets.models import Dimension, DimensionCategory, DimensionScope

from paths.identifiers import LOWER_IDENTIFIER_PATTERN

from nodes.models import InstanceConfig, NodeConfig

if TYPE_CHECKING:
    from uuid import UUID

VALID = re.compile(LOWER_IDENTIFIER_PATTERN)
COLUMNS = ('where', 'scope', 'source', 'dimension', 'identifier', 'uuid')


@dataclass(frozen=True)
class Finding:
    where: str
    scope: str
    config_source: str
    container: str
    identifier: str
    uuid: str

    def row(self) -> str:
        return '\t'.join((self.where, self.scope, self.config_source, self.container, repr(self.identifier), self.uuid))


@dataclass(frozen=True)
class ScopeRef:
    scope: str
    config_source: str
    dimension: str


def _is_valid(identifier: str) -> bool:
    return VALID.fullmatch(identifier) is not None


class Command(BaseCommand):
    help = 'List node, dimension and category identifiers the runtime model rejects. Read-only.'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('instances', nargs='*', help='Instance identifiers; default all instances.')

    def handle(self, *args, **options) -> None:
        selected: list[str] = options['instances']
        ics = InstanceConfig.objects.all()
        if selected:
            ics = ics.filter(identifier__in=selected)
            missing = set(selected) - set(ics.values_list('identifier', flat=True))
            if missing:
                self.stderr.write(f'Unknown instances: {", ".join(sorted(missing))}')
        ic_by_pk = {ic.pk: ic for ic in ics}

        findings = [
            *self._scan_dimension_rows(ic_by_pk, include_unscoped=not selected),
            *self._scan_nodes(ic_by_pk),
            *self._scan_specs(list(ic_by_pk.values())),
        ]
        if not findings:
            self.stdout.write(f'No invalid identifiers ({len(ic_by_pk)} instances checked).')
            return
        self.stdout.write(f'{len(findings)} invalid identifiers; valid is {LOWER_IDENTIFIER_PATTERN}')
        self.stdout.write('\t'.join(COLUMNS))
        for f in sorted(findings, key=lambda f: (f.scope, f.where, f.container, f.identifier)):
            self.stdout.write(f.row())
        fixable = sorted({f.scope for f in findings if f.where in ('category', 'spec') and f.config_source == 'database'})
        if fixable:
            self.stdout.write('\nTo repair the categories (plan first, then add --apply):')
            for scope in fixable:
                self.stdout.write(f'python manage.py rename_dimension_category {scope} --auto')

    @staticmethod
    def _scan_dimension_rows(ic_by_pk: dict[int, InstanceConfig], include_unscoped: bool) -> list[Finding]:
        ic_ct = ContentType.objects.get_for_model(InstanceConfig)
        ct_models = dict(ContentType.objects.values_list('pk', 'model'))
        dim_uuids: dict[int, UUID] = dict(Dimension.objects.values_list('pk', 'uuid'))

        findings: list[Finding] = []
        refs_by_dim: dict[int, list[ScopeRef]] = {}
        rows = DimensionScope.objects.values_list('dimension', 'scope_content_type', 'scope_id', 'identifier')
        for dim_pk, ct_pk, scope_id, identifier in rows:
            if ct_pk == ic_ct.pk:
                ic = ic_by_pk.get(scope_id)
                if ic is None:
                    continue
                ref = ScopeRef(ic.identifier, ic.config_source, identifier or '?')
            elif include_unscoped:
                ref = ScopeRef(f'{ct_models.get(ct_pk, ct_pk)}:{scope_id}', '-', identifier or '?')
            else:
                continue
            refs_by_dim.setdefault(dim_pk, []).append(ref)
            if identifier is not None and not _is_valid(identifier):
                findings.append(
                    Finding('dimension', ref.scope, ref.config_source, '-', identifier, str(dim_uuids.get(dim_pk, '?')))
                )

        cats = DimensionCategory.objects.exclude(identifier__isnull=True).exclude(identifier__regex=LOWER_IDENTIFIER_PATTERN)
        for cat_uuid, identifier, dim_pk in cats.values_list('uuid', 'identifier', 'dimension'):
            refs = refs_by_dim.get(dim_pk)
            if not refs:
                if include_unscoped:
                    findings.append(Finding('category', '(no scope)', '-', '?', identifier or '', str(cat_uuid)))
                continue
            findings.extend(
                Finding('category', ref.scope, ref.config_source, ref.dimension, identifier or '', str(cat_uuid)) for ref in refs
            )
        return findings

    @staticmethod
    def _scan_nodes(ic_by_pk: dict[int, InstanceConfig]) -> list[Finding]:
        nodes = NodeConfig.objects.filter(instance__in=ic_by_pk).exclude(identifier__regex=LOWER_IDENTIFIER_PATTERN)
        findings: list[Finding] = []
        for node_uuid, identifier, ic_pk in nodes.values_list('uuid', 'identifier', 'instance'):
            ic = ic_by_pk[ic_pk]
            findings.append(Finding('node', ic.identifier, ic.config_source, '-', identifier, str(node_uuid)))
        return findings

    @staticmethod
    def _scan_specs(ics: list[InstanceConfig]) -> list[Finding]:
        findings: list[Finding] = []
        for ic in ics:
            spec = ic.spec
            if spec is None:
                continue
            for dim in spec.dimensions:
                dim_id = str(dim.get('id', ''))
                if not _is_valid(dim_id):
                    findings.append(Finding('spec', ic.identifier, ic.config_source, '-', dim_id, '-'))
                for cat in dim.get('categories', []):
                    cat_id = str(cat.get('id', ''))
                    if not _is_valid(cat_id):
                        findings.append(Finding('spec', ic.identifier, ic.config_source, dim_id, cat_id, '-'))
        return findings
