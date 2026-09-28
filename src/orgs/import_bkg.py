"""Apply a validated BKG organization snapshot to the Paths organization tree."""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.db import transaction

import polars as pl

from frameworks.models import Framework, FrameworkOrganizationRoot
from orgs.models import Namespace, Organization, OrganizationClass, OrganizationIdentifier

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_COLUMNS = frozenset({
    'ars',
    'ags',
    'nuts3',
    'parent_ars',
    'name',
    'classification_identifier',
    'administrative_level',
    'primary_language',
    'primary_language_lowercase',
    'source_vintage',
})


@dataclass(frozen=True)
class ImportResult:
    created: int
    reused: int
    moved: int
    identifiers_created: int
    roots_attached: int
    name_differences: int


def _validated_rows(frame: pl.DataFrame) -> list[dict[str, object]]:  # noqa: C901, PLR0912
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f'Missing BKG columns: {sorted(missing)}. Regenerate the staged Parquet with the updated kausal-importers.'
        )
    rows = frame.sort(pl.col('ars').str.len_chars(), 'ars').to_dicts()
    if not rows:
        raise ValueError('BKG snapshot is empty')
    by_ars: dict[str, dict[str, object]] = {}
    by_ags: set[str] = set()
    by_nuts3: set[str] = set()
    for row in rows:
        ars = row['ars']
        ags = row['ags']
        parent = row['parent_ars']
        if not isinstance(ars, str) or not ars.isdigit() or len(ars) not in (2, 5, 9, 12):
            raise ValueError(f'Invalid ARS {ars!r}')
        if ars in by_ars:
            raise ValueError(f'Duplicate ARS {ars}')
        if ags is not None:
            if not isinstance(ags, str) or len(ags) != 8 or not ags.isdigit() or ags in by_ags:
                raise ValueError(f'Invalid or duplicate AGS {ags!r}')
            by_ags.add(ags)
        if parent is not None and (parent not in by_ars or len(parent) >= len(ars)):
            raise ValueError(f'{ars}: parent {parent!r} is missing or not an ancestor')
        if parent is None and len(ars) != 2:
            raise ValueError(f'{ars}: only states may be roots')
        if row['primary_language'] != 'de' or row['primary_language_lowercase'] != 'de':
            raise ValueError(f'{ars}: expected German language fields')
        if not isinstance(row['name'], str) or not row['name']:
            raise ValueError(f'{ars}: empty name')
        nuts3 = row['nuts3']
        if len(ars) == 5:
            if not isinstance(nuts3, str) or len(nuts3) != 5 or not nuts3.startswith('DE'):
                raise ValueError(f'{ars}: missing or invalid district NUTS3 code {nuts3!r}')
            if nuts3 in by_nuts3:
                raise ValueError(f'{ars}: duplicate NUTS3 code {nuts3}')
            by_nuts3.add(nuts3)
        by_ars[ars] = row
    for row in rows:
        if row['ags'] is None:
            continue
        parent = str(row['parent_ars'])
        while len(parent) != 5:
            parent = str(by_ars[parent]['parent_ars'])
        if row['nuts3'] != by_ars[parent]['nuts3']:
            raise ValueError(f'{row["ars"]}: municipal NUTS3 code disagrees with district {parent}')
    if len({row['source_vintage'] for row in rows}) != 1:
        raise ValueError('BKG rows must have a single source vintage')
    return rows


@transaction.atomic
def import_bkg_organizations(source: str | Path, *, framework: Framework) -> ImportResult:  # noqa: C901, PLR0912, PLR0915
    """Reconcile by ARS/AGS, preserving existing organization and instance FKs."""
    rows = _validated_rows(pl.read_parquet(source))
    classes = {obj.identifier: obj for obj in OrganizationClass.objects.all()}
    unknown = {str(row['classification_identifier']) for row in rows} - classes.keys()
    if unknown:
        raise ValueError(f'Unknown organization classes: {sorted(unknown)}; run setup_bisko.py first')
    ars_namespace = Namespace.objects.get(identifier='ars')
    ags_namespace = Namespace.objects.get(identifier='ags')
    nuts3_namespace, _ = Namespace.objects.get_or_create(
        identifier='nuts3', defaults={'name': 'NUTS-3-Region', 'user_editable': False}
    )
    codes = [str(row['ars']) for row in rows]
    ags_codes = [str(row['ags']) for row in rows if row['ags'] is not None]
    ars_existing = {
        ident.identifier: ident.organization
        for ident in OrganizationIdentifier.objects.select_related('organization').filter(
            namespace=ars_namespace, identifier__in=codes
        )
    }
    ags_existing = {
        ident.identifier: ident.organization
        for ident in OrganizationIdentifier.objects.select_related('organization').filter(
            namespace=ags_namespace, identifier__in=ags_codes
        )
    }
    resolved: dict[str, Organization] = {}
    claimed: set[int] = set()
    # Detect identifier conflicts before changing the tree.
    for row in rows:
        ars = str(row['ars'])
        ags = str(row['ags']) if row['ags'] is not None else None
        by_ars = ars_existing.get(ars)
        by_ags = ags_existing.get(ags) if ags else None
        if by_ars and by_ags and by_ars.pk != by_ags.pk:
            raise ValueError(f'{ars}: ARS and AGS identify different organizations')
        org = by_ars or by_ags
        if org:
            if org.pk in claimed:
                raise ValueError(f'{ars}: an organization matches multiple BKG rows')
            claimed.add(org.pk)
            resolved[ars] = org

    created = moved = identifiers_created = roots_attached = name_differences = 0
    for row in rows:
        ars = str(row['ars'])
        ags = str(row['ags']) if row['ags'] is not None else None
        parent = resolved.get(str(row['parent_ars'])) if row['parent_ars'] else None
        org = resolved.get(ars)
        if org is None:
            org = Organization(
                name=str(row['name']),
                primary_language='de',
                primary_language_lowercase='de',
                classification=classes[str(row['classification_identifier'])],
            )
            org = parent.add_child(instance=org) if parent else Organization.add_root(instance=org)
            resolved[ars] = org
            created += 1
        else:
            org.refresh_from_db()
            if org.name != row['name']:
                name_differences += 1  # Preserve locally maintained names.
            if org.classification_id is None:
                org.classification = classes[str(row['classification_identifier'])]
                org.save(update_fields=['classification'])
            elif org.classification_id != classes[str(row['classification_identifier'])].pk:
                raise ValueError(f'{ars}: existing organization has another classification')
            current_parent = org.get_parent()
            if (current_parent.pk if current_parent else None) != (parent.pk if parent else None):
                if parent:
                    org.move(parent, pos='last-child')
                else:
                    raise ValueError(f'{ars}: existing state is not a root')
                org.refresh_from_db()
                moved += 1
        for namespace, identifier in ((ars_namespace, ars), (ags_namespace, ags)):
            if identifier is None:
                continue
            _, was_created = OrganizationIdentifier.objects.get_or_create(
                namespace=namespace,
                identifier=identifier,
                defaults={'organization': org},
            )
            identifiers_created += was_created
        if parent is None:
            _, was_attached = FrameworkOrganizationRoot.objects.get_or_create(framework=framework, organization=org)
            roots_attached += was_attached
    # All administrative rows are reconciled before attaching statistical identifiers.
    # On a repeat import this writes only the missing identifiers, without saving Organizations.
    district_rows = [row for row in rows if len(str(row['ars'])) == 5]
    nuts3_codes = [str(row['nuts3']) for row in district_rows]
    existing_nuts3 = {
        ident.identifier: ident.organization_id
        for ident in OrganizationIdentifier.objects.filter(namespace=nuts3_namespace, identifier__in=nuts3_codes)
    }
    district_ids = {resolved[str(row['ars'])].pk for row in district_rows}
    other_codes: dict[int, set[str]] = defaultdict(set)
    for organization_id, code in OrganizationIdentifier.objects.filter(
        namespace=nuts3_namespace, organization_id__in=district_ids
    ).values_list('organization_id', 'identifier'):
        other_codes[organization_id].add(code)
    new_identifiers = []
    for row in district_rows:
        org = resolved[str(row['ars'])]
        code = str(row['nuts3'])
        if existing_nuts3.get(code, org.pk) != org.pk or other_codes[org.pk] - {code}:
            raise ValueError(f'{row["ars"]}: NUTS3 code {code} conflicts with an existing organization identifier')
        if code not in existing_nuts3:
            new_identifiers.append(OrganizationIdentifier(namespace=nuts3_namespace, identifier=code, organization=org))
    OrganizationIdentifier.objects.bulk_create(new_identifiers)
    identifiers_created += len(new_identifiers)
    return ImportResult(created, len(rows) - created, moved, identifiers_created, roots_attached, name_differences)
