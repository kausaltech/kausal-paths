"""
Rename dimension category identifiers that the runtime model rejects. Plans unless --apply.

The repair for what ``check_identifiers`` finds. Until the model editor validated identifiers, a
category could be saved as ``Straßenbahn`` or ``gaz ziemny``; the instance then fails to load, the
editor with it, so the fix cannot be made from the editor.

    python manage.py rename_dimension_category cads-bytom-6h4m6u --auto
    python manage.py rename_dimension_category cads-bytom-6h4m6u --auto --apply
    python manage.py rename_dimension_category cads-bytom-6h4m6u --rename 'odpady:Osady sciekowe=osady_sciekowe'

``--auto`` takes every invalid category of the instance and proposes an identifier by
transliteration (``Straßenbahn`` -> ``strassenbahn``, ``paliwo węglowe`` -> ``paliwo_weglowe``).
``--rename DIM:OLD=NEW`` names one explicitly, and wins over ``--auto`` for that category.

The category row keeps its pk and UUID, so dataset points and everything else that refers to it by
foreign key follow. Two copies hold the identifier as text and are handled here:

- ``InstanceConfig.spec.dimensions``, from which a database-sourced instance loads, is rewritten
  from the rows, as the editor does after its own edits. Only for a dimension the spec already
  carries; a dimension used only by datasets is not added to it.
- Node specs, binding transformations and the rest of the instance spec (scenarios, parameters) may
  name the category in a filter. Those are not rewritten: the command refuses while any exist, and
  ``--allow-references`` overrides, leaving them to be fixed by hand.

A yaml-sourced instance is refused: its categories come from the YAML, so that is where the fix
belongs, and the next sync would bring the old identifier back. A published instance keeps serving
its live revision until it is published again.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import transaction

from kausal_common.datasets.models import DimensionCategory, DimensionScope

from paths.identifiers import LOWER_IDENTIFIER_PATTERN

from nodes.graphql.editor import InstanceEditorMutation
from nodes.models import InstanceConfig, NodeConfig, NodeInputPortBinding

VALID = re.compile(LOWER_IDENTIFIER_PATTERN)
# Letters NFKD does not decompose into an ASCII base, and the German convention for umlauts.
TRANSLITERATION = str.maketrans({'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss', 'ł': 'l', 'ø': 'o', 'æ': 'ae', 'đ': 'd'})


def suggest_identifier(old: str) -> str:
    text = unicodedata.normalize('NFKD', old.lower().translate(TRANSLITERATION))
    text = text.encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '_', text).strip('_') or 'category'


def _dump(obj: object) -> str:
    # ensure_ascii=False on both sides: a needle escaped as ß never matches an unescaped spec.
    return json.dumps(obj, ensure_ascii=False)


def _invalid_categories(scopes: dict[str, DimensionScope]) -> dict[tuple[str, str], str | None]:
    return {
        (dim_id, cat_id): None
        for dim_id, scope in scopes.items()
        for cat_id in scope.dimension.categories.values_list('identifier', flat=True)
        if cat_id is not None and not VALID.fullmatch(cat_id)
    }


def _parse_renames(explicit: list[str]) -> tuple[dict[tuple[str, str], str | None], list[str]]:
    wanted: dict[tuple[str, str], str | None] = {}
    errors: list[str] = []
    for arg in explicit:
        dim_id, sep, rest = arg.partition(':')
        old, sep2, new = rest.rpartition('=')
        if not sep or not sep2 or not dim_id or not old:
            errors.append(f'--rename {arg!r}: expected DIM:OLD=NEW')
        else:
            wanted[(dim_id, old)] = new
    return wanted, errors


def _target_identifier(old: str, new: str | None, used: set[str]) -> tuple[str, str | None]:
    """Return the identifier to rename to, or an error. A suggestion steps aside from taken ones."""
    if new is not None:
        if not VALID.fullmatch(new):
            return new, f'{new!r} is not valid either; use a-z, 0-9 and _ only'
        if new in used:
            return new, f'{new!r} is already taken'
        return new, None
    base = target = suggest_identifier(old)
    n = 2
    while target in used:
        target, n = f'{base}_{n}', n + 1
    return target, None


@dataclass
class Rename:
    scope: DimensionScope
    category: DimensionCategory
    old: str
    new: str
    references: list[str] = field(default_factory=list)


class Command(BaseCommand):
    help = 'Rename dimension category identifiers the runtime model rejects. Plans unless --apply.'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('instance', help='Identifier of a database-sourced instance')
        parser.add_argument('--auto', action='store_true', help='Rename every invalid category, with suggested identifiers')
        parser.add_argument(
            '--rename', action='append', default=[], metavar='DIM:OLD=NEW', help='Rename one category; repeatable'
        )
        parser.add_argument('--allow-references', action='store_true', help='Apply although node specs or the spec name OLD')
        parser.add_argument('--apply', action='store_true', help='Write the changes; without it, only the plan is shown')

    def handle(self, *args, **options) -> None:
        ic = InstanceConfig.objects.filter(identifier=options['instance']).first()
        if ic is None:
            raise CommandError(f'No instance {options["instance"]!r}')
        if ic.config_source != 'database':
            raise CommandError(
                f'{ic.identifier} is {ic.config_source}-sourced: its categories come from the YAML config. '
                'Fix the identifier there; the next sync would restore the old one.'
            )
        if not options['auto'] and not options['rename']:
            raise CommandError('Nothing to do: give --auto or --rename DIM:OLD=NEW')

        renames, errors = self._plan(ic, options['rename'], auto=options['auto'])
        if not renames and not errors:
            self.stdout.write(f'{ic.identifier}: no invalid category identifiers.')
            return
        self._find_references(ic, renames)
        self._report(ic, renames, errors)

        blocked = [r for r in renames if r.references]
        if errors or (blocked and not options['allow_references']):
            if blocked and not errors:
                self.stdout.write('Refusing: fix the references first, or pass --allow-references.')
            raise CommandError('Nothing written.')
        if not options['apply']:
            self.stdout.write('Plan only; run again with --apply to write it.')
            return
        self._apply(ic, renames)
        self.stdout.write(self.style.SUCCESS(f'Renamed {len(renames)} categories in {ic.identifier}.'))
        if ic.live_revision_id is not None:
            self.stdout.write('The instance is published: publish it again for visitors to get the fix.')

    def _plan(self, ic: InstanceConfig, explicit: list[str], *, auto: bool) -> tuple[list[Rename], list[str]]:
        scopes = {
            s.identifier: s
            for s in DimensionScope.objects.for_instance_config(ic).select_related('dimension')
            if s.identifier is not None
        }
        wanted: dict[tuple[str, str], str | None] = {}
        if auto:
            wanted.update(_invalid_categories(scopes))
        parsed, errors = _parse_renames(explicit)
        wanted.update(parsed)

        renames: list[Rename] = []
        taken: dict[str, set[str]] = {}
        for (dim_id, old), new in wanted.items():
            scope = scopes.get(dim_id)
            if scope is None:
                errors.append(f'{dim_id}: no such dimension in {ic.identifier}')
                continue
            cat = scope.dimension.categories.filter(identifier=old).first()
            if cat is None:
                errors.append(f'{dim_id}: no category {old!r}')
                continue
            if dim_id not in taken:
                taken[dim_id] = {i for i in scope.dimension.categories.values_list('identifier', flat=True) if i is not None}
            target, error = _target_identifier(old, new, taken[dim_id])
            if error is not None:
                errors.append(f'{dim_id}: {error}')
                continue
            taken[dim_id].add(target)
            renames.append(Rename(scope, cat, old, target))
        return renames, errors

    @staticmethod
    def _find_references(ic: InstanceConfig, renames: list[Rename]) -> None:
        texts: list[tuple[str, str]] = [
            (f'node {nc.identifier}', nc.spec.model_dump_json())
            for nc in NodeConfig.objects.filter(instance=ic).exclude(spec__isnull=True)
            if nc.spec is not None
        ]
        texts.extend(
            (f'binding into node {b.node.identifier}', _dump([t.model_dump(mode='json') for t in b.transformations]))
            for b in NodeInputPortBinding.objects.filter(instance=ic).select_related('node')
        )
        if ic.spec is not None:
            rest_of_spec = ic.spec.model_dump(mode='json', exclude={'dimensions'})
            texts.append(('instance spec (scenarios, parameters, ...)', _dump(rest_of_spec)))
        for r in renames:
            needle = _dump(r.old)
            r.references = [where for where, text in texts if needle in text]

    def _report(self, ic: InstanceConfig, renames: list[Rename], errors: list[str]) -> None:
        self.stdout.write(f'{ic.identifier} ({ic.config_source}):')
        for r in renames:
            self.stdout.write(f'  {r.scope.identifier}: {r.old!r} -> {r.new!r}  ({r.category.uuid})')
            for where in r.references:
                self.stdout.write(f'      still named in {where}')
        for e in errors:
            self.stderr.write(f'  ERROR {e}')

    @staticmethod
    def _apply(ic: InstanceConfig, renames: list[Rename]) -> None:
        spec_dims = {d.get('id') for d in ic.spec.dimensions} if ic.spec is not None else set()
        with transaction.atomic():
            for r in renames:
                DimensionCategory.objects.filter(pk=r.category.pk).update(identifier=r.new)
            synced: set[int] = set()
            for r in renames:
                if r.scope.pk in synced or r.scope.identifier not in spec_dims:
                    continue
                InstanceEditorMutation._sync_spec_dimension_from_orm(ic, r.scope)
                synced.add(r.scope.pk)
        ic.invalidate_cache()
