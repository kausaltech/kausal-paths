"""
Rename a dataset's identifier in place, across every scope that holds it.

The BISKO datasets are being renamed wholesale (``bisko/*`` -> ``de/*`` and
``kommune/*``), which is a three-sided change: the DVC paths, the ``Dataset`` rows in
the database, and every reference in ``configs/``. This command owns the middle side,
which is the only one that cannot be done with a text editor.

Renaming in place rather than recreating matters because the graph references datasets
*by row*, not by name. ``NodeInputPortBinding.dataset`` and ``NodeDataset.dataset`` are
foreign keys, no binding's ``dataset_spec`` embeds an identifier, and node and instance
specs carry none either -- so an in-place rename keeps every binding, every pinned UUID
and every published revision intact, and the rename cannot be observed by the model
graph at all. Deleting and re-importing under the new name would instead mint a new
UUID and break each of those.

What moves with the identifier:

* ``Dataset.identifier``.
* ``Dataset.external_ref['dataset_id']``, when it names the old identifier. This is the
  DVC provenance stamp, so leaving it behind would make the row claim it came from a
  path that no longer exists. The DVC data has to be moved separately; this only
  records where it now lives.

What deliberately does not move:

* ``DatasetSchema.name`` -- the display name, which is not derived from the identifier
  and is often better than it ('Endenergie' for ``bisko/final_energy``). Pass
  ``--set-name`` to change it for a single rename, or leave it and treat display names
  as their own pass.
* ``InstanceRevisionDatasetPin.identifier`` -- a denormalized record of what the
  dataset was *called when that revision was published*. The pin's identity is its
  foreign key and ``dataset_uuid``; nothing resolves a pin by identifier. Rewriting it
  would falsify the retention manifest, so the pins are reported and left alone.

Usage:

    python manage.py rename_dataset bisko/final_energy kommune/endenergieverbrauch
    python manage.py rename_dataset bisko/final_energy kommune/endenergieverbrauch --apply
    python manage.py rename_dataset --from-file data/bisko/renames.yaml --apply

A mapping entry is either a bare target identifier or a table that also carries labels:

    bisko/energy_shares: de/energieanteile_verkehr

    bisko/endenergie_emissionsfaktoren:
      to: de/emissionsfaktoren_endenergie
      name_de: Emissionsfaktoren Endenergie
      name_en: End energy emission factors

Labels are stored the way ``modeltrans`` expects: the value for the *default* language
(``settings.LANGUAGE_CODE``) goes in the ``name`` column and the rest into ``i18n``. An
entry naming labels must include the default language, or the column would keep a stale
value while the translations moved on -- which is how the existing dimension categories
came to hold German in a column that is read as English.

Nothing is written without ``--apply``, and everything an ``--apply`` does happens in
one transaction, so a blocked rename cannot leave the set half-renamed.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

import ruamel.yaml
from rich import print

from kausal_common.datasets.models import Dataset
from kausal_common.i18n.pydantic import TranslatedString, get_modeltrans_attrs_from_str

from nodes.models import InstanceConfig, InstanceRevisionDatasetPin, NodeDataset, NodeInputPortBinding

if TYPE_CHECKING:
    from argparse import ArgumentParser

#: Dataset identifiers are ``namespace/name`` built from the same character class the
#: rest of the codebase uses for identifiers (``IdentifierValidator``): lowercase
#: ASCII, digits, hyphen and underscore. German names therefore have to be
#: transliterated -- ``fernwaerme``, not ``fernwärme``.
IDENTIFIER_RE = re.compile(r'^[a-z0-9_-]+(/[a-z0-9_-]+)+$')

#: ``Dataset.identifier`` is a CharField of this length; a longer name is refused up
#: front rather than truncated by the database.
MAX_IDENTIFIER_LENGTH = 100

#: Where to look for lingering references once the database side has moved.
CONFIG_ROOT = Path('configs')


@dataclass
class RowPlan:
    """One ``Dataset`` row that will be renamed."""

    pk: int
    uuid: str
    scope: str
    data_points: int
    bindings: int
    external_ref_id: str | None
    """The ``dataset_id`` currently stamped in ``external_ref``, when there is one."""
    pins: int


@dataclass
class EmptyTarget:
    """An existing target row that holds no data, so the source can take its place."""

    pk: int
    scope: str
    bindings: int
    node_datasets: int


@dataclass
class RenamePlan:
    """What one identifier rename affects, and why it may be refused."""

    old: str
    new: str
    rows: list[RowPlan] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    """Language code -> human-readable label, empty when the entry names none."""
    replace: list[EmptyTarget] = field(default_factory=list)
    """Empty target rows to delete first, so the source can be renamed into their place."""

    @property
    def is_actionable(self) -> bool:
        return bool(self.rows or self.replace) and not self.blockers

    @property
    def external_ref_updates(self) -> int:
        return sum(1 for row in self.rows if row.external_ref_id == self.old)


def _scope_label(dataset: Dataset) -> str:
    scope = dataset.scope
    if scope is None:
        return '(unscoped)'
    if isinstance(scope, InstanceConfig):
        return scope.identifier
    return str(scope)


def default_language() -> str:
    """
    Return the language whose value lives in the model's own column rather than in ``i18n``.

    ``modeltrans`` reads the plain field for the active language when it *is* the default,
    so this is the one label that cannot be left behind.
    """
    return settings.LANGUAGE_CODE.split('-')[0].lower()


def _validate_labels(labels: dict[str, str], plan: RenamePlan) -> None:
    if not labels:
        return
    lang = default_language()
    if lang not in labels:
        plan.blockers.append(
            f'labels given without name_{lang}; the {lang!r} value lives in the name column '
            'itself, so omitting it would leave a stale label behind'
        )
    for value in labels.values():
        if not value.strip():
            plan.blockers.append('a label is empty')


def _validate_new_identifier(new: str, plan: RenamePlan) -> None:
    if len(new) > MAX_IDENTIFIER_LENGTH:
        plan.blockers.append(f'target identifier is {len(new)} characters; the column holds {MAX_IDENTIFIER_LENGTH}')
    if not IDENTIFIER_RE.match(new):
        plan.blockers.append(
            f'target identifier {new!r} is not namespace/name in [a-z0-9_-] (umlauts must be transliterated: ae, oe, ue, ss)'
        )


def _classify_clash(plan: RenamePlan, dataset: Dataset, clash: Dataset, *, replace_empty_target: bool) -> None:
    """
    Decide whether an existing target row blocks the rename or can be stood aside.

    ``sync_instance_to_db`` run *before* the rename creates a row for every identifier the
    deployed config names, so the scope ends up holding both the old row (with the data) and a
    new empty one (with the bindings, because the spec named it). The empty row is not content:
    it is a placeholder the sync minted, and the data it should describe is still in the row
    being renamed. Deleting it and renaming the source into its place restores the intended
    state; re-running the sync then rebuilds the bindings.

    Only ever for a target with **no data points**. A target holding data is a genuine
    collision, and merging two datasets is not this command's business.
    """
    points = clash.data_points.count()
    scope = _scope_label(dataset)
    if points:
        plan.blockers.append(
            f'{scope} already has a dataset called {plan.new!r} (pk {clash.pk}) holding '
            f'{points} data point(s); renaming would violate unique_identifier_per_dataset_scope'
        )
        return
    # A published revision pins rows by foreign key and its manifest records what that
    # revision used, so deleting a pinned row would falsify history. Refused outright.
    pins = InstanceRevisionDatasetPin.objects.filter(dataset=clash).count()
    if pins:
        plan.blockers.append(
            f'{scope} has an empty {plan.new!r} (pk {clash.pk}) but {pins} published revision '
            'pin(s) reference it; it cannot be removed without rewriting that history'
        )
        return
    if not replace_empty_target:
        plan.blockers.append(
            f'{scope} already has an empty dataset called {plan.new!r} (pk {clash.pk}), '
            'probably from a sync that ran before this rename. Pass --replace-empty-target to '
            'delete it and rename this row into its place.'
        )
        return
    plan.replace.append(
        EmptyTarget(
            pk=clash.pk,
            scope=scope,
            bindings=NodeInputPortBinding.objects.filter(dataset=clash).count(),
            node_datasets=NodeDataset.objects.filter(dataset=clash).count(),
        )
    )


def build_rename_plan(
    old: str,
    new: str,
    *,
    labels: dict[str, str] | None = None,
    allow_missing: bool = False,
    replace_empty_target: bool = False,
) -> RenamePlan:
    """Work out what renaming ``old`` to ``new`` would do, without touching anything."""
    plan = RenamePlan(old=old, new=new, labels=dict(labels or {}))
    if old == new and not plan.labels:
        plan.blockers.append('source and target are the same identifier')
        return plan
    if old != new:
        _validate_new_identifier(new, plan)
    _validate_labels(plan.labels, plan)

    datasets = list(Dataset.objects.filter(identifier=old).select_related('schema', 'scope_content_type'))
    if not datasets and not allow_missing:
        plan.blockers.append('no dataset row carries this identifier (typo, or already renamed?)')
        return plan

    for dataset in datasets:
        # The unique constraint is (identifier, scope_content_type, scope_id), so a
        # collision is only a collision within the same scope. Checking globally would
        # refuse the normal case, where the target name is meant to exist once per city.
        clash = (
            Dataset.objects
            .filter(
                identifier=new,
                scope_content_type=dataset.scope_content_type,
                scope_id=dataset.scope_id,
            )
            .exclude(pk=dataset.pk)
            .first()
        )
        if clash is not None:
            _classify_clash(plan, dataset, clash, replace_empty_target=replace_empty_target)
        external_ref = dataset.external_ref or {}
        plan.rows.append(
            RowPlan(
                pk=dataset.pk,
                uuid=str(dataset.uuid),
                scope=_scope_label(dataset),
                data_points=dataset.data_points.count(),
                bindings=NodeInputPortBinding.objects.filter(dataset=dataset).count(),
                external_ref_id=external_ref.get('dataset_id') if isinstance(external_ref, dict) else None,
                pins=InstanceRevisionDatasetPin.objects.filter(dataset=dataset).count(),
            )
        )
    return plan


def check_cross_plan_collisions(plans: list[RenamePlan]) -> None:
    """
    Refuse two renames that would land the same identifier in the same scope.

    Several sources mapping onto one target is normal and intended: retiring the
    ``dataset_replacements`` indirection renames ``mainz/final_energy``,
    ``duesseldorf/final_energy`` and the rest onto one ``kommune/`` name, each in its own
    city's scope. What must not happen is two of them landing in the *same* scope, which
    the per-row clash check cannot see because neither target exists yet.
    """
    claimed: dict[tuple[str, str], str] = {}
    for plan in plans:
        for row in plan.rows:
            key = (plan.new, row.scope)
            if key in claimed:
                plan.blockers.append(f'{claimed[key]!r} and {plan.old!r} would both become {plan.new!r} in {row.scope}')
            else:
                claimed[key] = plan.old


def _write_labels(dataset: Dataset, labels: dict[str, str]) -> None:
    """
    Store the label on the dataset's schema, split the way ``modeltrans`` reads it back.

    ``get_modeltrans_attrs_from_str`` puts the default language's value in the plain
    column and the rest under ``name_<lang>`` keys, and converts the language codes to
    the format ``modeltrans`` expects -- which is the part that is easy to get wrong by
    hand, and is why this does not build the dict itself.
    """
    schema = dataset.schema
    assert schema is not None
    lang = default_language()
    translated = TranslatedString(**labels, default_language=lang)
    name, i18n = get_modeltrans_attrs_from_str(translated, 'name', lang)
    schema.name = name
    # Merge rather than replace: the schema's i18n may carry other translated fields.
    schema.i18n = {**(schema.i18n or {}), **i18n}
    schema.save(update_fields=['name', 'i18n'])


def print_rename_plan(plan: RenamePlan) -> None:
    """
    One line per row, kept inside a terminal width.

    The UUID is deliberately not shown: it does not change, so printing it per row says
    nothing the summary does not, and it pushes the line past the width every time.
    Flags are terse for the same reason, and explained underneath only when they appear.
    """
    print(f'\n[bold]{plan.old}[/bold] -> [bold]{plan.new}[/bold]')
    legend: set[str] = set()
    for row in plan.rows:
        flags = []
        if row.external_ref_id == plan.old:
            flags.append('ref')
            legend.add('  ref   = DVC provenance stamp restamped to the new identifier')
        elif row.external_ref_id is not None:
            flags.append('ref!')
            legend.add('  ref!  = external_ref names another dataset; left untouched')
        if row.pins:
            flags.append(f'pin:{row.pins}')
            legend.add('  pin:N = N published revision pins keep the old name, as the record of that publish')
        print(f'  {row.scope:<24} pk {row.pk:<6} {row.data_points:>6} pts  {row.bindings} bind  {" ".join(flags)}')
    for line in sorted(legend):
        print(f'[dim]{line}[/dim]')
    for target in plan.replace:
        print(
            f'  [yellow]replace[/yellow] {target.scope}: deleting empty pk {target.pk} first '
            f'({target.bindings} binding(s), {target.node_datasets} node-dataset(s) '
            'cleared; sync_instance_to_db rebuilds them)'
        )
    for lang, value in sorted(plan.labels.items()):
        print(f'  label ({lang}) -> {value!r}')
    for blocker in plan.blockers:
        print(f'  [red]refused:[/red] {blocker}')


def config_references(identifier: str) -> dict[str, int]:
    """Count remaining textual references per config file, so the operator knows what is left."""
    if not CONFIG_ROOT.is_dir():
        return {}
    found: dict[str, int] = {}
    for path in sorted(CONFIG_ROOT.rglob('*.yaml')):
        try:
            text = path.read_text()
        except OSError:
            continue
        count = text.count(identifier)
        if count:
            found[str(path)] = count
    return found


@dataclass
class MappingEntry:
    """One line of the mapping file: where the identifier goes, and what to call it."""

    to: str
    labels: dict[str, str] = field(default_factory=dict)


def _parse_entry(path: Path, old: str, value: object) -> MappingEntry:
    if isinstance(value, str):
        return MappingEntry(to=value)
    if not isinstance(value, dict):
        raise CommandError(f'{path}: {old!r} must map to an identifier or a table, got {value!r}')
    unknown = [key for key in value if key != 'to' and not key.startswith('name_')]
    if unknown:
        raise CommandError(f'{path}: {old!r} has unrecognised key(s): {", ".join(sorted(unknown))}')
    target = value.get('to')
    if not isinstance(target, str) or not target:
        raise CommandError(f"{path}: {old!r} needs a 'to' identifier")
    labels: dict[str, str] = {}
    for key, label in value.items():
        if not key.startswith('name_'):
            continue
        if not isinstance(label, str):
            raise CommandError(f'{path}: {old!r} label {key} must be a string, got {label!r}')
        labels[key.removeprefix('name_')] = label
    return MappingEntry(to=target, labels=labels)


def load_mapping(path: Path) -> dict[str, MappingEntry]:
    """
    Read the YAML mapping.

    An entry is either ``old: new`` or a table with ``to:`` and ``name_<lang>:`` keys; see
    the module docstring.
    """
    yaml = ruamel.yaml.YAML(typ='safe')
    try:
        raw = yaml.load(path.read_text())
    except OSError as exc:
        raise CommandError(f'Could not read {path}: {exc}') from exc
    if not isinstance(raw, dict) or not raw:
        raise CommandError(f'{path} must be a non-empty mapping of old identifier to new identifier')
    mapping: dict[str, MappingEntry] = {}
    for old, value in raw.items():
        if not isinstance(old, str):
            raise CommandError(f'{path}: every key must be an identifier string, got {old!r}')
        mapping[old] = _parse_entry(path, old, value)
    return mapping


def _delete_empty_targets(plan: RenamePlan) -> None:
    """
    Remove the empty rows a premature sync minted, so the real rows can take their names.

    The protected references are cleared rather than left: ``NodeInputPortBinding``
    and ``NodeDataset`` both point at the dataset with ``PROTECT``, and they
    describe a binding to a row that should never have existed. ``sync_instance_to_db``
    rebuilds them from the spec afterwards -- ``reconcile_input_bindings`` recreates the
    binding set on every run -- so nothing authored by hand is lost here.
    """
    for target in plan.replace:
        dataset = Dataset.objects.select_for_update().get(pk=target.pk)
        assert not dataset.data_points.exists(), 'an empty target must stay empty'
        NodeInputPortBinding.objects.filter(dataset=dataset).delete()
        NodeDataset.objects.filter(dataset=dataset).delete()
        schema = dataset.schema
        dataset.delete()
        # The schema is one-to-one with the dataset here, so a schema left with no datasets is
        # the sync's leftover too. Removing it keeps the admin's dataset list honest.
        if schema is not None and not schema.datasets.exists():
            schema.delete()
        print(f'{plan.old}: removed empty {plan.new!r} (pk {target.pk}) in {target.scope}')


class Command(BaseCommand):
    help = "Rename dataset identifiers in place, preserving each row's pk, UUID and bindings"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('old', nargs='?', help='Current dataset identifier')
        parser.add_argument('new', nargs='?', help='New dataset identifier')
        parser.add_argument(
            '--from-file',
            metavar='PATH',
            help='YAML file holding a flat "old: new" mapping, applied as one transaction',
        )
        parser.add_argument(
            '--set-name',
            metavar='NAME',
            help=(
                "Also set the schema's label, in the default language, on every row of this "
                'rename. Use the mapping file to give labels in more than one language.'
            ),
        )
        parser.add_argument(
            '--allow-missing',
            action='store_true',
            help='Treat an identifier with no rows as a no-op instead of refusing it',
        )
        parser.add_argument(
            '--replace-empty-target',
            action='store_true',
            help=(
                'When the target identifier already exists in a scope but holds no data -- as it '
                'does when sync_instance_to_db ran before this rename -- delete that empty row '
                'and rename this one into its place. Re-run sync_instance_to_db afterwards to '
                'rebuild the bindings.'
            ),
        )
        parser.add_argument('--apply', action='store_true', help='Write the renames (default is a plan only)')

    def _mapping(self, options: dict[str, Any]) -> dict[str, MappingEntry]:
        if options['from_file']:
            if options['old'] or options['new']:
                raise CommandError('Give either --from-file or an OLD NEW pair, not both')
            return load_mapping(Path(options['from_file']))
        if not options['old'] or not options['new']:
            raise CommandError('Name both OLD and NEW, or pass --from-file')
        return {options['old']: MappingEntry(to=options['new'])}

    def _apply(self, plans: list[RenamePlan]) -> None:
        with transaction.atomic():
            for plan in plans:
                _delete_empty_targets(plan)
                for row in plan.rows:
                    dataset = Dataset.objects.select_for_update().get(pk=row.pk)
                    fields = ['identifier']
                    dataset.identifier = plan.new
                    external_ref = dataset.external_ref
                    if isinstance(external_ref, dict) and external_ref.get('dataset_id') == plan.old:
                        dataset.external_ref = {**external_ref, 'dataset_id': plan.new}
                        fields.append('external_ref')
                    dataset.save(update_fields=fields)
                    if plan.labels and dataset.schema is not None:
                        _write_labels(dataset, plan.labels)
                    print(f'{plan.old} -> {plan.new} ({row.scope}, pk {row.pk})')

    def handle(self, *args: Any, **options: Any) -> None:
        mapping = self._mapping(options)
        plans = [
            build_rename_plan(
                old,
                entry.to,
                labels={default_language(): options['set_name']} if options['set_name'] else entry.labels,
                allow_missing=options['allow_missing'],
                replace_empty_target=options['replace_empty_target'],
            )
            for old, entry in mapping.items()
        ]
        check_cross_plan_collisions(plans)
        for plan in plans:
            print_rename_plan(plan)

        actionable = [p for p in plans if p.is_actionable]
        blocked = [p for p in plans if p.blockers]
        rows = sum(len(p.rows) for p in actionable)
        refs = sum(p.external_ref_updates for p in actionable)

        if blocked:
            # Refuse the whole set rather than apply the part that works: a half-renamed
            # namespace is harder to reason about than one that has not moved.
            raise CommandError(f'{len(blocked)} rename(s) refused; nothing was written. See above.')

        if not actionable:
            print('\n[yellow]Nothing to rename.[/yellow]')
            return

        if not options['apply']:
            print(
                f'\n[bold]Plan:[/bold] {len(actionable)} identifier(s), {rows} row(s), '
                f'{refs} external_ref stamp(s), '
                f'{sum(len(p.replace) for p in actionable)} empty target(s) removed. '
                'Every renamed row keeps its pk, UUID, data points, '
                'bindings and ports. Re-run with --apply to write.'
            )
            self._report_config_references(actionable)
            return

        self._apply(actionable)
        print(f'\n[green]Renamed {len(actionable)} identifier(s) across {rows} row(s).[/green]')
        self._report_config_references(actionable)

    def _report_config_references(self, plans: list[RenamePlan]) -> None:
        """Name the config files that still say the old identifier."""
        pending: dict[str, dict[str, int]] = {}
        for plan in plans:
            found = config_references(plan.old)
            if found:
                pending[plan.old] = found
        if not pending:
            return
        print('\n[yellow]Still referenced in configs (the database side alone is not enough):[/yellow]')
        for old, files in sorted(pending.items()):
            for path, count in sorted(files.items()):
                print(f'  {old} x{count}  {path}')
        print(
            '\n[dim]Check each before changing it: a reference can be deliberate. An instance '
            'pinned to a commit that predates the new path must keep the old name, and a '
            'deferred rename keeps it on purpose.[/dim]\n'
            'Once the intended ones are updated, re-run [bold]sync_instance_to_db[/bold] for '
            'database-sourced instances and [bold]dataset_status[/bold] to confirm nothing went stale.'
        )
