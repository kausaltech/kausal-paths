"""
Renaming a dataset identifier in place.

The promise is that the rename is invisible to the model graph: bindings, ports, UUIDs
and data points all stay attached to the same row, which now answers to the new name.
These tests hold that promise, and hold the refusals that keep it honest -- above all
that the whole set is refused rather than half-applied, since a namespace that is
partly renamed is worse than one that has not moved.
"""

from uuid import UUID

from django.core.management import call_command
from django.core.management.base import CommandError

import polars as pl
import pytest

from kausal_common.datasets.models import Dataset, DatasetMetric

from nodes.management.commands.load_dvc_dataset import Command as LoadCommand
from nodes.management.commands.rename_dataset import build_rename_plan, load_mapping
from nodes.models import NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.tests.test_load_dvc_dataset_refresh import make_context

pytestmark = pytest.mark.django_db

OLD = 'bisko/final_energy'
NEW = 'kommune/endenergieverbrauch'


def _import(ic, ds_id: str = OLD, commit: str = 'aaa111') -> Dataset:
    LoadCommand().sync_dataset(
        ic,
        make_context(pl.DataFrame({'Year': [2020, 2021], 'Value': [1.0, 2.0]}), {'Value': 'MWh/a'}, commit=commit),
        ds_id,
    )
    return Dataset.objects.get(identifier=ds_id, scope_id=ic.pk)


def _bind(ic, dataset: Dataset) -> NodeInputPortBinding:
    metric = DatasetMetric.objects.filter(schema=dataset.schema).first()
    node = NodeConfigFactory.create(instance=ic)
    return NodeInputPortBinding.objects.create(
        instance=ic,
        node=node,
        port_id=UUID('44444444-4444-4444-4444-444444444444'),
        dataset=dataset,
        metric=metric,
    )


# --- The promise ------------------------------------------------------------------------


def test_rename_keeps_the_row_its_uuid_and_its_bindings():
    ic = InstanceConfigFactory.create(name='rename-keeps', config_source='database')
    dataset = _import(ic)
    binding = _bind(ic, dataset)
    pk, uuid, points = dataset.pk, dataset.uuid, dataset.data_points.count()

    call_command('rename_dataset', OLD, NEW, '--apply')

    dataset.refresh_from_db()
    assert dataset.identifier == NEW
    assert dataset.pk == pk
    assert dataset.uuid == uuid  # anything that pinned the UUID still resolves
    assert dataset.data_points.count() == points
    binding.refresh_from_db()
    assert binding.dataset_id == pk  # the binding never moved


def test_a_plan_without_apply_writes_nothing():
    ic = InstanceConfigFactory.create(name='rename-planonly', config_source='database')
    _import(ic)

    call_command('rename_dataset', OLD, NEW)

    assert Dataset.objects.filter(identifier=OLD, scope_id=ic.pk).exists()
    assert not Dataset.objects.filter(identifier=NEW).exists()


def test_external_ref_is_restamped_so_provenance_names_the_new_path():
    ic = InstanceConfigFactory.create(name='rename-ref', config_source='database')
    dataset = _import(ic)
    assert (dataset.external_ref or {})['dataset_id'] == OLD

    call_command('rename_dataset', OLD, NEW, '--apply')

    dataset.refresh_from_db()
    ref = dataset.external_ref
    assert ref is not None
    assert ref['dataset_id'] == NEW
    # The commit is provenance about the data, not about the name, and must survive.
    assert ref['commit'] == 'aaa111'


def test_a_foreign_external_ref_is_left_alone():
    """Only a stamp naming the old identifier is ours to rewrite."""
    ic = InstanceConfigFactory.create(name='rename-foreign-ref', config_source='database')
    dataset = _import(ic)
    imported_ref = dataset.external_ref
    assert imported_ref is not None
    dataset.external_ref = {**imported_ref, 'dataset_id': 'someone_else/thing'}
    dataset.save(update_fields=['external_ref'])

    call_command('rename_dataset', OLD, NEW, '--apply')

    dataset.refresh_from_db()
    assert dataset.identifier == NEW
    ref = dataset.external_ref
    assert ref is not None
    assert ref['dataset_id'] == 'someone_else/thing'


def test_every_scope_holding_the_identifier_is_renamed():
    """The rename is global: the config will name the new identifier for all instances."""
    first = InstanceConfigFactory.create(name='rename-scope-a', config_source='database')
    second = InstanceConfigFactory.create(name='rename-scope-b', config_source='database')
    _import(first)
    _import(second)

    plan = build_rename_plan(OLD, NEW)
    assert len(plan.rows) == 2
    assert {row.scope for row in plan.rows} == {first.identifier, second.identifier}

    call_command('rename_dataset', OLD, NEW, '--apply')

    assert Dataset.objects.filter(identifier=OLD).count() == 0
    assert Dataset.objects.filter(identifier=NEW).count() == 2


def test_set_name_changes_the_display_name_of_a_single_row():
    ic = InstanceConfigFactory.create(name='rename-setname', config_source='database')
    dataset = _import(ic)

    call_command('rename_dataset', OLD, NEW, '--set-name', 'Endenergieverbrauch', '--apply')

    dataset.refresh_from_db()
    assert dataset.schema is not None
    assert dataset.schema.name == 'Endenergieverbrauch'


def test_a_label_applies_to_every_row_of_the_rename():
    """
    The schema is one-to-one with the dataset, so each city's row has its own schema.

    They are the same logical dataset, so they take the same label -- setting it on one
    and leaving the others would be the bug.
    """
    first = InstanceConfigFactory.create(name='rename-label-a', config_source='database')
    second = InstanceConfigFactory.create(name='rename-label-b', config_source='database')
    one = _import(first)
    two = _import(second)

    call_command('rename_dataset', OLD, NEW, '--set-name', 'End energy use', '--apply')

    for dataset in (one, two):
        dataset.refresh_from_db()
        assert dataset.schema is not None
        assert dataset.schema.name == 'End energy use'


# --- The refusals -----------------------------------------------------------------------


def test_a_target_taken_in_the_same_scope_is_refused():
    ic = InstanceConfigFactory.create(name='rename-clash', config_source='database')
    _import(ic, OLD)
    _import(ic, NEW)

    plan = build_rename_plan(OLD, NEW)

    assert any('already has a dataset called' in b for b in plan.blockers)
    assert not plan.is_actionable


def test_the_same_target_in_a_different_scope_is_not_a_clash():
    """Two cities may legitimately both hold the new name."""
    first = InstanceConfigFactory.create(name='rename-noclash-a', config_source='database')
    second = InstanceConfigFactory.create(name='rename-noclash-b', config_source='database')
    _import(first, OLD)
    _import(second, NEW)

    plan = build_rename_plan(OLD, NEW)

    assert plan.is_actionable
    assert [row.scope for row in plan.rows] == [first.identifier]


def test_an_unknown_identifier_is_refused_rather_than_silently_doing_nothing():
    plan = build_rename_plan('bisko/typo', NEW)
    assert any('no dataset row carries this identifier' in b for b in plan.blockers)

    tolerated = build_rename_plan('bisko/typo', NEW, allow_missing=True)
    assert tolerated.blockers == []
    assert tolerated.rows == []
    assert not tolerated.is_actionable  # a no-op, not something to write


@pytest.mark.parametrize(
    ('bad', 'label'),
    [
        ('de/fernwärme', 'umlaut'),
        ('de/Endenergie', 'uppercase'),
        ('endenergie', 'no-namespace'),
        ('de/energie verbrauch', 'space'),
    ],
)
def test_an_invalid_target_identifier_is_refused(bad: str, label: str):
    ic = InstanceConfigFactory.create(name=f'rename-bad-{label}', config_source='database')
    _import(ic)

    plan = build_rename_plan(OLD, bad)

    assert any('not namespace/name' in b for b in plan.blockers)


def test_an_overlong_target_identifier_is_refused():
    ic = InstanceConfigFactory.create(name='rename-toolong', config_source='database')
    _import(ic)

    plan = build_rename_plan(OLD, 'kommune/' + 'x' * 100)

    assert any('the column holds' in b for b in plan.blockers)


def test_renaming_to_itself_is_refused():
    plan = build_rename_plan(OLD, OLD)
    assert any('same identifier' in b for b in plan.blockers)


def test_a_blocked_entry_stops_the_whole_set(tmp_path):
    """
    One bad entry refuses the good ones too.

    A namespace that is half renamed is harder to reason about than one that has not
    moved, so the set is all-or-nothing.
    """
    ic = InstanceConfigFactory.create(name='rename-allornothing', config_source='database')
    _import(ic, 'bisko/energy_shares')
    _import(ic, OLD)
    path = tmp_path / 'renames.yaml'
    path.write_text(f'bisko/energy_shares: de/energieanteile_verkehr\n{OLD}: de/Bad Name\n')

    with pytest.raises(CommandError, match='refused'):
        call_command('rename_dataset', '--from-file', str(path), '--apply')

    assert Dataset.objects.filter(identifier='bisko/energy_shares').exists()
    assert Dataset.objects.filter(identifier=OLD).exists()
    assert not Dataset.objects.filter(identifier='de/energieanteile_verkehr').exists()


def test_a_whole_mapping_file_applies_together(tmp_path):
    ic = InstanceConfigFactory.create(name='rename-fromfile', config_source='database')
    _import(ic, 'bisko/energy_shares')
    _import(ic, OLD)
    path = tmp_path / 'renames.yaml'
    path.write_text(f'bisko/energy_shares: de/energieanteile_verkehr\n{OLD}: {NEW}\n')

    call_command('rename_dataset', '--from-file', str(path), '--apply')

    assert Dataset.objects.filter(identifier='de/energieanteile_verkehr').exists()
    assert Dataset.objects.filter(identifier=NEW).exists()
    assert not Dataset.objects.filter(identifier__startswith='bisko/').exists()


def test_a_pair_and_a_file_together_are_refused(tmp_path):
    path = tmp_path / 'renames.yaml'
    path.write_text(f'{OLD}: {NEW}\n')

    with pytest.raises(CommandError, match='not both'):
        call_command('rename_dataset', OLD, NEW, '--from-file', str(path))


def test_a_lone_identifier_without_a_target_is_refused():
    with pytest.raises(CommandError, match='Name both OLD and NEW'):
        call_command('rename_dataset', OLD)


# --- The mapping file -------------------------------------------------------------------


def test_many_sources_may_map_onto_one_target_in_disjoint_scopes(tmp_path):
    """
    Retiring dataset_replacements is exactly this shape.

    Every city's own `<city>/final_energy` becomes the one `kommune/` name, each in its
    own scope, so many-to-one is intended rather than a typo.
    """
    first = InstanceConfigFactory.create(name='rename-many-a', config_source='database')
    second = InstanceConfigFactory.create(name='rename-many-b', config_source='database')
    _import(first, 'mainz/final_energy')
    _import(second, 'duesseldorf/final_energy')
    path = tmp_path / 'renames.yaml'
    path.write_text(f'mainz/final_energy: {NEW}\nduesseldorf/final_energy: {NEW}\n')

    call_command('rename_dataset', '--from-file', str(path), '--apply')

    assert Dataset.objects.filter(identifier=NEW).count() == 2
    assert not Dataset.objects.filter(identifier__endswith='/final_energy').exists()


def test_two_sources_landing_on_one_target_in_one_scope_are_refused(tmp_path):
    """The per-row clash check cannot see this: neither target exists yet."""
    ic = InstanceConfigFactory.create(name='rename-many-clash', config_source='database')
    _import(ic, 'mainz/final_energy')
    _import(ic, 'duesseldorf/final_energy')
    path = tmp_path / 'renames.yaml'
    path.write_text(f'mainz/final_energy: {NEW}\nduesseldorf/final_energy: {NEW}\n')

    with pytest.raises(CommandError, match='refused'):
        call_command('rename_dataset', '--from-file', str(path), '--apply')

    assert Dataset.objects.filter(identifier='mainz/final_energy').exists()
    assert Dataset.objects.filter(identifier='duesseldorf/final_energy').exists()


def test_mapping_file_reads_the_short_form(tmp_path):
    path = tmp_path / 'renames.yaml'
    path.write_text(f'{OLD}: {NEW}\nbisko/energy_shares: de/energieanteile_verkehr\n')

    mapping = load_mapping(path)

    assert {old: entry.to for old, entry in mapping.items()} == {
        OLD: NEW,
        'bisko/energy_shares': 'de/energieanteile_verkehr',
    }
    assert all(not entry.labels for entry in mapping.values())


def test_mapping_file_reads_the_long_form_with_labels(tmp_path):
    path = tmp_path / 'renames.yaml'
    path.write_text(f'{OLD}:\n  to: {NEW}\n  name_de: Endenergieverbrauch\n  name_en: End energy use\n')

    entry = load_mapping(path)[OLD]

    assert entry.to == NEW
    assert entry.labels == {'de': 'Endenergieverbrauch', 'en': 'End energy use'}


def test_mapping_file_refuses_an_unrecognised_key(tmp_path):
    path = tmp_path / 'renames.yaml'
    path.write_text(f'{OLD}:\n  to: {NEW}\n  nmae_de: typo\n')

    with pytest.raises(CommandError, match='unrecognised key'):
        load_mapping(path)


def test_mapping_file_refuses_a_table_without_a_target(tmp_path):
    path = tmp_path / 'renames.yaml'
    path.write_text(f'{OLD}:\n  name_de: Endenergieverbrauch\n')

    with pytest.raises(CommandError, match="needs a 'to' identifier"):
        load_mapping(path)


# --- Labels -----------------------------------------------------------------------------


def test_labels_are_split_the_way_modeltrans_reads_them_back(tmp_path):
    """
    The default language's value belongs in the plain column, the rest in ``i18n``.

    Storing German in the column while the site default is English is what makes an
    English reader see German, which is the bug these labels exist to avoid.
    """
    ic = InstanceConfigFactory.create(name='rename-i18n', config_source='database')
    dataset = _import(ic)
    path = tmp_path / 'renames.yaml'
    path.write_text(f'{OLD}:\n  to: {NEW}\n  name_en: End energy use\n  name_de: Endenergieverbrauch\n')

    call_command('rename_dataset', '--from-file', str(path), '--apply')

    dataset.refresh_from_db()
    schema = dataset.schema
    assert schema is not None
    assert schema.name == 'End energy use'  # settings.LANGUAGE_CODE is 'en'
    assert schema.i18n.get('name_de') == 'Endenergieverbrauch'


def test_labels_without_the_default_language_are_refused():
    """Otherwise the column keeps a stale label while the translations move on."""
    plan = build_rename_plan(OLD, NEW, labels={'de': 'Endenergieverbrauch'})

    assert any('name_en' in b for b in plan.blockers)


def test_a_rename_without_labels_leaves_the_existing_one_alone():
    ic = InstanceConfigFactory.create(name='rename-nolabel', config_source='database')
    dataset = _import(ic)
    before = dataset.schema.name if dataset.schema else None

    call_command('rename_dataset', OLD, NEW, '--apply')

    dataset.refresh_from_db()
    assert dataset.schema is not None
    assert dataset.schema.name == before


def test_labels_can_be_set_without_changing_the_identifier():
    """A pure label pass is legitimate: same name in, same name out."""
    ic = InstanceConfigFactory.create(name='rename-labelonly', config_source='database')
    dataset = _import(ic)

    call_command('rename_dataset', OLD, OLD, '--set-name', 'End energy use', '--apply')

    dataset.refresh_from_db()
    assert dataset.identifier == OLD
    assert dataset.schema is not None
    assert dataset.schema.name == 'End energy use'


def test_mapping_file_refuses_an_empty_or_non_mapping_document(tmp_path):
    path = tmp_path / 'renames.yaml'
    path.write_text('[]\n')

    with pytest.raises(CommandError, match='non-empty mapping'):
        load_mapping(path)
