"""
Which datasets a restamp run touches, and which it must leave alone.

`restamp_dataset_sources` attaches `metadata['sources']` to published DVC datasets that have
none. Its selection logic is the part worth testing without a repository: everything else is
read-modify-write against DVC.
"""

import pytest

from nodes.constants import SOURCE_TARGET_DATA_POINT, SOURCE_TARGET_DATASET
from tools.restamp_dataset_sources import dataset_level_sources, requested_identifiers
from tools.upload_new_dataset import SourceRegistryEntry

# The functions under test are pure, but importing the uploader pulls in Django models.
pytestmark = pytest.mark.django_db


def _entry(target, datasets=None, **fields):
    return SourceRegistryEntry(
        fields={'target': target, **fields},
        datasets=frozenset(datasets) if datasets else None,
    )


REGISTRY = {
    'ifeu': _entry(SOURCE_TARGET_DATASET, ['de/emissionsfaktoren_erzeugung'], authority='ifeu'),
    'Lokale Kopie': _entry(SOURCE_TARGET_DATASET, ['kommune/emissionsfaktoren_erzeugung'], authority='Kommune'),
    'Verkehrsdaten': _entry(
        SOURCE_TARGET_DATASET,
        ['de/fahrleistung_strassenverkehr', 'de/energieverbrauch_uebriger_verkehr'],
        authority='ifeu',
    ),
    'Zeilenquelle': _entry(SOURCE_TARGET_DATA_POINT, authority='ifeu'),
}


def test_the_registry_drives_the_run_when_no_identifiers_are_given():
    # Order follows the registry and carries no meaning; the set is what is asserted.
    assert sorted(requested_identifiers(REGISTRY, [])) == [
        'de/emissionsfaktoren_erzeugung',
        'de/energieverbrauch_uebriger_verkehr',
        'de/fahrleistung_strassenverkehr',
        'kommune/emissionsfaktoren_erzeugung',
    ]


def test_a_data_point_source_is_never_restamped():
    """Row-level provenance is placed by an upload's `Source` cells, not by this tool."""
    assert 'Zeilenquelle' not in {s['name'] for s in dataset_level_sources(REGISTRY, 'de/emissionsfaktoren_erzeugung')}


def test_skip_removes_the_external_dvc_placeholders():
    """
    The two BISKO transport tables are read straight from DVC and never imported.

    `metadata['sources']` only becomes `DataSource` rows through `load_dvc_dataset`, so a stamp
    on a dataset nothing imports reaches nothing -- while the push rewrites and re-uploads the
    whole parquet, and those two are 2.35M and 1.23M rows. Their provenance still belongs in the
    registry, so they are skipped here rather than deleted from it.
    """
    skip = ['de/fahrleistung_strassenverkehr', 'de/energieverbrauch_uebriger_verkehr']

    assert sorted(requested_identifiers(REGISTRY, [], skip)) == [
        'de/emissionsfaktoren_erzeugung',
        'kommune/emissionsfaktoren_erzeugung',
    ]
    # And when the run is named explicitly, so a --skip cannot be defeated by listing it.
    assert requested_identifiers(REGISTRY, [*skip, 'de/emissionsfaktoren_erzeugung'], skip) == ['de/emissionsfaktoren_erzeugung']


def test_a_source_bound_to_no_dataset_cannot_drive_a_run(capsys):
    """
    A dataset-level source with an empty `Datasets` column applies to *every* dataset.

    That is right when it is asked about one -- a single-dataset upload has nothing to choose
    between -- and useless as the thing that decides which datasets to visit, since it names
    none. So it cannot drive a run, and the run says so rather than passing over it.
    """
    registry = {**REGISTRY, 'Ungebunden': _entry(SOURCE_TARGET_DATASET, authority='ifeu')}

    assert 'Ungebunden' not in requested_identifiers(registry, [])
    assert 'Ungebunden' in capsys.readouterr().out
    # It does still attach once a dataset is named, which is why it is worth keeping separate
    # from the registry the other cases use.
    assert 'Ungebunden' in {s['name'] for s in dataset_level_sources(registry, 'de/emissionsfaktoren_erzeugung')}


def test_the_namespace_decides_which_source_a_shared_leaf_name_gets():
    """Six BISKO leaf names exist in both `de/` and `kommune/`; the leaf alone cannot choose."""
    assert [s['name'] for s in dataset_level_sources(REGISTRY, 'de/emissionsfaktoren_erzeugung')] == ['ifeu']
    assert [s['name'] for s in dataset_level_sources(REGISTRY, 'kommune/emissionsfaktoren_erzeugung')] == ['Lokale Kopie']
