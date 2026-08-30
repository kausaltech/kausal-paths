"""
The upload side of dataset-level data sources: registry CSV -> metadata['sources'].

A dataset whose provenance is uniform -- one publication, one update, nothing to say per
row -- has no 'Source' column at all, so these entries cannot be discovered from the data.
They come from the sources registry's `Target` column and have to survive a code path that
used to return early whenever no citations were found.
"""

import polars as pl
import pytest

from nodes.constants import SOURCE_TARGET_DATASET
from tools.upload_new_dataset import (
    SourceRegistryEntry,
    build_sources_metadata,
    check_registry_dataset_names,
    load_sources_registry,
)

# The functions under test are pure, but importing the uploader pulls in Django models.
pytestmark = pytest.mark.django_db

REGISTRY_CSV = """Name,Authority,URL,Description,Edition,Target,Datasets,RAG Rating
Energiebilanz,StaLa,https://example.com/eb,Regional energy balance,2024,dataset,,Green
Verkehrsmodell,City,,,2023,dataset,Fahrleistung,
Handbuch,UBA,https://example.com/hb,,,,,Amber
"""


def _registry(tmp_path, text=REGISTRY_CSV):
    path = tmp_path / 'sources.csv'
    path.write_text(text)
    return load_sources_registry(str(path))


def test_registry_reads_target_edition_and_datasets(tmp_path):
    registry = _registry(tmp_path)

    assert registry['Energiebilanz'].target == SOURCE_TARGET_DATASET
    assert registry['Energiebilanz'].fields['edition'] == '2024'
    assert registry['Energiebilanz'].datasets is None
    assert registry['Verkehrsmodell'].datasets == frozenset({'Fahrleistung'})
    # No Target cell means the historical behaviour: cited per row.
    assert registry['Handbuch'].target == 'data_point'
    # An unmapped column is still folded into the description...
    assert registry['Handbuch'].fields['description'] == 'RAG Rating: Amber'
    # ...but the columns this reader understands must not be, or 'Target' would arrive as prose.
    assert registry['Energiebilanz'].fields['description'] == 'Regional energy balance; RAG Rating: Green'


def test_registry_refuses_unknown_target(tmp_path):
    with pytest.raises(ValueError, match='has Target'):
        _registry(tmp_path, 'Name,Target\nEnergiebilanz,datasets\n')


def test_registry_refuses_datasets_on_a_data_point_source(tmp_path):
    with pytest.raises(ValueError, match='lists Datasets'):
        _registry(tmp_path, 'Name,Target,Datasets\nHandbuch,data_point,Fahrleistung\n')


def test_dataset_level_sources_survive_a_dataset_with_no_source_column(tmp_path):
    registry = _registry(tmp_path)
    df = pl.DataFrame({'Year': [2020], 'Value': [1.0]})

    sources = build_sources_metadata(df, registry, 'Energieverbrauch')

    # Both dataset-level sources are emitted -- several are as legitimate as one -- but
    # 'Verkehrsmodell' is restricted to another dataset, and 'Handbuch' is cited per row.
    assert sources == [
        {
            'name': 'Energiebilanz',
            'authority': 'StaLa',
            'url': 'https://example.com/eb',
            'description': 'Regional energy balance; RAG Rating: Green',
            'edition': '2024',
            'target': 'dataset',
        }
    ]


def test_several_dataset_level_sources_on_one_dataset(tmp_path):
    registry = _registry(tmp_path)
    df = pl.DataFrame({'Year': [2020], 'Value': [1.0]})

    sources = build_sources_metadata(df, registry, 'Fahrleistung')

    assert [s['name'] for s in sources or []] == ['Energiebilanz', 'Verkehrsmodell']


def test_per_row_citations_still_work_alongside(tmp_path):
    registry = _registry(tmp_path)
    df = pl.DataFrame({'Year': [2020, 2021], 'Value': [1.0, 2.0], 'Source': ['Handbuch', None]})

    sources = build_sources_metadata(df, registry, 'Energieverbrauch')

    assert [(s['name'], s['target']) for s in sources or []] == [
        ('Energiebilanz', 'dataset'),
        ('Handbuch', 'data_point'),
    ]


def test_a_dataset_level_source_may_not_also_be_cited_per_row(tmp_path):
    registry = _registry(tmp_path)
    df = pl.DataFrame({'Year': [2020], 'Value': [1.0], 'Source': ['Energiebilanz']})

    with pytest.raises(ValueError, match='is also cited'):
        build_sources_metadata(df, registry, 'Energieverbrauch')


def test_no_sources_at_all_still_returns_none():
    df = pl.DataFrame({'Year': [2020], 'Value': [1.0]})
    assert build_sources_metadata(df, None) is None
    assert build_sources_metadata(df, {}) is None


def test_uncited_source_without_a_registry_entry_warns(tmp_path):
    registry = _registry(tmp_path)
    df = pl.DataFrame({'Year': [2020], 'Value': [1.0], 'Source': ['Nicht registriert']})

    with pytest.warns(UserWarning, match='not in the sources registry'):
        sources = build_sources_metadata(df, registry, 'Energieverbrauch')

    assert {s['name'] for s in sources or []} == {'Energiebilanz', 'Nicht registriert'}


def test_datasets_restriction_naming_a_dataset_the_run_does_not_produce_is_refused(tmp_path):
    registry = _registry(tmp_path)

    check_registry_dataset_names(registry, ['Fahrleistung', 'Energieverbrauch'])
    with pytest.raises(ValueError, match='which this run does not produce'):
        check_registry_dataset_names(registry, ['Energieverbrauch'])


def test_a_single_dataset_run_ignores_a_datasets_restriction():
    # plain_csv / plain_csv_wide produce one dataset and pass no name; there is nothing to
    # choose between, so the restriction is moot rather than exclusionary.
    entry = SourceRegistryEntry(fields={'target': SOURCE_TARGET_DATASET}, datasets=frozenset({'Fahrleistung'}))
    assert entry.applies_to(None) is True
    assert entry.applies_to('Fahrleistung') is True
    assert entry.applies_to('Energieverbrauch') is False
