from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import polars as pl
import pytest
from dvc_pandas import Dataset as DVCPandasDataset, DatasetManifest, DatasetMeta as DVCPandasMeta, IndexColumn

from datasets.dvc_shapes import (
    DVC_SOURCE_SHAPE_VERSION,
    SourceShape,
    _recorded_hashes,
    dvc_source_shape_is_recorded,
    known_dvc_source_shapes,
    record_dvc_source_shape,
    shape_from_manifest,
    shape_of_dvc_dataset,
)
from datasets.models import DVCSourceManifest, DVCSourceShape
from datasets.tests.test_manifests import REVISION, SPEC
from nodes.constraints.values import PortValue
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot

pytestmark = pytest.mark.django_db


def _dataset(frame: dict[str, list[Any]], *, index: list[IndexColumn] | None, units: dict[str, str], hash: str | None = None):  # noqa: A002
    return DVCPandasDataset(
        pl.DataFrame(frame),
        DVCPandasMeta(identifier='test/data', units=units, index_columns=index, hash=hash),
    )


def _manifest(identifier: str, content_hash: str, index_columns: list[Any] | None) -> DatasetManifest:
    return DatasetManifest(
        repository_url=SPEC.url,
        revision=REVISION,
        identifier=identifier,
        content_hash=content_hash,
        object_url='s3://example/files/md5/' + content_hash,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        units={'Value': 'kt/a'},
        index_columns=index_columns,
    )


def _save(manifest: DatasetManifest) -> None:
    DVCSourceManifest.objects.create(
        repository_url=SPEC.url,
        revision=REVISION,
        remote_name='storage',
        format_version=1,
        dataset_identifier=manifest.identifier,
        content=manifest.model_dump(mode='json'),
    )


def test_an_indexed_source_has_its_index_as_dimensions() -> None:
    dataset = _dataset(
        {'Year': [2020], 'sector': ['a'], 'Value': [1.0], 'Source': ['x']},
        index=['Year', 'sector'],
        units={'Value': 'kt/a'},
    )
    assert shape_of_dvc_dataset(dataset) == SourceShape(
        dimension_columns=('sector',), metric_units={'Value': 'kt/a'}, metric_dimension_columns={'Value': ('sector',)}
    )


def test_each_metric_has_only_the_dimensions_it_has_values_in() -> None:
    dataset = _dataset(
        {
            'Year': [2020, 2020],
            'crop': ['wheat', None],
            'livestock': [None, 'cattle'],
            'area': [1.0, None],
            'number': [None, 2.0],
        },
        index=['Year', 'crop', 'livestock'],
        units={'area': 'ha', 'number': 'count'},
    )
    assert shape_of_dvc_dataset(dataset).metric_dimension_columns == {'area': ('crop',), 'number': ('livestock',)}


def test_a_metric_may_fill_a_dimension_only_in_part() -> None:
    dataset = _dataset(
        {'Year': [2020, 2020], 'subsector': ['rice', 'lime'], 'crop': ['rice', None], 'factor': [1.0, 2.0]},
        index=['Year', 'subsector', 'crop'],
        units={'factor': 't/t'},
    )
    shape = shape_of_dvc_dataset(dataset)
    assert shape.metric_dimension_columns == {'factor': ('subsector', 'crop')}
    assert shape.metric_partial_columns == {'factor': ('crop',)}


def test_an_unindexed_source_is_scalar_only_when_nothing_else_could_be_a_dimension() -> None:
    plain = _dataset({'Year': [2020], 'Value': [1.0], 'Comment': ['x']}, index=None, units={'Value': 'kt/a'})
    assert shape_of_dvc_dataset(plain).dimension_columns == ()
    keyed = _dataset({'Year': [2020], 'sector': ['a'], 'Value': [1.0]}, index=None, units={'Value': 'kt/a'})
    # Which of its columns become dimensions is decided by the binding's operations.
    assert shape_of_dvc_dataset(keyed).dimension_columns is None


def test_a_dvc_file_records_a_shape_only_when_it_names_index_columns() -> None:
    assert shape_from_manifest(_manifest('a', 'c' * 32, ['Year', 'sector'])) == SourceShape(
        dimension_columns=('sector',), metric_units={'Value': 'kt/a'}, metric_dimension_columns={'Value': ('sector',)}
    )
    several = _manifest('a', 'c' * 32, ['Year', 'sector']).model_copy(update={'units': {'a': 'kt/a', 'b': 'kt/a'}})
    assert shape_from_manifest(several) is None
    assert shape_from_manifest(_manifest('a', 'c' * 32, None)) is None
    assert shape_from_manifest(_manifest('a', 'c' * 32, [])) is None
    unnamed = {'kind': 'range', 'name': None, 'start': 0, 'stop': 1, 'step': 1}
    assert shape_from_manifest(_manifest('a', 'c' * 32, [unnamed])) is None


def test_a_recorded_shape_is_found_through_the_pinned_manifest() -> None:
    content_hash = uuid4().hex
    _recorded_hashes.discard(content_hash)
    _save(_manifest('test/data', content_hash, None))
    dataset = _dataset(
        {'Year': [2020], 'sector': ['a'], 'Value': [1.0]}, index=['Year', 'sector'], units={'Value': 'kt/a'}, hash=content_hash
    )
    assert known_dvc_source_shapes(SPEC, ['test/data']) == {}

    assert not dvc_source_shape_is_recorded(content_hash)
    record_dvc_source_shape(dataset)
    record_dvc_source_shape(dataset)  # once per hash
    assert dvc_source_shape_is_recorded(content_hash)
    _recorded_hashes.discard(content_hash)
    assert dvc_source_shape_is_recorded(content_hash)  # from the database
    assert DVCSourceShape.objects.filter(content_hash=content_hash, format_version=DVC_SOURCE_SHAPE_VERSION).count() == 1
    assert known_dvc_source_shapes(SPEC, ['test/data', 'test/unknown']) == {
        'test/data': SourceShape(
            dimension_columns=('sector',), metric_units={'Value': 'kt/a'}, metric_dimension_columns={'Value': ('sector',)}
        )
    }


def _yaml_instance(input_dataset: str | dict[str, Any] = 'test/data') -> dict[str, Any]:
    return {
        'id': 'dvc_shapes',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'DVC shapes',
        'owner': 'Owner',
        'theme_identifier': 'default',
        'target_year': 2030,
        'model_end_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2020,
        'emission_unit': 'kt/a',
        'dataset_repo': {'url': SPEC.url, 'commit': REVISION, 'dvc_remote': 'storage'},
        'dimensions': [{'id': 'sector', 'label': 'Sector', 'categories': [{'id': 'a', 'label': 'A'}]}],
        'nodes': [
            {
                'id': 'emissions',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Emissions',
                'unit': 'kt/a',
                'quantity': 'emissions',
                'input_datasets': [input_dataset],
            },
        ],
        'scenarios': [{'id': 'default', 'name': 'Default', 'default': True}],
    }


def _solve_emissions(instance: dict[str, Any]):
    snapshot = parse_instance_snapshot(instance, instance_uuid=uuid4())
    nodes = []
    for node in snapshot.nodes:
        assert node.spec is not None
        ports = [port.model_copy(update={'dimensions': None}) for port in node.spec.output_ports]
        nodes.append(node.model_copy(update={'spec': node.spec.model_copy(update={'output_ports': ports})}))
    snapshot = snapshot.model_copy(update={'nodes': nodes})

    loader = object.__new__(InstanceLoader)
    loader.instance_config = None
    loader.yaml_file_path = None
    loader._stash_snapshot_bindings(snapshot)
    graph = loader._instance_graph

    node = next(meta for meta in graph.nodes if meta.identifier == 'emissions')
    result = graph.solve_constraints()
    return graph, result, result.shapes[PortValue(node.id, node.spec.output_ports[0].id, 'output')]


def test_a_yaml_graph_infers_through_a_recorded_dvc_shape() -> None:
    _save(_manifest('test/data', uuid4().hex, ['Year', 'sector']))
    graph, _, shape = _solve_emissions(_yaml_instance())
    assert shape.dimensions == {graph.dimension_by_identifier['sector'].id}


def test_raw_key_columns_are_followed_through_the_binding() -> None:
    _save(_manifest('test/data', uuid4().hex, ['Vuosi', 'Sektori', 'Kaupunki', 'Tyyppi']))
    binding = {
        'id': 'test/data',
        'filters': [
            {'rename_col': 'Vuosi', 'value': 'Year'},
            {'rename_col': 'Sektori', 'value': 'sector'},
            {'column': 'Kaupunki', 'value': 'Espoo'},
            {'column': 'Tyyppi', 'flatten': True},
        ],
    }
    graph, result, shape = _solve_emissions(_yaml_instance(binding))
    assert shape.dimensions == {graph.dimension_by_identifier['sector'].id}
    assert not result.conflicts


def test_a_raw_key_column_the_binding_leaves_leaves_the_shape_unknown() -> None:
    # Some node classes read raw columns themselves, so a leftover is not an error in itself.
    _save(_manifest('test/data', uuid4().hex, ['Year', 'Sektori', 'Kaupunki']))
    binding = {'id': 'test/data', 'filters': [{'rename_col': 'Sektori', 'value': 'sector'}]}
    _, result, shape = _solve_emissions(_yaml_instance(binding))
    assert shape.dimensions is None
    assert not result.conflicts


def test_a_metric_drops_the_dimensions_it_has_no_values_in() -> None:
    content_hash = uuid4().hex
    _save(_manifest('test/data', content_hash, None))
    _recorded_hashes.discard(content_hash)
    record_dvc_source_shape(
        _dataset(
            {
                'Year': [2020, 2020],
                'sector': ['a', None],
                'Kaupunki': ['Espoo', 'Espoo'],
                'Value': [1.0, None],
                'Other': [None, 1.0],
            },
            index=['Year', 'sector', 'Kaupunki'],
            units={'Value': 'kt/a', 'Other': 'kt/a'},
            hash=content_hash,
        )
    )
    binding = {'id': 'test/data', 'column': 'Other', 'filters': [{'column': 'Kaupunki', 'value': 'Espoo'}]}
    _, result, shape = _solve_emissions(_yaml_instance(binding))
    assert shape.dimensions == frozenset()
    assert not result.conflicts


def test_filtering_a_metric_that_fills_a_dimension_in_part_leaves_the_shape_unknown() -> None:
    content_hash = uuid4().hex
    _save(_manifest('test/data', content_hash, None))
    _recorded_hashes.discard(content_hash)
    record_dvc_source_shape(
        _dataset(
            {'Year': [2020, 2020], 'sector': ['a', 'b'], 'Tyyppi': ['x', None], 'Value': [1.0, 2.0]},
            index=['Year', 'sector', 'Tyyppi'],
            units={'Value': 'kt/a'},
            hash=content_hash,
        )
    )
    # Selecting `b` would leave `Tyyppi` null throughout, which the runtime drops; without the data
    # the solver cannot tell what a selection leaves.
    binding = {'id': 'test/data', 'filters': [{'dimension': 'sector', 'categories': ['a']}]}
    _, result, shape = _solve_emissions(_yaml_instance(binding))
    assert shape.dimensions is None
    assert not result.conflicts
