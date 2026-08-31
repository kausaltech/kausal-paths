"""Focused regression tests for the first Zürich port migrations."""

from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import NAMESPACE_URL, uuid3

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.ch.zuerich import BuildingEnergy, DistrictHeatProductionMix
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.dimensions import Dimension, DimensionCategory
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader, InstanceYAMLConfig
from nodes.instance_parser import parse_instance_snapshot
from nodes.tests.factories import NodeFactory
from nodes.tests.node_input_harness import bind, binding, frame, node_case
from nodes.units import unit_registry

if TYPE_CHECKING:
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


class _DistrictHeatProductionMixFactory(NodeFactory):
    class Meta:
        model = DistrictHeatProductionMix


def _dimension_frame(values: list[float], carriers: list[str]):
    raw = pl.DataFrame({YEAR_COLUMN: [2020] * len(values), 'energy_carrier': carriers, VALUE_COLUMN: values})
    return to_ppdf(
        raw,
        DataFrameMeta(
            units={VALUE_COLUMN: unit_registry.parse_units('%')},
            primary_keys=[YEAR_COLUMN, 'energy_carrier'],
        ),
    )


def _district_node(context, *, use_gas_network: bool) -> DistrictHeatProductionMix:
    context.dimensions['energy_carrier'] = Dimension(
        id='energy_carrier',
        label=TranslatedString('Energy carrier', default_language='en'),
        categories=[
            DimensionCategory(id=carrier, label=TranslatedString(carrier, default_language='en'))
            for carrier in ('natural_gas', 'wood')
        ],
    )
    node = cast(
        'DistrictHeatProductionMix',
        _DistrictHeatProductionMixFactory.create(
            context=context,
            unit=unit_registry.parse_units('%'),
            quantity='mix',
            input_datasets=[],
            input_dimension_ids=['energy_carrier'],
            output_dimension_ids=['energy_carrier'],
        ),
    )
    template = next(parameter for parameter in node.allowed_parameters if parameter.local_id == 'use_gas_network')
    fields = template.model_dump()
    fields['value'] = use_gas_network
    node.add_parameter(type(template)(**fields))
    return node


def test_building_energy_declares_and_uses_named_inputs() -> None:
    assert BuildingEnergy.energy_port.role == 'energy'
    assert BuildingEnergy.other_fuel_use_port.role == 'other_fuel_use'

    node = object.__new__(BuildingEnergy)
    node.id = 'building_energy'
    bind(
        node,
        [
            binding('energy', frame([10.0], unit='GWh/a')),
            binding('other_fuel_use', frame([3.0], unit='GWh/a')),
        ],
    )
    result = node.compute()
    assert result[VALUE_COLUMN].to_list() == [7.0]


def test_district_heat_ports_keep_optional_gas_inputs_distinct() -> None:
    assert DistrictHeatProductionMix.base_mix_port.role == 'base_mix'
    assert DistrictHeatProductionMix.additive_port.role == 'additive'
    assert DistrictHeatProductionMix.gas_mix_port.role == 'gas_mix'
    assert DistrictHeatProductionMix.grid_share_port.role == 'grid_share'
    assert DistrictHeatProductionMix.gas_mix_port.required is False
    assert DistrictHeatProductionMix.grid_share_port.required is False


def test_district_heat_base_and_additive_inputs_are_available_in_binding_order() -> None:
    base = _dimension_frame([60.0], ['natural_gas'])
    additive = _dimension_frame([40.0], ['wood'])
    node = bind(
        node_case(DistrictHeatProductionMix.base_mix_port, DistrictHeatProductionMix.additive_port),
        [binding('base_mix', base), binding('additive', additive)],
    )
    assert node.get_input(DistrictHeatProductionMix.base_mix_port) is base
    assert list(node.iter_inputs(DistrictHeatProductionMix.additive_port)) == [additive]


def test_district_heat_missing_optional_gas_inputs_is_not_a_binding_error() -> None:
    node = bind(
        node_case(DistrictHeatProductionMix.base_mix_port, DistrictHeatProductionMix.gas_mix_port),
        [binding('base_mix', _dimension_frame([100.0], ['wood']))],
    )
    assert node.get_input(DistrictHeatProductionMix.gas_mix_port) is None


def test_district_heat_requires_gas_inputs_only_when_network_mix_is_enabled(context) -> None:
    base = _dimension_frame([60.0, 40.0], ['natural_gas', 'wood'])

    disabled = bind(_district_node(context, use_gas_network=False), [binding('base_mix', base)])
    result = disabled.compute()
    yearly_totals = result.paths.sum_over_dims(['energy_carrier'])[VALUE_COLUMN]
    assert yearly_totals.to_list() == pytest.approx([100.0] * len(yearly_totals))

    enabled = bind(_district_node(context, use_gas_network=True), [binding('base_mix', base)])
    with pytest.raises(NodeError, match=r"Input role 'gas_mix'.*required"):
        enabled.compute()


def test_zuerich_yaml_projects_legacy_bindings_to_semantic_roles() -> None:
    data = InstanceYAMLConfig.load_for_entrypoint(Path('configs/zuerich.yaml').resolve()).data
    assert data is not None
    snapshot = parse_instance_snapshot(data, instance_uuid=uuid3(NAMESPACE_URL, 'paths:zuerich-runtime-input-test'))
    loader = object.__new__(InstanceLoader)
    loader.instance_config = None
    loader._stash_snapshot_bindings(snapshot)

    building = next(
        node for node in loader._instance_graph.nodes if node.identifier == 'building_end_energy_consumption_historical'
    )
    assert [building.role_for_input_port(port) for port in building.spec.input_ports] == [
        'energy',
        'other_fuel_use',
        None,
    ]

    district = next(node for node in loader._instance_graph.nodes if node.identifier == 'district_heat_production_mix')
    assert {district.role_for_input_port(port) for port in district.spec.input_ports} == {
        'base_mix',
        'additive',
        'gas_mix',
        'grid_share',
    }


def test_zuerich_yaml_does_not_read_the_persisted_dataset_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = Path('configs/zuerich.yaml').resolve()
    data = InstanceYAMLConfig.load_for_entrypoint(yaml_path).data
    assert data is not None
    snapshot = parse_instance_snapshot(data, instance_uuid=uuid3(NAMESPACE_URL, 'paths:zuerich-catalog-test'))

    def fail_if_called(_instance_config: object) -> None:
        raise AssertionError('YAML graph construction must not read the persisted dataset catalog')

    monkeypatch.setattr('nodes.instance_serialization.build_instance_snapshot', fail_if_called)

    loader = object.__new__(InstanceLoader)
    loader.instance_config = cast('InstanceConfig', object())
    loader.yaml_file_path = yaml_path
    loader._stash_snapshot_bindings(snapshot)

    resolved_dataset_ids = {dataset.identifier for dataset in loader._instance_graph.datasets}
    assert {
        binding.dataset_source.dataset for binding in snapshot.dataset_bindings if binding.dataset_source is not None
    } <= resolved_dataset_ids
