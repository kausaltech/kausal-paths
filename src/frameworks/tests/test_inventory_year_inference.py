from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory

from frameworks.conversion import infer_inventory_years
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.instance_serialization import build_instance_snapshot
from nodes.models import InstanceConfig, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry
from nodes.value_validation import RequiredValueCombination, ValueContract

pytestmark = pytest.mark.django_db


def _instance(activity_years: list[int]) -> InstanceConfig:
    """
    Build an instance with one inventory input and one factor input over 2018-2023.

    The factor table has a value in every year, as the local copies of the national tables
    do; only the activity years may decide the calendar.
    """
    instance = InstanceFactory.create()
    config = InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test',
        spec=InstanceModelSpec(years=YearsSpec(reference=2018, min_historical=2018, max_historical=2023, target=2030)),
    )
    unit = unit_registry.parse_units('t/a')
    activity_port, factor_port = uuid4(), uuid4()
    node = NodeConfigFactory.create(
        instance=config,
        identifier='consumer',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'),
            input_ports=[
                InputPortDef(
                    id=activity_port,
                    unit=unit,
                    quantity='emissions',
                    binding_owner='instance',
                    validation=ValueContract(combinations=[RequiredValueCombination(categories={})]),
                ),
                InputPortDef(id=factor_port, unit=unit, quantity='emissions', binding_owner='instance'),
            ],
            output_ports=[OutputPortDef(id=uuid4(), unit=unit, quantity='emissions')],
        ),
    )
    for identifier, port_id, years in (
        ('activity', activity_port, activity_years),
        ('factors', factor_port, list(range(2018, 2024))),
    ):
        dataset = DatasetFactory.create(identifier=identifier, scope=config)
        metric = DatasetMetricFactory.create(schema=dataset.schema, name='value', label='Value', unit='t/a')
        for year in years:
            DataPointFactory.create(dataset=dataset, metric=metric, date=date(year, 1, 1), value=Decimal(1))
        # An empty cell is not an inventoried year.
        DataPointFactory.create(dataset=dataset, metric=metric, date=date(2017, 1, 1), value=None)
        NodeInputPortBinding.objects.create(instance=config, node=node, port_id=port_id, dataset=dataset, metric=metric)
    return config


def test_years_without_inventory_data_are_skipped():
    config = _instance([2018, 2019, 2021, 2023])
    assert infer_inventory_years(config, build_instance_snapshot(config)) == [2020, 2022]


def test_an_edge_year_without_data_keeps_the_span():
    config = _instance([2020])
    assert infer_inventory_years(config, build_instance_snapshot(config)) == [2019, 2021, 2022]


def test_nothing_to_infer_from_leaves_the_calendar_undeclared():
    config = _instance([])
    assert infer_inventory_years(config, build_instance_snapshot(config)) is None
