from pathlib import Path
from typing import TYPE_CHECKING

from django.conf import settings

import polars as pl
import pytest
import yaml

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.node_defs import FormulaConfig, NodeSpec
from nodes.edges import Edge
from nodes.formula import FormulaNode
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from common import polars as ppl
    from nodes.context import Context

pytestmark = pytest.mark.django_db


class PopulationSource(Node):
    def __init__(self, context: Context, identifier: str, rows: Sequence[tuple[int, float | None]]) -> None:
        super().__init__(
            id=identifier,
            context=context,
            name=TranslatedString(identifier, default_language='en'),
            unit=unit_registry.parse_units('cap'),
            quantity='population',
        )
        self.frame = to_ppdf(
            pl.DataFrame(
                {
                    YEAR_COLUMN: [year for year, _value in rows],
                    VALUE_COLUMN: [value for _year, value in rows],
                    FORECAST_COLUMN: [False] * len(rows),
                },
                schema={YEAR_COLUMN: pl.Int64, VALUE_COLUMN: pl.Float64, FORECAST_COLUMN: pl.Boolean},
            ),
            DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('cap')}, primary_keys=[YEAR_COLUMN]),
        )

    def compute(self) -> ppl.PathsDataFrame:
        return self.frame


@pytest.mark.parametrize(
    ('own_rows', 'expected'),
    [
        ([(2023, None)], {2023: 90, 2024: 91}),
        ([(2023, 100)], {2023: 100, 2024: 91}),
        ([(2023, 0)], {2023: 0, 2024: 91}),
        ([(2023, 100), (2024, 101), (2030, 110)], {2023: 100, 2024: 101, 2030: 110}),
    ],
)
def test_population_model_preserves_provider_values_until_municipal_values_exist(
    own_rows: list[tuple[int, float | None]],
    expected: dict[int, float],
) -> None:
    instance = InstanceFactory.create(id='population-selection', name='Population selection')
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=instance.name)
    context = instance.context
    config = yaml.safe_load((Path(settings.BASE_DIR) / 'configs/modules/bisko/population-defaults.yaml').read_text())
    definition = next(node for node in config['nodes'] if node['id'] == 'population')
    target = FormulaNode(
        id='population',
        context=context,
        name=TranslatedString('Population', default_language='en'),
        unit=unit_registry.parse_units('cap'),
        quantity='population',
    )
    target._spec = NodeSpec(type_config=FormulaConfig(formula=definition['params']['formula']))
    for identifier, rows in (('own', own_rows), ('provider', [(2023, 90.0), (2024, 91.0)])):
        source = PopulationSource(context, identifier, rows)
        edge = Edge(input_node=source, output_node=target, tags=[identifier])
        source.add_edge(edge)
        target.add_edge(edge)
    output = target.compute()
    assert dict(output.select(YEAR_COLUMN, VALUE_COLUMN).iter_rows()) == expected
