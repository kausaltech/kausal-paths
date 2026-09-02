"""
End-to-end cover for the placeholder values on the Additional Historical Data tab.

The unit tests in ``test_placeholder_lookup`` pin the mapping rules and the frame
shaping. What they cannot see is whether the values reach a client, and that is where
this feature has failed twice: once when a model swap left the node lookup matching
nothing, and once when the frame read for measure linkage was routed through a pipeline
that had since started dropping the ``uuid`` column. Both produced an empty
``placeholderDataPoints`` and a blank grey cell, which is indistinguishable from "this
city has not filled that year in" -- so nothing complained for weeks either time.

This test walks the path the client actually walks:

    framework -> section -> measureTemplates -> measure(frameworkConfigId:)
        -> placeholderDataPoints

which is *not* the ``framework.config.measures`` path, and reaches the resolver through
the object cache and a real request rather than through direct calls.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from common.polars import DataFrameMeta, to_ppdf
from frameworks.datasets import FrameworkMeasureDVCDataset2
from frameworks.models import Framework, FrameworkConfig, Measure, MeasureTemplate, Section
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from common.polars import PathsDataFrame
    from nodes.instance import Instance
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

MEASURE_UUID = '44444444-4444-4444-4444-444444444444'
BASELINE_YEAR = 2019
WINDOW = (2020, 2021, 2022, 2023, 2024, 2025)

QUERY = """
query GetPlaceholders($fw: ID!, $section: ID!, $fwcId: ID!) {
  framework(identifier: $fw) {
    section(identifier: $section) {
      measureTemplates {
        uuid
        measure(frameworkConfigId: $fwcId) {
          placeholderDataPoints { year value }
        }
      }
    }
  }
}
"""


def _series(years: tuple[int, ...]) -> PathsDataFrame:
    """Build a plain one-metric output frame, the shape a level node emits."""
    import polars as pl

    df = pl.DataFrame({
        YEAR_COLUMN: list(years),
        VALUE_COLUMN: [float(y - 2000) for y in years],
        FORECAST_COLUMN: [False] * len(years),
    })
    return to_ppdf(df, DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('%')}, primary_keys=[YEAR_COLUMN]))


def _city_data_binding(context: Any) -> FrameworkMeasureDVCDataset2:
    """
    Build a binding that claims MEASURE_UUID and has to *load* to say so.

    Deliberately does not seed ``_uuid_frame``/``_uuid_frame_loaded``. Pre-seeding the memo
    is convenient and it is what the unit tests do, but it means ``get_uuid_frame`` never
    runs its body -- and that body is where this feature broke the second time, when the
    measure-datapoint overlay moved into ``before_temporal_fill`` and started dropping
    ``uuid`` before the method could read it. A test that seeds the memo cannot see that
    class of regression at all. Handing it a payload store instead makes the real read run.
    """
    import polars as pl

    ds = FrameworkMeasureDVCDataset2(id='nzc/test_city_data', context=context)
    ds.tags = ['city_data']
    frame = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [BASELINE_YEAR], 'uuid': [MEASURE_UUID], VALUE_COLUMN: [1.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('%')}, primary_keys=[YEAR_COLUMN, 'uuid']),
    )
    ds.payload_ref = cast('Any', object())
    ds.payload_store = cast('Any', SimpleNamespace(get_dataframe=lambda _ref: frame))
    return ds


@pytest.fixture
def superuser_client(client: Any, instance_config: InstanceConfig) -> PathsTestClient:
    """
    Return an authenticated client: ``resolve_measure`` reads a permission-filtered cache.

    An anonymous request gets ``framework.config == null`` and ``measure == null`` with no
    GraphQL error at all -- the config simply is not in ``cache.framework_configs``. That is
    indistinguishable from a broken resolver, and it is worth a comment because it cost an
    afternoon of chasing the wrong layer once already.
    """
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(instance_config)
    return tc


@pytest.fixture
def measure_on_a_node(instance: Instance, instance_config: InstanceConfig) -> tuple[FrameworkConfig, MeasureTemplate]:
    """Wire one MeasureTemplate to one node of the autouse instance, through a city_data binding."""
    from nodes.tests.factories import NodeFactory

    node = NodeFactory.create(context=instance.context)
    node.input_dataset_instances = [_city_data_binding(instance.context)]
    node.get_output_pl = lambda **_kwargs: _series(WINDOW)  # type: ignore[method-assign]

    fwc = FrameworkConfigFactory.create(instance_config=instance_config, baseline_year=BASELINE_YEAR)
    section = Section.add_root(instance=Section(framework=fwc.framework, identifier='data_collection', name='Data collection'))
    template = MeasureTemplate.objects.create(section=section, name='Share of something', unit='%', uuid=MEASURE_UUID)
    Measure.objects.create(framework_config=fwc, measure_template=template)
    return fwc, template


def test_placeholder_data_points_reach_the_client(
    superuser_client: PathsTestClient,
    measure_on_a_node: tuple[FrameworkConfig, MeasureTemplate],
) -> None:
    """
    The whole point of the feature: a city sees the model's planned value in an unfilled cell.

    Asserting the *years* rather than only "not empty" is deliberate. An empty list is what
    both past regressions produced, but a resolver that silently narrowed the window would
    also read as working, and the window is what decides which cells the city can be helped
    with.
    """
    fwc, _template = measure_on_a_node

    data = superuser_client.query_data(
        QUERY,
        variables={'fw': fwc.framework.identifier, 'section': 'data_collection', 'fwcId': str(fwc.pk)},
    )

    (mt,) = data['framework']['section']['measureTemplates']
    assert mt['uuid'] == MEASURE_UUID
    points = mt['measure']['placeholderDataPoints']
    assert [p['year'] for p in points] == list(WINDOW)
    assert [p['value'] for p in points] == [float(y - 2000) for y in WINDOW]


def test_a_measure_no_node_claims_gets_no_placeholders(
    superuser_client: PathsTestClient,
    instance: Instance,
    instance_config: InstanceConfig,
) -> None:
    """The honest empty case: nothing in the graph carries this measure, so the cell stays blank."""
    fwc = FrameworkConfigFactory.create(instance_config=instance_config, baseline_year=BASELINE_YEAR)
    section = Section.add_root(instance=Section(framework=fwc.framework, identifier='data_collection', name='Data collection'))
    template = MeasureTemplate.objects.create(section=section, name='Unclaimed', unit='%', uuid=MEASURE_UUID)
    Measure.objects.create(framework_config=fwc, measure_template=template)

    data = superuser_client.query_data(
        QUERY,
        variables={'fw': fwc.framework.identifier, 'section': 'data_collection', 'fwcId': str(fwc.pk)},
    )

    (mt,) = data['framework']['section']['measureTemplates']
    assert mt['measure']['placeholderDataPoints'] == []
    assert Framework.objects.filter(pk=fwc.framework.pk).exists()
    assert template.uuid
