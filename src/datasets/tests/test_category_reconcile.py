from datetime import date

import pytest

from kausal_common.datasets.models import DataPointDimensionCategory, DatasetMetricValidationRule, DimensionCategory
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from datasets.category_reconcile import apply_changes, plan_dimension
from nodes.dimensions import Dimension, DimensionCategory as RuntimeCategory
from nodes.models import DatasetMaterialization
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def _declared(*categories: tuple[str, list[str]]) -> Dimension:
    return Dimension(
        id='energy_carrier',
        label='Energy carrier',
        categories=[RuntimeCategory(id=cid, label=cid, aliases=aliases) for cid, aliases in categories],
    )


@pytest.fixture
def rig():
    dimension = DimensionFactory.create()
    cats = {
        cid: DimensionCategoryFactory.create(dimension=dimension, identifier=cid) for cid in ('gas', 'geothermatl', 'geothermal')
    }
    schema = DatasetSchemaFactory.create()
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
    metric = DatasetMetricFactory.create(schema=schema)
    dataset = DatasetFactory.create(scope=InstanceConfigFactory.create(name='Local', config_source='database'), schema=schema)
    return dimension, cats, metric, dataset


def test_alias_merges_data_and_unaliased_category_is_deleted(rig) -> None:
    dimension, cats, metric, dataset = rig
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=3)
    point.dimension_categories.add(cats['geothermatl'])
    DimensionCategoryFactory.create(dimension=dimension, identifier='unused')

    changes = plan_dimension('energy_carrier', dimension, _declared(('gas', []), ('geothermal', ['geothermatl'])))

    assert [(c.category.identifier, c.action, c.refusals) for c in changes] == [
        ('geothermatl', 'merge', []),
        ('unused', 'delete', []),
    ]
    assert apply_changes(changes) == [dataset]
    assert set(dimension.categories.values_list('identifier', flat=True)) == {'gas', 'geothermal'}
    assert list(point.dimension_categories.all()) == [cats['geothermal']]
    assert DatasetMaterialization.objects.filter(dataset=dataset).exists()
    assert plan_dimension('energy_carrier', dimension, _declared(('gas', []), ('geothermal', ['geothermatl']))) == []


def test_refuses_deleting_a_category_with_data(rig) -> None:
    dimension, cats, metric, dataset = rig
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=3)
    point.dimension_categories.add(cats['geothermatl'])

    changes = plan_dimension('energy_carrier', dimension, _declared(('gas', []), ('geothermal', [])))

    assert changes[0].action == 'delete'
    assert changes[0].refusals
    with pytest.raises(ValueError, match='Refusing'):
        apply_changes(changes)
    assert DataPointDimensionCategory.objects.filter(dimension_category=cats['geothermatl']).exists()


def test_refuses_merging_into_a_populated_category(rig) -> None:
    _dimension, cats, metric, dataset = rig
    for category in ('geothermatl', 'geothermal'):
        point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=3)
        point.dimension_categories.add(cats[category])

    changes = plan_dimension('energy_carrier', rig[0], _declared(('gas', []), ('geothermal', ['geothermatl'])))

    assert changes[0].action == 'merge'
    assert 'both categories have values' in changes[0].refusals[0]


def test_refuses_a_category_named_by_uuid(rig) -> None:
    _dimension, cats, metric, _dataset = rig
    DatasetMetricValidationRule.objects.create(metric=metric, rule={'kind': 'custom', 'category': str(cats['geothermatl'].uuid)})

    changes = plan_dimension('energy_carrier', rig[0], _declared(('gas', []), ('geothermal', ['geothermatl'])))

    assert any('validation rule' in refusal for refusal in changes[0].refusals)
    assert DimensionCategory.objects.filter(pk=cats['geothermatl'].pk).exists()
