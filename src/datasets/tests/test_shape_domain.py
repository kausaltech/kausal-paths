from uuid import uuid4

from django.contrib.contenttypes.models import ContentType
from django.db import transaction

import pytest

from kausal_common.datasets.models import DimensionScope
from kausal_common.datasets.tests.factories import (
    DatasetFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from datasets.materialization import materializations_with_validation_hashes, refresh_dataset_materialization
from datasets.shape_domain import SHAPE_SPEC_KEY, CategoryDomainResolver, dataset_category_domain
from datasets.validation import dataset_validation_hash
from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.shape_defs import ShapeCombinationSpec, ShapeSpec
from nodes.models import DatasetMaterialization
from nodes.template_graph import publish_template_instance
from nodes.template_spec import ensure_instance_shapes
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db

YEARS = YearsSpec(reference=2020, min_historical=2020, max_historical=2022, target=2030)


def _combination(sector: str) -> ShapeCombinationSpec:
    return ShapeCombinationSpec(uuid=uuid4(), identifier=sector, categories={'sector': sector})


@pytest.fixture
def rig():
    """Build a template with a standard and an extension point, and a municipality that extends it."""
    standard = ShapeSpec(uuid=uuid4(), identifier='std/sectors', dimensions=['sector'], combinations=[_combination('households')])
    point = ShapeSpec(uuid=uuid4(), identifier='sectors', inherits=[standard.uuid], closed=True, owner='instance')
    template = InstanceConfigFactory.create(
        name='Template', config_source='database', spec=InstanceModelSpec(years=YEARS, shapes=[standard, point])
    )
    framework = FrameworkFactory.create(template_instance=template)
    dimension = DimensionFactory.create()
    for identifier in ('households', 'industry'):
        DimensionCategoryFactory.create(dimension=dimension, identifier=identifier)
    DimensionScope.objects.create(
        dimension=dimension,
        identifier='sector',
        scope_content_type=ContentType.objects.get_for_model(framework),
        scope_id=framework.pk,
    )
    revision = publish_template_instance(template)
    local = InstanceModelSpec(years=YEARS)
    ensure_instance_shapes(local, template.ensure_spec())
    local.shapes[0] = local.shapes[0].model_copy(update={'combinations': [_combination('industry')]})
    municipality = InstanceConfigFactory.create(
        name='Municipality', config_source='database', template_revision=revision, spec=local
    )
    FrameworkConfig.objects.create(framework=framework, instance_config=template)
    FrameworkConfig.objects.create(framework=framework, instance_config=municipality)
    schema = DatasetSchemaFactory.create()
    shaped = {SHAPE_SPEC_KEY: str(point.uuid)}
    template_dataset = DatasetFactory.create(scope=template, schema=schema, spec=dict(shaped))
    municipal_dataset = DatasetFactory.create(scope=municipality, schema=schema, spec=dict(shaped))
    return template_dataset, municipal_dataset, municipality


def _materialize(dataset) -> None:
    with transaction.atomic():
        refresh_dataset_materialization(dataset, touch=False)


def _sectors(domain) -> set[str]:
    return {combination.identifier for combination in domain.combinations}


def test_a_shared_schema_resolves_per_instance(rig) -> None:
    template_dataset, municipal_dataset, _ = rig

    template_domain = dataset_category_domain(template_dataset)
    municipal_domain = dataset_category_domain(municipal_dataset)

    assert _sectors(template_domain) == {'households'}
    assert _sectors(municipal_domain) == {'households', 'industry'}
    assert municipal_domain.mode == 'closed'
    # The schema the two datasets share stores none of it.
    assert template_dataset.schema.category_domain.combinations == []


def test_a_dataset_without_a_shape_keeps_its_schema_domain(rig) -> None:
    template_dataset, _, _ = rig
    template_dataset.spec = {}
    assert dataset_category_domain(template_dataset) == template_dataset.schema.category_domain


def test_editing_the_municipal_record_changes_the_validation_hash(rig) -> None:
    _, municipal_dataset, municipality = rig
    _materialize(municipal_dataset)
    before = dataset_validation_hash(municipal_dataset)
    spec = municipality.ensure_spec()
    spec.shapes[0] = spec.shapes[0].model_copy(update={'combinations': []})
    municipality.save(update_fields=['spec'])

    assert dataset_validation_hash(municipal_dataset) != before
    ((_, bulk_hash),) = materializations_with_validation_hashes(DatasetMaterialization.objects.filter(dataset=municipal_dataset))
    assert bulk_hash == dataset_validation_hash(municipal_dataset)


def test_one_resolver_reads_each_instance_once(rig, django_assert_max_num_queries) -> None:
    _, municipal_dataset, _ = rig
    resolver = CategoryDomainResolver()
    resolver.for_dataset(municipal_dataset)
    with django_assert_max_num_queries(1):
        resolver.for_dataset(municipal_dataset)
