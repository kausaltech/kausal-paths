"""Tests for the rename_dimension_category management command."""

from io import StringIO
from typing import TYPE_CHECKING
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.core.management.base import CommandError

import pytest

from kausal_common.datasets.models import DimensionCategory, DimensionScope
from kausal_common.datasets.tests.factories import DimensionCategoryFactory, DimensionFactory

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.transform_def import SelectCategoriesTransformation
from nodes.graphql.editor import InstanceEditorMutation
from nodes.management.commands.rename_dimension_category import suggest_identifier
from nodes.models import InstanceConfig, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dimension

pytestmark = pytest.mark.django_db


@pytest.fixture
def db_ic() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030))
    return InstanceConfigFactory.create(identifier=instance.id, instance=instance, config_source='database', spec=spec)


def _dimension(ic: InstanceConfig, identifier: str, categories: list[str], *, in_spec: bool = True) -> Dimension:
    """Create a dimension with categories stored verbatim, as the editor did before it validated them."""
    dim = DimensionFactory.create(name=identifier)
    ct = ContentType.objects.get_for_model(ic)
    scope = DimensionScope.objects.create(dimension=dim, scope_content_type=ct, scope_id=ic.pk, identifier=identifier)
    for i, cat_id in enumerate(categories):
        DimensionCategoryFactory.create(dimension=dim, identifier=cat_id, label=cat_id, order=i)
    if in_spec:
        InstanceEditorMutation._sync_spec_dimension_from_orm(ic, scope)
    return dim


def _spec_categories(ic: InstanceConfig, dim_id: str) -> list[str] | None:
    ic.refresh_from_db()
    assert ic.spec is not None
    dim = next((d for d in ic.spec.dimensions if d['id'] == dim_id), None)
    return None if dim is None else [c['id'] for c in dim['categories']]


def _run(*args: str) -> str:
    out = StringIO()
    call_command('rename_dimension_category', *args, stdout=out, stderr=StringIO())
    return out.getvalue()


@pytest.mark.parametrize(
    ('old', 'new'),
    [
        ('Straßenbahn', 'strassenbahn'),
        ('paliwo węglowe', 'paliwo_weglowe'),
        ('Osady sciekowe', 'osady_sciekowe'),
        ('sewage 2', 'sewage_2'),
        ('Wärme-Pumpe', 'waerme_pumpe'),
        ('***', 'category'),
    ],
)
def test_suggest_identifier(old: str, new: str):
    assert suggest_identifier(old) == new


def test_auto_plan_writes_nothing(db_ic: InstanceConfig):
    _dimension(db_ic, 'vehicle_type', ['Straßenbahn', 'bus'])
    out = _run(db_ic.identifier, '--auto')
    assert "'Straßenbahn' -> 'strassenbahn'" in out
    assert 'bus' not in out
    assert DimensionCategory.objects.filter(identifier='Straßenbahn').exists()
    assert _spec_categories(db_ic, 'vehicle_type') == ['Straßenbahn', 'bus']


def test_auto_apply_renames_row_and_spec(db_ic: InstanceConfig):
    dim = _dimension(db_ic, 'vehicle_type', ['Straßenbahn', 'bus'])
    cat_uuid = dim.categories.get(identifier='Straßenbahn').uuid
    _run(db_ic.identifier, '--auto', '--apply')
    assert dim.categories.get(uuid=cat_uuid).identifier == 'strassenbahn'
    assert _spec_categories(db_ic, 'vehicle_type') == ['strassenbahn', 'bus']


def test_auto_avoids_taken_identifier(db_ic: InstanceConfig):
    _dimension(db_ic, 'waste', ['Sewage', 'sewage'])
    _run(db_ic.identifier, '--auto', '--apply')
    assert _spec_categories(db_ic, 'waste') == ['sewage_2', 'sewage']


def test_dimension_outside_spec_is_not_added(db_ic: InstanceConfig):
    dim = _dimension(db_ic, 'source', ['Housing Calculation, Arup'], in_spec=False)
    _run(db_ic.identifier, '--auto', '--apply')
    assert dim.categories.get().identifier == 'housing_calculation_arup'
    assert _spec_categories(db_ic, 'source') is None


def test_explicit_rename(db_ic: InstanceConfig):
    _dimension(db_ic, 'odpady', ['Osady sciekowe', 'Test1'])
    _run(db_ic.identifier, '--rename', 'odpady:Osady sciekowe=sludge', '--apply')
    assert _spec_categories(db_ic, 'odpady') == ['sludge', 'Test1']


@pytest.mark.parametrize('target', ['Sludge', 'test1'])
def test_explicit_rename_refuses_invalid_or_taken(db_ic: InstanceConfig, target: str):
    _dimension(db_ic, 'odpady', ['Osady sciekowe', 'test1'])
    with pytest.raises(CommandError):
        _run(db_ic.identifier, '--rename', f'odpady:Osady sciekowe={target}', '--apply')
    assert _spec_categories(db_ic, 'odpady') == ['Osady sciekowe', 'test1']


def test_reference_in_binding_blocks(db_ic: InstanceConfig):
    # A transformation validates its categories, so it can only name a valid one; renaming that
    # one explicitly is the case where a reference exists.
    _dimension(db_ic, 'vehicle_type', ['bus'])
    source = NodeConfigFactory.create(instance=db_ic, identifier='source_node')
    target = NodeConfigFactory.create(instance=db_ic, identifier='target_node')
    NodeInputPortBinding.objects.create(
        instance=db_ic,
        source_node=source,
        node=target,
        source_port_id=uuid4(),
        port_id=uuid4(),
        transformations=[SelectCategoriesTransformation(dimension='vehicle_type', categories=['bus'])],
        tags=[],
    )
    with pytest.raises(CommandError):
        _run(db_ic.identifier, '--rename', 'vehicle_type:bus=coach', '--apply')
    assert _spec_categories(db_ic, 'vehicle_type') == ['bus']

    _run(db_ic.identifier, '--rename', 'vehicle_type:bus=coach', '--apply', '--allow-references')
    assert _spec_categories(db_ic, 'vehicle_type') == ['coach']


def test_yaml_sourced_instance_is_refused(db_ic: InstanceConfig):
    InstanceConfig.objects.filter(pk=db_ic.pk).update(config_source='yaml')
    with pytest.raises(CommandError, match='YAML'):
        _run(db_ic.identifier, '--auto')
