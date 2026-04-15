"""Tests for dimension-related GraphQL queries and mutations in the model editor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import DimensionCategory, DimensionScope
from kausal_common.datasets.tests.factories import DimensionCategoryFactory, DimensionFactory

from nodes.defs.instance_defs import InstanceSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dimension

    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig


gql = str

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceSpec(
        primary_language='en',
        owner='Test Owner',
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=spec,
    )


@pytest.fixture
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


def _make_dimension(ic: InstanceConfig, identifier: str, name: str, categories: list[str] | None = None) -> Dimension:
    """Create a Dimension scoped to an InstanceConfig with optional categories."""
    dim = DimensionFactory.create(name=name)
    ct = ContentType.objects.get_for_model(ic)
    DimensionScope.objects.create(dimension=dim, scope_content_type=ct, scope_id=ic.pk, identifier=identifier)
    if categories:
        for label in categories:
            DimensionCategoryFactory.create(dimension=dim, identifier=label.lower(), label=label)
    return dim


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

INSTANCE_DIMENSIONS = gql("""
query ModelInstance($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        dimensions {
            id
            identifier
            name
            categories {
                id
                identifier
                label
                order
                previousSibling
                nextSibling
            }
        }
    }
}
""")


def test_query_instance_dimensions(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Energy', 'Transport', 'Buildings'])
    data = gql_client.query_data(INSTANCE_DIMENSIONS, variables={'instanceId': str(db_instance_config.pk)})
    dims = data['modelInstance']['dimensions']
    assert len(dims) == 1
    d = dims[0]
    assert d['identifier'] == 'sector'
    assert d['name'] == 'Sector'
    assert d['id'] == str(dim.uuid)

    cats = d['categories']
    assert len(cats) == 3
    assert [c['label'] for c in cats] == ['Energy', 'Transport', 'Buildings']
    assert [c['order'] for c in cats] == [1, 2, 3]

    # Sibling chain
    assert cats[0]['previousSibling'] is None
    assert cats[0]['nextSibling'] == cats[1]['id']
    assert cats[1]['previousSibling'] == cats[0]['id']
    assert cats[1]['nextSibling'] == cats[2]['id']
    assert cats[2]['previousSibling'] == cats[1]['id']
    assert cats[2]['nextSibling'] is None


def test_query_empty_dimension(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    _make_dimension(db_instance_config, 'empty', 'Empty Dim')
    data = gql_client.query_data(INSTANCE_DIMENSIONS, variables={'instanceId': str(db_instance_config.pk)})
    d = data['modelInstance']['dimensions'][0]
    assert d['categories'] == []


def test_query_multiple_dimensions(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    _make_dimension(db_instance_config, 'sector', 'Sector', ['A'])
    _make_dimension(db_instance_config, 'fuel', 'Fuel Type', ['Gas', 'Oil'])
    data = gql_client.query_data(INSTANCE_DIMENSIONS, variables={'instanceId': str(db_instance_config.pk)})
    dims = data['modelInstance']['dimensions']
    assert len(dims) == 2
    identifiers = {d['identifier'] for d in dims}
    assert identifiers == {'sector', 'fuel'}


# ---------------------------------------------------------------------------
# createDimensionCategories
# ---------------------------------------------------------------------------

CREATE_CATS = gql("""
mutation CreateCats($instanceId: ID!, $input: [CreateDimensionCategoryInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        createDimensionCategories(input: $input) {
            ... on InstanceDimension {
                id
                categories {
                    id
                    identifier
                    label
                    order
                    previousSibling
                    nextSibling
                }
            }
        }
    }
}
""")


def _create_cats(gql_client, ic, cats_input):
    data = gql_client.query_data(
        CREATE_CATS,
        variables={
            'instanceId': str(ic.pk),
            'input': cats_input,
        },
    )
    return data['instanceEditor']['createDimensionCategories']


def test_create_single_category(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector')
    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'Energy'},
        ],
    )
    cats = result['categories']
    assert len(cats) == 1
    assert cats[0]['label'] == 'Energy'
    assert cats[0]['identifier'] is None
    assert cats[0]['previousSibling'] is None
    assert cats[0]['nextSibling'] is None


def test_create_bulk_categories_list_order(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Bulk create without sibling hints: list order determines ordering."""
    dim = _make_dimension(db_instance_config, 'sector', 'Sector')
    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'First'},
            {'dimensionId': str(dim.uuid), 'label': 'Second'},
            {'dimensionId': str(dim.uuid), 'label': 'Third'},
        ],
    )
    cats = result['categories']
    assert [c['label'] for c in cats] == ['First', 'Second', 'Third']


def test_create_category_with_identifier(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector')
    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'Energy', 'identifier': 'energy'},
        ],
    )
    assert result['categories'][0]['identifier'] == 'energy'


def test_create_category_duplicate_identifier_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Energy'])
    gql_client.query_errors(
        CREATE_CATS,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': [{'dimensionId': str(dim.uuid), 'label': 'New Energy', 'identifier': 'energy'}],
        },
    )


def test_create_category_with_client_uuid(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector')
    client_uuid = uuid4()
    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'Energy', 'id': str(client_uuid)},
        ],
    )
    assert result['categories'][0]['id'] == str(client_uuid)


def test_create_category_with_previous_sibling(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Insert a new category after an existing one."""
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Alpha', 'Gamma'])
    alpha = DimensionCategory.objects.get(dimension=dim, identifier='alpha')

    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'Beta', 'previousSibling': str(alpha.uuid)},
        ],
    )
    cats = result['categories']
    assert [c['label'] for c in cats] == ['Alpha', 'Beta', 'Gamma']


def test_create_category_with_next_sibling(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Insert a new category before an existing one."""
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Beta', 'Gamma'])
    beta = DimensionCategory.objects.get(dimension=dim, identifier='beta')

    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'Alpha', 'nextSibling': str(beta.uuid)},
        ],
    )
    cats = result['categories']
    assert [c['label'] for c in cats] == ['Alpha', 'Beta', 'Gamma']


def test_create_category_both_siblings_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['A', 'B'])
    a = DimensionCategory.objects.get(dimension=dim, identifier='a')
    b = DimensionCategory.objects.get(dimension=dim, identifier='b')
    gql_client.query_errors(
        CREATE_CATS,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': [{'dimensionId': str(dim.uuid), 'label': 'X', 'previousSibling': str(a.uuid), 'nextSibling': str(b.uuid)}],
        },
    )


def test_create_batch_with_chained_client_uuids(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Batch create with client UUIDs referencing earlier items in the same batch."""
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Existing'])
    existing = DimensionCategory.objects.get(dimension=dim, identifier='existing')

    uuid_a = uuid4()
    uuid_b = uuid4()
    result = _create_cats(
        gql_client,
        db_instance_config,
        [
            {'dimensionId': str(dim.uuid), 'label': 'A', 'id': str(uuid_a), 'previousSibling': str(existing.uuid)},
            {'dimensionId': str(dim.uuid), 'label': 'B', 'id': str(uuid_b), 'previousSibling': str(uuid_a)},
        ],
    )
    cats = result['categories']
    assert [c['label'] for c in cats] == ['Existing', 'A', 'B']


# ---------------------------------------------------------------------------
# updateDimensionCategories
# ---------------------------------------------------------------------------

UPDATE_CATS = gql("""
mutation UpdateCats($instanceId: ID!, $input: [UpdateDimensionCategoryInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        updateDimensionCategories(input: $input) {
            ... on InstanceDimension {
                id
                categories {
                    id
                    identifier
                    label
                    order
                    previousSibling
                    nextSibling
                }
            }
        }
    }
}
""")


def _update_cats(gql_client, ic, cats_input):
    data = gql_client.query_data(
        UPDATE_CATS,
        variables={
            'instanceId': str(ic.pk),
            'input': cats_input,
        },
    )
    return data['instanceEditor']['updateDimensionCategories']


def test_update_category_label(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Energy'])
    cat = DimensionCategory.objects.get(dimension=dim, identifier='energy')
    result = _update_cats(
        gql_client,
        db_instance_config,
        [
            {'categoryId': str(cat.uuid), 'label': 'Renewable Energy'},
        ],
    )
    assert result['categories'][0]['label'] == 'Renewable Energy'


def test_update_category_identifier(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['Energy'])
    cat = DimensionCategory.objects.get(dimension=dim, identifier='energy')
    result = _update_cats(
        gql_client,
        db_instance_config,
        [
            {'categoryId': str(cat.uuid), 'identifier': 'renewable'},
        ],
    )
    assert result['categories'][0]['identifier'] == 'renewable'


def test_move_category_to_front(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Move last category to front using nextSibling."""
    dim = _make_dimension(db_instance_config, 's', 'S', ['A', 'B', 'C'])
    a = DimensionCategory.objects.get(dimension=dim, identifier='a')
    c = DimensionCategory.objects.get(dimension=dim, identifier='c')

    result = _update_cats(
        gql_client,
        db_instance_config,
        [
            {'categoryId': str(c.uuid), 'nextSibling': str(a.uuid)},
        ],
    )
    assert [cat['label'] for cat in result['categories']] == ['C', 'A', 'B']


def test_move_category_to_middle(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Move first category to after second using previousSibling."""
    dim = _make_dimension(db_instance_config, 's', 'S', ['A', 'B', 'C'])
    a = DimensionCategory.objects.get(dimension=dim, identifier='a')
    b = DimensionCategory.objects.get(dimension=dim, identifier='b')

    result = _update_cats(
        gql_client,
        db_instance_config,
        [
            {'categoryId': str(a.uuid), 'previousSibling': str(b.uuid)},
        ],
    )
    assert [cat['label'] for cat in result['categories']] == ['B', 'A', 'C']


def test_bulk_reorder(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    """Full reorder: reverse A,B,C to C,B,A using previousSibling chain."""
    dim = _make_dimension(db_instance_config, 's', 'S', ['A', 'B', 'C'])
    a = DimensionCategory.objects.get(dimension=dim, identifier='a')
    b = DimensionCategory.objects.get(dimension=dim, identifier='b')
    c = DimensionCategory.objects.get(dimension=dim, identifier='c')

    result = _update_cats(
        gql_client,
        db_instance_config,
        [
            # C goes first (no hint needed — finalize will handle relative placement)
            # Actually, to reverse: B after C, A after B
            {'categoryId': str(b.uuid), 'previousSibling': str(c.uuid)},
            {'categoryId': str(a.uuid), 'previousSibling': str(b.uuid)},
        ],
    )
    assert [cat['label'] for cat in result['categories']] == ['C', 'B', 'A']


def test_update_nonexistent_category_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    gql_client.query_errors(
        UPDATE_CATS,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': [{'categoryId': str(uuid4()), 'label': 'X'}],
        },
    )


# ---------------------------------------------------------------------------
# deleteDimensionCategory
# ---------------------------------------------------------------------------

DELETE_CAT = gql("""
mutation DeleteCat($instanceId: ID!, $categoryId: UUID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteDimensionCategory(categoryId: $categoryId) {
            ... on OperationInfo {
                messages { message }
            }
        }
    }
}
""")


def test_delete_category(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector', ['A', 'B', 'C'])
    b = DimensionCategory.objects.get(dimension=dim, identifier='b')

    gql_client.query_data(
        DELETE_CAT,
        variables={
            'instanceId': str(db_instance_config.pk),
            'categoryId': str(b.uuid),
        },
    )
    remaining = list(DimensionCategory.objects.filter(dimension=dim).order_by('order').values_list('identifier', flat=True))
    assert remaining == ['a', 'c']


def test_delete_nonexistent_category_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    gql_client.query_errors(
        DELETE_CAT,
        variables={
            'instanceId': str(db_instance_config.pk),
            'categoryId': str(uuid4()),
        },
    )


# ---------------------------------------------------------------------------
# updateDimension
# ---------------------------------------------------------------------------

UPDATE_DIM = gql("""
mutation UpdateDim($instanceId: ID!, $input: UpdateDimensionInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateDimension(input: $input) {
            ... on InstanceDimension {
                id
                identifier
                name
            }
        }
    }
}
""")


def test_update_dimension_name(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    dim = _make_dimension(db_instance_config, 'sector', 'Sector')
    data = gql_client.query_data(
        UPDATE_DIM,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {'dimensionId': str(dim.uuid), 'name': 'Industry Sector'},
        },
    )
    result = data['instanceEditor']['updateDimension']
    assert result['name'] == 'Industry Sector'
    assert result['identifier'] == 'sector'
    dim.refresh_from_db()
    assert dim.name == 'Industry Sector'


def test_update_nonexistent_dimension_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    gql_client.query_errors(
        UPDATE_DIM,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {'dimensionId': str(uuid4()), 'name': 'X'},
        },
    )
