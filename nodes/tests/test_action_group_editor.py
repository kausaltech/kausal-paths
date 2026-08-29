from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import ActionConfig, NodeSpec
from nodes.models import InstanceChangeOperation, InstanceModelLogEntry
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory

if TYPE_CHECKING:
    from django.test import Client

    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig


pytestmark = pytest.mark.django_db


ACTION_NODE_CLASS = 'nodes.actions.simple.AdditiveAction'


@pytest.fixture
def db_instance_config() -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=spec,
    )


@pytest.fixture
def gql_client(client: Client, db_instance_config: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    test_client = PathsTestClient(client)
    test_client.set_instance(db_instance_config)
    return test_client


ACTION_GROUP_FIELDS = """
fragment ActionGroupFields on ActionGroupType {
    uuid
    id
    identifier
    name
    color
    order
    previousSibling
    nextSibling
}
"""

CREATE_ACTION_GROUP = (
    ACTION_GROUP_FIELDS
    + """
mutation CreateActionGroup($instanceId: ID!, $input: CreateActionGroupInput!) {
    instanceEditor(instanceId: $instanceId) {
        createActionGroup(input: $input) {
            ... on ActionGroupType { ...ActionGroupFields }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""
)

UPDATE_ACTION_GROUP = (
    ACTION_GROUP_FIELDS
    + """
mutation UpdateActionGroup($instanceId: ID!, $id: UUID!, $input: UpdateActionGroupInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateActionGroup(id: $id, input: $input) {
            ... on ActionGroupType { ...ActionGroupFields }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""
)

DELETE_ACTION_GROUP = """
mutation DeleteActionGroup($instanceId: ID!, $id: UUID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteActionGroup(id: $id) { messages { kind message } }
    }
}
"""

QUERY_ACTION_GROUPS = (
    ACTION_GROUP_FIELDS
    + """
query ActionGroups {
    instance {
        editor {
            actionGroups { ...ActionGroupFields }
        }
    }
}
"""
)


def _create_group(
    gql_client: PathsTestClient,
    instance: InstanceConfig,
    identifier: str,
    **position: str,
) -> dict[str, object]:
    data = gql_client.query_data(
        CREATE_ACTION_GROUP,
        variables={
            'instanceId': instance.identifier,
            'input': {'identifier': identifier, 'name': identifier.title(), **position},
        },
    )
    return data['instanceEditor']['createActionGroup']


def test_action_group_crud_and_sibling_ordering(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    first = _create_group(gql_client, db_instance_config, 'first')
    third = _create_group(gql_client, db_instance_config, 'third')
    second = _create_group(
        gql_client,
        db_instance_config,
        'second',
        previousSibling=str(first['uuid']),
        nextSibling=str(third['uuid']),
    )

    data = gql_client.query_data(QUERY_ACTION_GROUPS)
    groups = data['instance']['editor']['actionGroups']
    assert [group['identifier'] for group in groups] == ['first', 'second', 'third']
    assert [group['id'] for group in groups] == ['first', 'second', 'third']
    assert [group['order'] for group in groups] == [0, 1, 2]
    assert groups[0]['previousSibling'] is None
    assert groups[0]['nextSibling'] == second['uuid']
    assert groups[1]['previousSibling'] == first['uuid']
    assert groups[1]['nextSibling'] == third['uuid']

    data = gql_client.query_data(
        UPDATE_ACTION_GROUP,
        variables={
            'instanceId': db_instance_config.identifier,
            'id': second['uuid'],
            'input': {
                'identifier': 'renamed',
                'name': 'Renamed group',
                'color': '#123456',
                'nextSibling': first['uuid'],
            },
        },
    )
    updated = data['instanceEditor']['updateActionGroup']
    assert updated['uuid'] == second['uuid']
    assert updated['id'] == 'renamed'
    assert updated['identifier'] == 'renamed'
    assert updated['name'] == 'Renamed group'
    assert updated['color'] == '#123456'
    assert updated['order'] == 0

    gql_client.query_data(
        DELETE_ACTION_GROUP,
        variables={'instanceId': db_instance_config.identifier, 'id': third['uuid']},
    )
    db_instance_config.refresh_from_db()
    assert db_instance_config.spec is not None
    assert [(group.id, group.order) for group in db_instance_config.spec.action_groups] == [
        ('renamed', 0),
        ('first', 1),
    ]

    operations = list(InstanceChangeOperation.objects.filter(instance_config=db_instance_config).order_by('created_at'))
    assert [operation.action for operation in operations] == [
        'action_group.create',
        'action_group.create',
        'action_group.create',
        'action_group.update',
        'action_group.delete',
    ]
    assert list(
        InstanceModelLogEntry.objects.filter(operation__in=operations).order_by('id').values_list('target_uuid', flat=True)
    ) == [
        UUID(str(first['uuid'])),
        UUID(str(third['uuid'])),
        UUID(str(second['uuid'])),
        UUID(str(second['uuid'])),
        UUID(str(third['uuid'])),
    ]


def test_delete_action_group_rejects_referenced_group(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    group = _create_group(gql_client, db_instance_config, 'referenced')
    NodeConfigFactory.create(
        instance=db_instance_config,
        identifier='referencing_action',
        spec=NodeSpec(
            type_config=ActionConfig(
                node_class=ACTION_NODE_CLASS,
                group=UUID(str(group['uuid'])),
            ),
        ),
    )

    gql_client.query_errors(
        DELETE_ACTION_GROUP,
        variables={'instanceId': db_instance_config.identifier, 'id': group['uuid']},
        assert_error_message='referencing_action',
    )
    db_instance_config.refresh_from_db()
    assert db_instance_config.spec is not None
    assert [candidate.uuid for candidate in db_instance_config.spec.action_groups] == [UUID(str(group['uuid']))]


def test_action_group_rejects_inconsistent_sibling_pair(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    first = _create_group(gql_client, db_instance_config, 'first')
    _create_group(gql_client, db_instance_config, 'middle')
    third = _create_group(gql_client, db_instance_config, 'third')

    gql_client.query_errors(
        CREATE_ACTION_GROUP,
        variables={
            'instanceId': db_instance_config.identifier,
            'input': {
                'identifier': 'invalid',
                'name': 'Invalid',
                'previousSibling': first['uuid'],
                'nextSibling': third['uuid'],
            },
        },
        assert_error_message='do not describe one gap',
    )


def test_create_action_group_accepts_client_uuid(
    gql_client: PathsTestClient,
    db_instance_config: InstanceConfig,
) -> None:
    group_uuid = uuid4()
    data = gql_client.query_data(
        CREATE_ACTION_GROUP,
        variables={
            'instanceId': db_instance_config.identifier,
            'input': {
                'id': str(group_uuid),
                'identifier': 'client_identity',
                'name': 'Client identity',
            },
        },
    )

    assert data['instanceEditor']['createActionGroup']['uuid'] == str(group_uuid)
