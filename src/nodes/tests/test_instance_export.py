"""The ``instance.export`` GraphQL field, export provenance, and loading a document back in."""

import json
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from paths.tests.graphql import PathsTestClient

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.instance_import import InstanceImportError, import_instance_export
from nodes.instance_serialization import InstanceExport, InstanceSnapshot, export_instance
from nodes.models import InstanceConfig, NodeConfig
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from orgs.tests.factories import OrganizationFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from users.models import User

pytestmark = pytest.mark.django_db

EXPORT_QUERY = '{ instance { identifier export } }'


@pytest.fixture
def db_instance() -> InstanceConfig:
    instance = InstanceFactory.create(id='export-src')
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030))
    ic = InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        name='Export source',
        config_source='database',
        spec=spec,
    )
    NodeConfigFactory.create(instance=ic, identifier='first_node', name='First node')
    NodeConfigFactory.create(instance=ic, identifier='second_node', name='Second node')
    return ic


@pytest.fixture
def superuser() -> User:
    return UserFactory.create(is_superuser=True)


def _gql(client: Client, user: User, ic: InstanceConfig) -> PathsTestClient:
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def test_superuser_gets_a_loadable_export_document(client: Client, superuser: User, db_instance: InstanceConfig) -> None:
    data = _gql(client, superuser, db_instance).query_data(EXPORT_QUERY)

    document = data['instance']['export']
    assert document['instance']['metadata']['identifier'] == db_instance.identifier
    assert sorted(n['identifier'] for n in document['instance']['nodes']) == ['first_node', 'second_node']
    assert document['exported_at'] is not None
    assert document['exported_from'] == 'http://testserver/'
    assert document['draft_head_token'] is None, 'no change operations recorded yet'

    reloaded = InstanceExport.from_serialized_data(document)
    assert reloaded.model_dump(mode='json') == document, 'the document is a fixed point of dump/load'
    from_text = InstanceExport.model_validate_json(json.dumps(document))
    assert from_text.model_dump(mode='json') == document, 'JSON-mode validation agrees with the dict path'


def test_instance_admin_is_not_enough(client: Client, db_instance: InstanceConfig) -> None:
    admin = UserFactory.create()
    db_instance.permission_policy().admin_role.assign_user(db_instance, admin)

    errors = _gql(client, admin, db_instance).query_errors(EXPORT_QUERY, assert_error_message='superuser')

    assert errors[0].get('path') == ['instance', 'export']


def test_export_records_provenance_only_when_asked(db_instance: InstanceConfig) -> None:
    in_process = export_instance(db_instance)
    assert in_process.exported_from is None
    assert in_process.exported_at is not None

    outbound = export_instance(db_instance, exported_from='https://paths.example/')
    assert outbound.exported_from == 'https://paths.example/'


def test_from_serialized_data_runs_snapshot_upgraders(db_instance: InstanceConfig, monkeypatch) -> None:
    document = export_instance(db_instance).model_dump(mode='json')
    seen: list[int] = []
    original = InstanceSnapshot.from_serialized_data.__func__

    def spy(cls, data):
        seen.append(data['schema_version'])
        return original(cls, data)

    monkeypatch.setattr(InstanceSnapshot, 'from_serialized_data', classmethod(spy))

    InstanceExport.from_serialized_data(document)

    assert seen == [document['instance']['schema_version']]


def test_import_creates_a_database_instance_preserving_identity(db_instance: InstanceConfig) -> None:
    export = export_instance(db_instance)
    remote_uuid = UUID('00000000-0000-4000-8000-000000000001')
    export = export.model_copy(
        update={
            'instance': export.instance.model_copy(
                update={
                    'metadata': export.instance.metadata.model_copy(update={'identifier': 'downloaded', 'uuid': remote_uuid}),
                }
            )
        }
    )

    ic = import_instance_export(export, organization=str(db_instance.organization.uuid))

    assert ic.identifier == 'downloaded'
    assert ic.uuid == remote_uuid, 'a free UUID is kept so the mirror stays the same entity'
    assert ic.config_source == 'database'
    assert ic.organization == db_instance.organization
    assert ic.name == 'Export source (downloaded)', 'the source name is taken, so it is disambiguated'
    assert sorted(ic.nodes.values_list('identifier', flat=True)) == ['first_node', 'second_node']


def test_import_into_taken_uuid_mints_a_new_one(db_instance: InstanceConfig) -> None:
    export = export_instance(db_instance)

    ic = import_instance_export(export, identifier='mirror', name='Mirror', organization=db_instance.organization.name)

    assert ic.uuid != db_instance.uuid
    assert ic.name == 'Mirror'


def test_import_refuses_a_populated_target(db_instance: InstanceConfig) -> None:
    export = export_instance(db_instance)

    with pytest.raises(InstanceImportError, match='already has 2 nodes'):
        import_instance_export(export, identifier=db_instance.identifier)

    assert NodeConfig.objects.filter(instance=db_instance).count() == 2


def test_import_needs_an_organization_choice_when_ambiguous(db_instance: InstanceConfig) -> None:
    other = OrganizationFactory.create(name='Other org')
    export = export_instance(db_instance)

    with pytest.raises(InstanceImportError, match='Several organizations'):
        import_instance_export(export, identifier='ambiguous')

    ic = import_instance_export(export, identifier='chosen', organization=str(other.uuid))
    assert ic.organization == other
