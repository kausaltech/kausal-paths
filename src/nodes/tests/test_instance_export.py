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
    original = InstanceSnapshot.from_serialized_data.__func__  # type: ignore[attr-defined]

    def spy(cls, data, *, compose=True):
        seen.append(data['schema_version'])
        return original(cls, data, compose=compose)

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


def test_a_document_with_v1_dataset_bodies_still_imports(db_instance: InstanceConfig) -> None:
    """
    A deployment still on `DatasetSnapshot` v1 exports wide tables keyed by identifier.

    `InstanceExport.from_serialized_data` upgrades them against the export's own catalog,
    so a production export can be imported locally across the change.
    """
    import datetime
    from decimal import Decimal

    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import DataPointComment, Dataset, DimensionScope
    from kausal_common.datasets.tests.factories import (
        DataPointFactory,
        DatasetFactory,
        DatasetMetricFactory,
        DatasetSchemaDimensionFactory,
        DimensionCategoryFactory,
        DimensionFactory,
    )

    dimension = DimensionFactory.create(name='Sector')
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(db_instance),
        scope_id=db_instance.pk,
        identifier='sector',
    )
    homes = DimensionCategoryFactory.create(dimension=dimension, identifier='homes', label='Homes')
    dataset = DatasetFactory.create(identifier='city/energy', scope=db_instance)
    DatasetSchemaDimensionFactory.create(schema=dataset.schema, dimension=dimension)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='energy', label='Energy', unit='MWh')
    DataPointFactory.create(
        dataset=dataset, metric=metric, date=datetime.date(2020, 1, 1), value=Decimal(7), dimension_categories=[homes]
    )

    document = json.loads(export_instance(db_instance).model_dump_json())
    document['instance']['schema_version'] = 14
    v1_body = {
        'schema_version': 1,
        'uuid': str(dataset.uuid),
        'identifier': 'city/energy',
        'name': {'en': 'Energy'},
        'dimensions': ['sector'],
        'metrics': [{'identifier': 'energy', 'label': {'en': 'Energy'}, 'unit': 'MWh', 'validation_rules': []}],
        'data': {
            'schema': {'fields': [{'name': 'Year'}, {'name': 'sector'}, {'name': 'energy', 'unit': 'MWh'}]},
            'data': [{'Year': 2020, 'sector': 'homes', 'energy': 7.0}],
        },
        'comments': [{'point': {'year': 2020, 'metric': 'energy', 'categories': ['homes']}, 'text': 'metered'}],
    }
    document['datasets'] = [v1_body]
    document['instance']['metadata'] |= {'identifier': 'downloaded', 'uuid': '00000000-0000-4000-8000-000000000002'}

    export = InstanceExport.from_serialized_data(document)
    (body,) = export.datasets
    assert body.schema_version == 2
    (point,) = body.points
    assert (point.value, point.categories) == (7.0, {dimension.uuid: homes.uuid})

    ic = import_instance_export(export, organization=str(db_instance.organization.uuid))
    imported = Dataset.objects.get(scope_id=ic.pk, identifier='city/energy')
    assert list(imported.data_points.values_list('value', flat=True)) == [Decimal(7)]
    assert list(DataPointComment.objects.filter(data_point__dataset=imported).values_list('text', flat=True)) == ['metered']


def test_every_uuid_in_an_export_says_what_it_is() -> None:
    """A copy can only rekey what it can classify; a new uuid field needs a kind (`paths.uuid_kinds`)."""
    from paths.uuid_kinds import unmarked_uuid_fields

    assert unmarked_uuid_fields(InstanceExport) == []


def test_a_rekeyed_export_takes_the_seeded_instance_uuid(db_instance: InstanceConfig) -> None:
    from uuid import uuid4

    target = uuid4()
    exported = export_instance(db_instance)
    copy, rekeying = exported.rekeyed(seed={db_instance.uuid: target})

    assert copy.instance.metadata.uuid == target
    assert copy.instance.copy_of == db_instance.uuid
    assert {node.identifier for node in copy.instance.nodes} == {'first_node', 'second_node'}
    assert {node.uuid for node in copy.instance.nodes}.isdisjoint({node.uuid for node in exported.instance.nodes})
    assert all(node.copy_of == source.uuid for node, source in zip(copy.instance.nodes, exported.instance.nodes, strict=True))
    assert rekeying.outside() == {}
    reloaded = InstanceExport.from_serialized_data(json.loads(json.dumps(copy.model_dump(mode='json'))))
    assert reloaded.model_dump(mode='json') == copy.model_dump(mode='json')
