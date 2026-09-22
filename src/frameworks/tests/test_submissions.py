from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.exceptions import PermissionDenied

import pytest

from paths.tests.graphql import PathsTestClient

from datasets.validation import InstanceDatasetValidationError
from frameworks import submissions as ops
from frameworks.identity import AGS_PARAMETER, IdentityError, ensure_municipal_organization
from frameworks.models import Submission, SubmissionStatus
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.models import InstanceConfig
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from orgs.models import Organization
from params.param import StringParameter
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

pytestmark = pytest.mark.django_db


def make_instance(*, owner: str = 'Stadt Beispiel', ags: str | None = None, config_source: str = 'database') -> InstanceConfig:
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030))
    if ags is not None:
        spec.params.append(StringParameter(local_id=AGS_PARAMETER, label='AGS', value=ags))
    return InstanceConfigFactory.create(
        identifier=instance.id, instance=instance, config_source=config_source, owner=owner, spec=spec
    )


# --- identity ---


def test_municipal_organization_is_created_from_the_ags_parameter() -> None:
    ic = make_instance(owner='Landeshauptstadt Mainz', ags='07315000')
    fwc = FrameworkConfigFactory.create(instance_config=ic)

    org = ensure_municipal_organization(ic)

    assert org is not None
    assert org.name == 'Landeshauptstadt Mainz'
    assert list(org.identifiers.values_list('namespace__identifier', 'identifier')) == [('ags', '07315000')]
    ic.refresh_from_db()
    fwc.refresh_from_db()
    assert ic.organization == org
    assert (fwc.organization_name, fwc.organization_identifier) == ('Landeshauptstadt Mainz', '07315000')


def test_instances_with_the_same_ags_share_one_organization() -> None:
    first = make_instance(owner='Stadt A', ags='09462000')
    second = make_instance(owner='Stadt A (Kopie)', ags='09462000')
    org = ensure_municipal_organization(first)
    assert ensure_municipal_organization(second) == org
    assert Organization.objects.filter(identifiers__identifier='09462000').count() == 1


def test_instance_without_ags_is_left_alone() -> None:
    ic = make_instance()
    before = ic.organization_id
    assert ensure_municipal_organization(ic) is None
    ic.refresh_from_db()
    assert ic.organization_id == before


def test_malformed_ags_is_rejected() -> None:
    with pytest.raises(IdentityError):
        ensure_municipal_organization(make_instance(ags='7315000'))


# --- lifecycle ---


def test_submission_moves_from_draft_to_final_and_pins_the_published_revision() -> None:
    user = UserFactory.create(is_superuser=True)
    ic = make_instance()
    sub = ops.create_submission(ic, period_start=2021, user=user)
    assert (sub.status, sub.period_end, sub.supersedes) == (SubmissionStatus.DRAFT, 2021, None)

    with pytest.raises(ops.SubmissionError):
        ops.create_submission(ic, period_start=2021, user=user)
    with pytest.raises(ops.SubmissionError):
        ops.finalise(sub, user=user)

    sub = ops.request_review(sub, user=user)
    sub = ops.return_to_draft(sub, user=user)
    sub = ops.request_review(sub, user=user)
    sub = ops.finalise(sub, user=user)

    ic.refresh_from_db()
    assert sub.status == SubmissionStatus.FINAL
    assert sub.instance_revision_id is not None
    assert sub.instance_revision_id == ic.live_revision_id
    assert sub.finalised_by == user
    assert sub.finalised_at is not None
    with pytest.raises(ops.SubmissionError):
        ops.discard(sub)


def test_finalising_without_publish_permission_is_denied() -> None:
    sub = ops.request_review(ops.create_submission(make_instance(), period_start=2021, user=None), user=None)
    with pytest.raises(PermissionDenied):
        ops.finalise(sub, user=UserFactory.create())
    sub.refresh_from_db()
    assert sub.status == SubmissionStatus.IN_REVIEW


def test_correction_supersedes_the_final_submission_only_when_finalised() -> None:
    ic = make_instance()
    first = ops.finalise(ops.request_review(ops.create_submission(ic, period_start=2021, user=None), user=None), user=None)

    correction = ops.create_submission(ic, period_start=2021, user=None)
    assert correction.supersedes == first
    first.refresh_from_db()
    assert first.status == SubmissionStatus.FINAL

    correction = ops.finalise(ops.request_review(correction, user=None), user=None)
    first.refresh_from_db()
    assert first.status == SubmissionStatus.SUPERSEDED
    assert correction.status == SubmissionStatus.FINAL
    assert Submission.objects.filter(instance_config=ic, status=SubmissionStatus.FINAL).count() == 1


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'period_start': 2020, 'period_end': 2021}, 'single year'),
        ({'period_start': 2009}, 'before the first historical year'),
        ({'period_start': 2023}, 'after the last historical year'),
    ],
)
def test_create_validates_the_period(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ops.SubmissionError, match=message):
        ops.create_submission(make_instance(), user=None, **kwargs)


def test_yaml_instance_cannot_be_submitted() -> None:
    with pytest.raises(ops.SubmissionError, match='database-backed'):
        ops.create_submission(make_instance(config_source='yaml'), period_start=2021, user=None)


# --- GraphQL ---


MUTATE = """
mutation($id: ID!) { instanceEditor(instanceId: $id) { submissions { %s } } }
"""


@pytest.fixture
def ic() -> InstanceConfig:
    ic = make_instance(owner='Stadt Beispiel', ags='07111000')
    FrameworkConfigFactory.create(instance_config=ic, framework=FrameworkFactory.create(identifier='bisko-test'))
    ensure_municipal_organization(ic)
    return ic


def gql_for(client: Client, ic: InstanceConfig, *, superuser: bool) -> PathsTestClient:
    if superuser:
        client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(ic)
    return tc


def mutate(gql: PathsTestClient, ic: InstanceConfig, body: str) -> dict[str, Any]:
    return gql.query_data(MUTATE % body, variables={'id': ic.identifier})['instanceEditor']['submissions']


def test_graphql_lifecycle_and_blocked_finalisation(client: Client, ic: InstanceConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    gql = gql_for(client, ic, superuser=True)
    created = mutate(gql, ic, 'create(periodStart: 2021) { ... on Submission { id status periodEnd } }')['create']
    assert created['status'] == 'DRAFT'
    assert created['periodEnd'] == 2021
    sub_id = created['id']
    mutate(gql, ic, 'requestReview(submissionId: "%s") { __typename }' % sub_id)

    def refuse(self: InstanceConfig, user: Any = None) -> None:
        raise InstanceDatasetValidationError([])

    monkeypatch.setattr(InstanceConfig, 'publish_instance', refuse)
    result = mutate(gql, ic, 'finalise(submissionId: "%s") { __typename }' % sub_id)['finalise']
    assert result['__typename'] == 'DatasetValidationViolations'
    assert Submission.objects.get(uuid=sub_id).status == SubmissionStatus.IN_REVIEW

    monkeypatch.undo()
    result = mutate(gql, ic, 'finalise(submissionId: "%s") { ... on Submission { status finalisedAt } }' % sub_id)['finalise']
    assert result['status'] == 'FINAL'
    assert result['finalisedAt'] is not None


def test_open_submissions_are_hidden_from_readers(client: Client, ic: InstanceConfig) -> None:
    final = ops.finalise(ops.request_review(ops.create_submission(ic, period_start=2020, user=None), user=None), user=None)
    ops.create_submission(ic, period_start=2021, user=None)

    query = (
        '{ framework(identifier: "bisko-test") { configs {'
        ' organizationIdentifiers { namespace identifier } submissions { id status } } } }'
    )
    editor = gql_for(client, ic, superuser=True).query_data(query)
    (config,) = editor['framework']['configs']
    assert config['organizationIdentifiers'] == [{'namespace': 'ags', 'identifier': '07111000'}]
    assert {s['status'] for s in config['submissions']} == {'FINAL', 'DRAFT'}

    client.logout()
    reader = gql_for(client, ic, superuser=False).query_data('{ instance { submissions { id status } } }')
    assert reader['instance']['submissions'] == [{'id': str(final.uuid), 'status': 'FINAL'}]
