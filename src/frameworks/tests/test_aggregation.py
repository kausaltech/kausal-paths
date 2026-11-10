"""
Aggregation over the organization tree.

The tree is deliberately not a German one: the framework declares its own
reporting units, and nothing in the aggregation may know what a Gemeinde is.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied
from wagtail.models import Revision

import pytest

from kausal_common.people.models import ObjectRole

from paths.tests.graphql import PathsTestClient

from frameworks.aggregation import Aggregation, AggregationBasis, AggregationError, ExclusionReason, aggregate
from frameworks.models import (
    Framework,
    FrameworkOrganizationRoot,
    OrganizationAccessGrant,
    OrganizationPopulation,
    ReportingSpec,
    ReportingUnitSpec,
    Submission,
    SubmissionStatus,
)
from frameworks.organization_access import ViewerScope, viewer_scope
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.models import InstanceConfig, InstanceMemberAssignment, InstanceMemberRole
from nodes.tests.factories import InstanceConfigFactory
from orgs.models import Organization, OrganizationClass
from orgs.tests.factories import OrganizationFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from users.models import User

pytestmark = pytest.mark.django_db

EVERYTHING = ViewerScope(everything=True, paths=())
YEAR = 2023


@dataclass
class Tree:
    framework: Framework
    orgs: dict[str, Organization]
    instances: dict[str, InstanceConfig]

    def aggregate(
        self, name: str, *, scope: ViewerScope = EVERYTHING, year: int = YEAR, first: int = 100, offset: int = 0
    ) -> Aggregation:
        return aggregate(self.framework, self.orgs[name], year=year, scope=scope, first=first, offset=offset)

    def member(self, name: str, role: InstanceMemberRole = InstanceMemberRole.EDITOR) -> User:
        user = UserFactory.create()
        InstanceMemberAssignment.objects.create(instance_config=self.instances[name], user=user, role=role)
        return user


def _submit(instance: InstanceConfig, status: SubmissionStatus, *, year: int = YEAR) -> Submission:
    if status != SubmissionStatus.FINAL:
        return Submission.objects.create(instance_config=instance, period_start=year, period_end=year, status=status)
    content_type = ContentType.objects.get_for_model(InstanceConfig)
    revision = Revision.objects.create(
        content_type=content_type, base_content_type=content_type, object_id=str(instance.pk), content={}
    )
    return Submission.objects.create(
        instance_config=instance,
        period_start=year,
        period_end=year,
        status=status,
        instance_revision=revision,
        finalised_at=datetime(year + 1, 3, 1, tzinfo=UTC),
    )


@pytest.fixture
def tree() -> Tree:
    """
    Build a state with two districts, an association and a city that is its own district.

    state
    ├── north (district), with a balance of its own
    │   ├── a   final
    │   ├── b   draft
    │   ├── c   no instance
    │   └── union (association)
    │       ├── d   final, with a correction in review
    │       └── e   no submission for the year, final for the year before
    ├── south (district)
    │   ├── f   final
    │   └── g   no instance, no population observation
    └── city (city territory)
        └── h   in review
    """
    classes = {
        identifier: OrganizationClass.objects.create(identifier=identifier, name=identifier)
        for identifier in ('state', 'district', 'association', 'city_territory', 'town')
    }
    framework = FrameworkFactory.create(reporting=ReportingSpec(units=ReportingUnitSpec(classes=['town'])))
    orgs: dict[str, Organization] = {}

    def add(name: str, classification: str, parent: str | None = None) -> None:
        orgs[name] = OrganizationFactory.create(
            name=name, classification=classes[classification], parent=orgs[parent] if parent else None
        )

    add('state', 'state')
    add('north', 'district', 'state')
    for town in 'abc':
        add(town, 'town', 'north')
    add('union', 'association', 'north')
    for town in 'de':
        add(town, 'town', 'union')
    add('south', 'district', 'state')
    for town in 'fg':
        add(town, 'town', 'south')
    add('city', 'city_territory', 'state')
    add('h', 'town', 'city')
    FrameworkOrganizationRoot.objects.create(
        framework=framework, organization=orgs['state'], territorial_reference_date=date(2024, 12, 31)
    )

    instances: dict[str, InstanceConfig] = {}
    for name in ('a', 'b', 'd', 'e', 'f', 'h', 'north'):
        instances[name] = InstanceConfigFactory.create(
            identifier=f'aggregation-{name}', name=name, organization=orgs[name], config_source='database'
        )
        FrameworkConfigFactory.create(framework=framework, instance_config=instances[name])
    for name in 'abcdefh':
        OrganizationPopulation.objects.create(
            framework=framework, organization=orgs[name], year=YEAR, value=100, source_dataset='test', source_revision='pin'
        )

    _submit(instances['a'], SubmissionStatus.FINAL)
    _submit(instances['b'], SubmissionStatus.DRAFT)
    _submit(instances['d'], SubmissionStatus.FINAL)
    _submit(instances['d'], SubmissionStatus.IN_REVIEW)
    _submit(instances['e'], SubmissionStatus.FINAL, year=YEAR - 1)
    _submit(instances['f'], SubmissionStatus.FINAL)
    _submit(instances['h'], SubmissionStatus.IN_REVIEW)
    _submit(instances['north'], SubmissionStatus.FINAL)
    return Tree(framework, orgs, instances)


def test_state_lists_its_districts_and_counts_only_reporting_units(tree: Tree) -> None:
    result = tree.aggregate('state')

    assert result.basis == AggregationBasis.PUBLISHED_SUM
    assert result.territorial_reference_date == date(2024, 12, 31)
    assert result.population_source_revision == 'pin'
    assert not result.members_restricted
    assert result.member_count == 3
    assert [entry.organization.name for entry in result.members] == ['north', 'south', 'h']

    own = result.own.coverage
    assert (own.reporting_units, own.activated, own.finalised) == (8, 6, 3)
    # g has no observation, so the territory's population is only partly known.
    assert (own.population, own.partial_population, own.finalised_population) == (None, 700, 300)

    north = result.members[0]
    assert not north.is_reporting_unit
    assert north.exclusion_reason is None
    assert (north.coverage.reporting_units, north.coverage.activated, north.coverage.finalised) == (5, 4, 2)
    assert (north.coverage.population, north.coverage.finalised_population) == (500, 200)


def test_a_territory_balance_stands_beside_the_sum_and_is_not_added(tree: Tree) -> None:
    north = tree.aggregate('state').members[0]

    assert north.final_submission is not None
    assert north.final_submission.instance_config == tree.instances['north']
    assert north.instance_identifier == 'aggregation-north'
    # The district's own balance covers the same ground as its towns: two towns are final, not three.
    assert north.coverage.finalised == 2
    assert north.coverage.activated == 4


def test_a_territory_holding_one_unit_is_presented_as_that_unit(tree: Tree) -> None:
    city = tree.aggregate('state').members[2]

    assert city.organization == tree.orgs['h']
    assert city.collapsed_from == tree.orgs['city']
    assert city.is_reporting_unit
    assert city.open_status == SubmissionStatus.IN_REVIEW
    assert city.exclusion_reason == ExclusionReason.NOT_FINAL
    assert city.coverage.reporting_units == 1

    # Asked for directly, the territory still lists its unit.
    direct = tree.aggregate('city')
    assert [entry.organization.name for entry in direct.members] == ['h']
    assert direct.members[0].collapsed_from is None


def test_unit_members_state_why_they_are_not_counted(tree: Tree) -> None:
    result = tree.aggregate('north')
    members = {entry.organization.name: entry for entry in result.members}

    assert list(members) == ['a', 'b', 'c', 'union']
    assert members['a'].exclusion_reason is None
    assert members['a'].final_submission is not None
    assert members['a'].open_status is None
    assert (members['b'].exclusion_reason, members['b'].open_status) == (ExclusionReason.NOT_FINAL, SubmissionStatus.DRAFT)
    assert members['c'].exclusion_reason == ExclusionReason.NO_INSTANCE
    assert members['c'].latest_year is None
    assert not members['union'].is_reporting_unit
    assert members['union'].coverage.finalised == 1

    union = {entry.organization.name: entry for entry in tree.aggregate('union').members}
    # A correction in review leaves the final submission counted.
    assert (union['d'].exclusion_reason, union['d'].open_status) == (None, SubmissionStatus.IN_REVIEW)
    assert union['d'].final_submission is not None
    assert (union['e'].exclusion_reason, union['e'].latest_year) == (ExclusionReason.NO_SUBMISSION, YEAR - 1)


def test_members_are_paged_and_the_total_is_not(tree: Tree) -> None:
    result = tree.aggregate('north', first=2, offset=1)

    assert [entry.organization.name for entry in result.members] == ['b', 'c']
    assert result.member_count == 4
    assert result.own.coverage.reporting_units == 5


def test_units_count_only_in_the_years_they_existed(tree: Tree) -> None:
    Organization.objects.filter(pk=tree.orgs['c'].pk).update(dissolution_date=date(YEAR - 1, 12, 31))
    Organization.objects.filter(pk=tree.orgs['b'].pk).update(founding_date=date(YEAR, 1, 1))

    this_year = tree.aggregate('north')
    assert this_year.own.coverage.reporting_units == 4
    assert [entry.organization.name for entry in this_year.members] == ['a', 'b', 'union']

    year_before = tree.aggregate('north', year=YEAR - 1)
    assert year_before.own.coverage.reporting_units == 4
    assert [entry.organization.name for entry in year_before.members] == ['a', 'c', 'union']
    assert year_before.own.coverage.finalised == 1


def test_a_unit_account_sees_the_totals_above_it_but_not_its_peers(tree: Tree) -> None:
    scope = viewer_scope(tree.framework, tree.member('a'))

    own = tree.aggregate('a', scope=scope)
    assert not own.members_restricted
    assert own.own.final_submission is not None

    north = tree.aggregate('north', scope=scope)
    assert north.members_restricted
    assert [entry.organization.name for entry in north.members] == ['a']
    assert north.members[0].final_submission is not None
    assert north.member_count == 1
    assert north.own.details_withheld
    assert not north.own.figures_withheld
    # The district's own balance is not the viewer's to read; its coverage is.
    assert north.own.final_submission is None
    assert north.own.instance_identifier is None
    assert north.own.coverage.finalised == 2

    state = tree.aggregate('state', scope=scope)
    assert [entry.organization.name for entry in state.members] == ['north']
    assert state.members[0].details_withheld
    assert state.members[0].coverage.finalised == 2

    with pytest.raises(PermissionDenied):
        tree.aggregate('south', scope=scope)
    with pytest.raises(PermissionDenied):
        tree.aggregate('b', scope=scope)


def test_totals_of_a_small_territory_are_withheld_from_its_members(tree: Tree) -> None:
    scope = viewer_scope(tree.framework, tree.member('f'))

    south = tree.aggregate('south', scope=scope)

    # Two units: what the other one did follows from the total and the viewer's own.
    assert south.own.figures_withheld
    coverage = south.own.coverage
    assert (coverage.reporting_units, coverage.activated, coverage.finalised) == (2, None, None)
    assert coverage.finalised_population is None
    assert coverage.partial_population == 100
    assert [entry.organization.name for entry in south.members] == ['f']
    assert not south.members[0].figures_withheld

    # A subtree grant on the same territory reads it in full.
    reader = UserFactory.create()
    OrganizationAccessGrant.objects.create(
        framework=tree.framework, organization=tree.orgs['south'], user=reader, role=ObjectRole.VIEWER
    )
    granted = tree.aggregate('south', scope=viewer_scope(tree.framework, reader))
    assert not granted.own.figures_withheld
    assert granted.own.coverage.finalised == 1
    assert [entry.organization.name for entry in granted.members] == ['f', 'g']


def test_a_suspended_membership_reads_nothing(tree: Tree) -> None:
    user = tree.member('a')
    InstanceMemberAssignment.objects.filter(user=user).update(
        suspended_at=datetime(2026, 1, 1, tzinfo=UTC), retention_until=date(2027, 1, 1)
    )

    with pytest.raises(PermissionDenied):
        tree.aggregate('north', scope=viewer_scope(tree.framework, user))


def test_a_framework_without_reporting_units_has_no_aggregation(tree: Tree) -> None:
    tree.framework.reporting = ReportingSpec()
    tree.framework.save(update_fields=['reporting'])

    with pytest.raises(AggregationError, match='declares no reporting units'):
        tree.aggregate('state')


AGGREGATION_QUERY = """
    query ($id: ID!, $year: Int!) {
      framework(identifier: "%s") {
        organization(id: $id) {
          name
          reportingUnitCount
          ancestors {
            name
            aggregation(year: $year) {
              kind year basis territorialReferenceDate memberCount membersRestricted
              own {
                detailsWithheld figuresWithheld
                finalSubmission { id }
                coverage { reportingUnitCount activatedCount finalisedCount population finalisedPopulation }
              }
              members {
                organization { name reportingUnitCount }
                collapsedFrom { name }
                isReportingUnit exclusionReason openStatus latestYear
                finalSubmission { status periodStart }
                coverage { reportingUnitCount finalisedCount }
              }
            }
          }
        }
      }
    }
"""


def test_graphql_gives_a_unit_account_its_ancestors(tree: Tree, client: Client) -> None:
    client.force_login(tree.member('a'))
    gql = PathsTestClient(client)
    query = AGGREGATION_QUERY % tree.framework.identifier

    organization = gql.query_data(query, variables={'id': str(tree.orgs['a'].uuid), 'year': YEAR})['framework']['organization']

    assert (organization['name'], organization['reportingUnitCount']) == ('a', 1)
    assert [ancestor['name'] for ancestor in organization['ancestors']] == ['state', 'north']
    north = organization['ancestors'][1]['aggregation']
    assert north == {
        'kind': 'INVENTORY',
        'year': YEAR,
        'basis': 'PUBLISHED_SUM',
        'territorialReferenceDate': '2024-12-31',
        'memberCount': 1,
        'membersRestricted': True,
        'own': {
            'detailsWithheld': True,
            'figuresWithheld': False,
            'finalSubmission': None,
            'coverage': {
                'reportingUnitCount': 5,
                'activatedCount': 4,
                'finalisedCount': 2,
                'population': 500,
                'finalisedPopulation': 200,
            },
        },
        'members': [
            {
                'organization': {'name': 'a', 'reportingUnitCount': 1},
                'collapsedFrom': None,
                'isReportingUnit': True,
                'exclusionReason': None,
                'openStatus': None,
                'latestYear': YEAR,
                'finalSubmission': {'status': 'FINAL', 'periodStart': YEAR},
                'coverage': {'reportingUnitCount': 1, 'finalisedCount': 1},
            }
        ],
    }
    # A peer's organization is not the account's to fetch, and neither is the territory itself.
    for other in ('b', 'north'):
        found = gql.query_data(query, variables={'id': str(tree.orgs[other].uuid), 'year': YEAR})
        assert found['framework']['organization'] is None


def test_graphql_refuses_an_aggregation_outside_the_viewers_reach(tree: Tree, client: Client) -> None:
    reader = UserFactory.create()
    OrganizationAccessGrant.objects.create(
        framework=tree.framework, organization=tree.orgs['south'], user=reader, role=ObjectRole.VIEWER
    )
    client.force_login(reader)
    gql = PathsTestClient(client)
    query = (
        """
        query ($id: ID!) {
          framework(identifier: "%s") {
            organization(id: $id) {
              aggregation(year: 2023) { members { organization { name } collapsedFrom { name } } }
              ancestors { name }
            }
          }
        }
    """
        % tree.framework.identifier
    )

    south = gql.query_data(query, variables={'id': str(tree.orgs['south'].uuid)})['framework']['organization']
    assert [member['organization']['name'] for member in south['aggregation']['members']] == ['f', 'g']
    assert [ancestor['name'] for ancestor in south['ancestors']] == ['state']
