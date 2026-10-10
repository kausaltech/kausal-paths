"""A dataset reads from the edition of the instance source it is asked about, not from its live row."""

from datetime import date

import pytest

from kausal_common.datasets.models import DataPoint, DatasetMetricValidationRule

from datasets.editions import DatasetEdition, DatasetEditions, evaluate_edition_rules
from datasets.graphql.types import DatasetType
from datasets.tests.test_data_entry import make_query
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


def _add_ceiling(query_dataset_id) -> None:
    dataset = DataPoint.objects.filter(dataset__uuid=query_dataset_id).first()
    assert dataset is not None
    DatasetMetricValidationRule.objects.create(
        metric=dataset.metric, rule={'kind': 'value_range', 'enforcement': 'block_publish', 'max': 50}
    )


def test_a_pinned_edition_reads_the_revision_while_the_row_moves_on() -> None:
    query = make_query(pinned=True)
    id = query.graph.datasets[0].id
    edition = query.editions.get(id)
    assert edition is not None
    assert not edition.is_live
    assert edition.revision is not None
    dataset = DatasetType.from_edition(edition)
    assert sorted(point.value for point in dataset.data_points() if point.value is not None) == [4, 8]
    # The live row holds 99 for 2023; the edition does not see it.
    assert DataPoint.objects.get(dataset__uuid=id, date=date(2023, 1, 1)).value == 99
    # Nothing on the live row is offered for editing through a frozen edition.
    assert dataset._model is None
    assert not dataset.schema_is_editable()
    assert dataset.last_modified_at() == edition.revision.created_at


def test_the_draft_edition_reads_the_live_row() -> None:
    query = make_query()
    edition = query.editions.get(query.graph.datasets[0].id)
    assert edition is not None
    assert edition.is_live
    dataset = DatasetType.from_edition(edition)
    assert dataset._model is edition.row
    assert sorted(point.value for point in dataset.data_points() if point.value is not None) == [4, 8]
    assert [point.id for point in dataset.data_points()] == [
        str(point.uuid) for point in DataPoint.objects.filter(dataset=edition.row).order_by('date')
    ]


def test_rules_travel_with_the_content() -> None:
    """A frozen edition is checked against the rules it was recorded with, not the ones added since."""
    pinned = make_query(pinned=True)
    _add_ceiling(pinned.graph.datasets[0].id)
    edition = pinned.editions.get(pinned.graph.datasets[0].id)
    assert edition is not None
    assert evaluate_edition_rules(edition) == []

    live = DatasetEdition.from_row(edition.row) if edition.row is not None else None
    assert live is not None
    violations = evaluate_edition_rules(live)
    assert [violation.kind for violation in violations] == ['value_range']
    assert violations[0].dataset_uuid == edition.id


def test_a_published_edition_without_a_pin_fails_only_when_read() -> None:
    query = make_query()
    query.editions.published = True
    edition = query.editions.get(query.graph.datasets[0].id)
    assert edition is not None
    assert not edition.is_live
    dataset = DatasetType.from_edition(edition)
    # Its structure is still known.
    assert edition.meta.metrics
    with pytest.raises(ValueError, match='no revision pin'):
        dataset.data_points()


def test_a_reader_without_access_does_not_list_the_dataset() -> None:
    """The graph names the dataset to anyone reading the instance source; a listing still follows the dataset policy."""
    query = make_query()
    editions = DatasetEditions(query.graph, UserFactory.create())
    assert editions.get(query.graph.datasets[0].id) is not None
    assert editions.all() == []
    assert editions.viewable == {}
