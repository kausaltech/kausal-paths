"""Upgrading v1 dataset payloads (wide table, natural keys) to the uuid-keyed v2 form."""

from datetime import date
from uuid import uuid4

import pytest

from kausal_common.i18n.pydantic import set_i18n_context

from datasets.legacy_snapshot import LiveIdentities, upgrade_dataset_snapshot_v1
from datasets.snapshot import DatasetSnapshot
from nodes.defs.graph import DimensionCategoryMeta, DimensionMeta

SECTOR = DimensionMeta(
    id=uuid4(),
    identifier='sector',
    categories=(DimensionCategoryMeta(id=uuid4(), identifier='homes'), DimensionCategoryMeta(id=uuid4(), identifier='shops')),
)
HOMES, SHOPS = (category.id for category in SECTOR.categories)

pytestmark = pytest.mark.django_db


def v1_payload(rows: list[dict], metrics: list[dict], **extra) -> dict:
    fields = [{'name': 'Year'}, {'name': 'sector'}, *({'name': m.get('column', m['identifier'])} for m in metrics)]
    return {
        'schema_version': 1,
        'uuid': str(uuid4()),
        'identifier': 'test/energy',
        'name': {'en': 'Energy'},
        'dimensions': ['sector'],
        'metrics': [{k: v for k, v in m.items() if k != 'column'} | {'unit': 'MWh'} for m in metrics],
        'data': {'schema': {'fields': fields}, 'data': rows},
        **extra,
    }


def upgrade(content: dict, identities: LiveIdentities | None = None) -> DatasetSnapshot:
    with set_i18n_context('en', []):
        return DatasetSnapshot.model_validate(
            upgrade_dataset_snapshot_v1(content, base=None, dimensions=[SECTOR], identities=identities)
        )


def test_cells_become_points_with_category_uuids() -> None:
    snapshot = upgrade(
        v1_payload(
            [{'Year': 2020, 'sector': 'homes', 'value': 1.5}, {'Year': 2020, 'sector': 'shops', 'value': None}],
            [{'identifier': 'value'}],
        )
    )
    assert snapshot.meta.declared_dimension_ids == (SECTOR.id,)
    assert {(p.date, p.categories[SECTOR.id], p.value) for p in snapshot.points} == {
        (date(2020, 1, 1), HOMES, 1.5),
        (date(2020, 1, 1), SHOPS, None),  # an empty cell stays a cell
    }


def test_the_same_payload_always_upgrades_to_the_same_content() -> None:
    """Without live rows to borrow from, uuids are derived, so a hash over the result is stable."""
    content = v1_payload([{'Year': 2020, 'sector': 'homes', 'value': 1.0}], [{'identifier': 'value'}])
    assert upgrade(content).model_dump(mode='json') == upgrade(content).model_dump(mode='json')


def test_live_rows_lend_their_uuids() -> None:
    content = v1_payload(
        [{'Year': 2020, 'sector': 'homes', 'value': 1.0}],
        [{'identifier': 'value'}],
        comments=[{'point': {'year': 2020, 'metric': 'value', 'categories': ['homes']}, 'text': 'metered'}],
    )
    derived = upgrade(content)
    (point,) = derived.points
    point_id, comment_id = uuid4(), uuid4()
    live = LiveIdentities(points={(point.metric, 2020, frozenset({HOMES})): point_id})
    live.comments[point_id, 'metered'].append(comment_id)
    (borrowed,) = upgrade(content, live).points
    assert borrowed.id == point_id
    assert [comment.id for comment in borrowed.comments] == [comment_id]


def test_a_split_row_lands_as_one_point_per_cell() -> None:
    """Two halves of a split row each hold the other's cell empty; the valued one wins."""
    snapshot = upgrade(
        v1_payload(
            [
                {'Year': 2020, 'sector': 'homes', 'value': 10.5, 'quality': None},
                {'Year': 2020, 'sector': 'homes', 'value': None, 'quality': 3.0},
            ],
            [{'identifier': 'value'}, {'identifier': 'quality'}],
        )
    )
    values = {(m.identifier, p.value) for p in snapshot.points for m in snapshot.meta.metrics if m.id == p.metric}
    assert values == {('value', 10.5), ('quality', 3.0)}


def test_a_nameless_metric_is_read_from_its_label_column() -> None:
    """
    A metric without a ``name`` had its uuid as the v1 identifier and its label as the column.

    `DBDataset.deserialize_df` named the column ``Coalesce(name, label, uuid)``; the v1
    import once dropped every data point of such a metric by looking under the uuid.
    """
    snapshot = upgrade(
        v1_payload(
            [{'Year': 2020, 'sector': 'homes', 'Floor Area': 12.0}],
            [{'identifier': '461f-uuid', 'label': {'en': 'Floor Area'}, 'column': 'Floor Area'}],
        )
    )
    assert [p.value for p in snapshot.points] == [12.0]
