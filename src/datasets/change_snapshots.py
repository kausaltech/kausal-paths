"""Change-log snapshots of dataset rows."""

from typing import TYPE_CHECKING, Any

from frameworks.evidence import evidence_snapshot

if TYPE_CHECKING:
    from kausal_common.datasets.models import DataPoint


def data_point_snapshot(dp: DataPoint) -> dict[str, Any]:
    """Lightweight snapshot for change tracking."""
    # Decimal → float: JSONField can't serialize Decimal natively and
    # DataPoint values don't need cents-grade precision.
    return {
        'uuid': str(dp.uuid),
        'dataset_uuid': str(dp.dataset.uuid),
        'date': dp.date.isoformat() if dp.date else None,
        'value': float(dp.value) if dp.value is not None else None,
        'metric_uuid': str(dp.metric.uuid) if dp.metric else None,
        'dimension_category_uuids': [str(cat.uuid) for cat in dp.dimension_categories.all()],
        'evidence': evidence_snapshot(dp),
    }
