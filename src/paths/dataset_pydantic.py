"""
Paths-side bridge for ``kausal_common.datasets.models.Dataset`` revision payloads.

The ``Dataset`` model lives in ``kausal_common`` and is shared with Kausal
Watch, but the Pydantic snapshot shape is Paths-specific. The snapshot
type lives in ``datasets.snapshot``; this module provides the entry point called from
``Dataset.serializable_data()`` under the ``IS_PATHS`` gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets.snapshot import DatasetSnapshot

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset


def dataset_serializable_data(obj: Dataset) -> dict[str, Any]:
    """Entry point called by ``Dataset.serializable_data`` under IS_PATHS."""
    return DatasetSnapshot.from_model(obj).model_dump(mode='json')
