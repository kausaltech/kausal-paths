"""Where a dataset binding's forecast begins when nothing on the binding says so."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from nodes.datasets import DatasetWithFilters, DVCDataset
from nodes.defs.node_defs import InputDatasetDef
from nodes.defs.transform_def import SetForecastFromOp
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot

pytestmark = pytest.mark.django_db


def _context(*, opt_in: bool):
    config: dict[str, Any] = {
        'id': 'forecast_test',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Forecast test',
        'owner': 'Owner',
        'target_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2024,
        'features': {'forecast_after_maximum_historical_year': opt_in},
        'nodes': [],
    }
    return InstanceLoader(snapshot=parse_instance_snapshot(config, instance_uuid=uuid4())).context


def _forecast_years(ds: DatasetWithFilters) -> list[int]:
    return [op.year for op in ds.transformations if isinstance(op, SetForecastFromOp)]


def test_opted_in_instances_start_the_forecast_after_the_last_historical_year() -> None:
    ds = DVCDataset.from_def(InputDatasetDef(id='mainz/actions'), _context(opt_in=True))
    assert ds.forecast_from == 2025
    assert _forecast_years(ds) == [2025]


def test_other_instances_are_unchanged() -> None:
    ds = DVCDataset.from_def(InputDatasetDef(id='mainz/actions'), _context(opt_in=False))
    assert ds.forecast_from is None
    assert _forecast_years(ds) == []


def test_the_binding_and_then_the_dataset_win_over_the_instance() -> None:
    context = _context(opt_in=True)
    ds = DVCDataset.from_def(InputDatasetDef(id='mainz/actions', forecast_from=2030), context)
    assert _forecast_years(ds) == [2030]

    kwargs = DatasetWithFilters.kwargs_from_def(InputDatasetDef(id='mainz/actions'))
    DatasetWithFilters.apply_forecast_defaults(kwargs, context, dataset_default=2022)
    assert kwargs['forecast_from'] == 2022
    assert [op.year for op in kwargs['transformations'] if isinstance(op, SetForecastFromOp)] == [2022]
