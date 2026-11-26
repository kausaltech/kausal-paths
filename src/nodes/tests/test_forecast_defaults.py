"""Where a dataset binding's forecast begins when nothing on the binding says so."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import polars as pl
import pytest

from common.polars import DataFrameMeta, to_ppdf
from datasets.runtime import DatasetWithFilters, DVCDataset
from nodes.defs.node_defs import InputDatasetDef
from nodes.defs.transform_def import SetForecastFromOp
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot
from nodes.transforms import _set_forecast_from
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _context(*, opt_in: bool, hold: bool = False):
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
        'features': {'forecast_after_maximum_historical_year': opt_in, 'hold_inputs_after_maximum_historical_year': hold},
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


def _forecast_ops(ds: DatasetWithFilters) -> list[SetForecastFromOp]:
    return [op for op in ds.transformations if isinstance(op, SetForecastFromOp)]


def test_holding_inputs_implies_the_forecast_boundary_and_marks_it() -> None:
    ds = DVCDataset.from_def(InputDatasetDef(id='mainz/actions'), _context(opt_in=False, hold=True))
    assert [(op.year, op.hold_last_value) for op in _forecast_ops(ds)] == [(2025, True)]


def test_a_binding_or_dataset_with_its_own_forecast_year_keeps_its_values() -> None:
    context = _context(opt_in=True, hold=True)
    ds = DVCDataset.from_def(InputDatasetDef(id='mainz/actions', forecast_from=2030), context)
    assert [(op.year, op.hold_last_value) for op in _forecast_ops(ds)] == [(2030, False)]

    kwargs = DatasetWithFilters.kwargs_from_def(InputDatasetDef(id='mainz/actions'))
    DatasetWithFilters.apply_forecast_defaults(kwargs, context, dataset_default=2027)
    ops = [op for op in kwargs['transformations'] if isinstance(op, SetForecastFromOp)]
    assert [(op.year, op.hold_last_value) for op in ops] == [(2027, False)]


def test_held_values_end_at_the_last_historical_year_except_where_a_series_starts_later() -> None:
    frame = to_ppdf(
        pl.DataFrame({
            'Year': [2023, 2024, 2025, 2026, 2026, 2027],
            'carrier': ['gas', 'gas', 'gas', 'gas', 'biogas', 'biogas'],
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }),
        meta=DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year', 'carrier']),
    )
    out = _set_forecast_from(frame, SetForecastFromOp(year=2025, hold_last_value=True))
    rows = sorted(out.select(['carrier', 'Year', 'Value', 'Forecast']).rows())
    # Gas ends at 2024. Biogas has nothing before 2025 and keeps its first year alone, as history.
    assert rows == [('biogas', 2026, 5.0, False), ('gas', 2023, 1.0, False), ('gas', 2024, 2.0, False)]
