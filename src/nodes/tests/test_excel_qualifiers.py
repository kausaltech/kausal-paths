"""Qualifiers in exported rows must describe those rows, including ungraded values."""

import polars as pl
import pytest

from common import qualifiers
from common.polars import DataFrameMeta, to_ppdf
from nodes.excel_results import InstanceResultExcel
from nodes.tests.factories import InstanceFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_export_projects_reporting_and_missing_assessments_with_source_identity() -> None:
    context = InstanceFactory.create().context
    context.qualifiers = qualifiers.QualifierCatalog((
        *qualifiers.BUILTIN_QUALIFIERS.definitions,
        qualifiers.QualifierDefinition('assessment', qualifiers.Propagation.COVERED_SCORE),
    ))
    frame = to_ppdf(
        pl.DataFrame({'Year': [2020], 'Value': [0.0]}).with_columns(
            qualifiers.make(reported=pl.lit(value=True)).alias('Value__qual')
        ),
        meta=DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year']),
    )
    frame = qualifiers.with_selected_source(frame, 'node-uuid')
    exported = frame.select(InstanceResultExcel._qualifier_expressions(context, frame)).to_dicts()
    assert exported == [
        {
            'reported.any': True,
            'reported.all': True,
            'assessment.score': None,
            'assessment.coverage': None,
            'sources': 'node-uuid',
        }
    ]
