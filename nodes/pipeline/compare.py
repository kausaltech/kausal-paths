from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from deepdiff import DeepDiff

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from nodes.datasets import JSONDataset

from .executor import execute_pipeline_ir
from .ir import compile_pipeline_ir_to_spec

if TYPE_CHECKING:
    import common.polars as ppl
    from nodes.node import Node
    from nodes.pipeline.compat import PipelineCompatibleNode


def _sort_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    primary_key = list(schema['primaryKey'])
    primary_key.sort()
    schema['primaryKey'] = primary_key

    fields = [dict(field) for field in schema['fields']]
    fields.sort(key=lambda field: field['name'])
    for field in fields:
        constraints = field.get('constraints')
        if constraints and 'enum' in constraints:
            constraints['enum'].sort()
    schema['fields'] = fields
    return schema


def _get_sorted_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(table['data'])
    sort_keys: list[str] = []
    if rows:
        sort_keys = sorted(col for col in rows[0].keys() if col != VALUE_COLUMN)
    rows.sort(key=lambda row: tuple(row.get(col) for col in sort_keys))
    return rows


class PipelineOutputComparison(BaseModel):
    node_id: str
    success: bool
    skipped: bool = False
    reason: str | None = None
    schema_diff: dict[str, Any] = Field(default_factory=dict)
    row_diff: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)


def compare_node_with_lowered_pipeline(
    node: PipelineCompatibleNode, original_df: ppl.PathsDataFrame, target_node: Node | None, metric: str | None
) -> PipelineOutputComparison:
    try:
        ir = node.lower_to_pipeline_ir()
    except NotImplementedError as exc:
        return PipelineOutputComparison(
            node_id=node.id,
            success=True,
            skipped=True,
            reason=str(exc),
        )

    compare_pipeline_compatibility = node.context.compare_pipeline_compatibility
    node.context.compare_pipeline_compatibility = False
    try:
        lowered_df = execute_pipeline_ir(node, ir)
    finally:
        node.context.compare_pipeline_compatibility = compare_pipeline_compatibility
    if metric is not None:
        meta = lowered_df.get_meta()
        cols = meta.primary_keys.copy()
        cols.append(metric)
        if FORECAST_COLUMN in lowered_df.columns:
            cols.append(FORECAST_COLUMN)
        lowered_df = lowered_df.select(cols)
        lowered_df = lowered_df.rename({metric: VALUE_COLUMN})
    if target_node is not None:
        lowered_df = node._get_output_for_target(lowered_df, target_node, skip_dim_test=False)

    legacy = JSONDataset.serialize_df(original_df)
    lowered = JSONDataset.serialize_df(lowered_df)

    schema_diff = DeepDiff(_sort_schema(legacy['schema']), _sort_schema(lowered['schema']))
    row_diff = DeepDiff(_get_sorted_rows(legacy), _get_sorted_rows(lowered), math_epsilon=1e-6)
    pipeline = compile_pipeline_ir_to_spec(ir).model_dump(mode='json', by_alias=True, exclude_none=True)

    return PipelineOutputComparison(
        node_id=node.id,
        success=not schema_diff and not row_diff,
        schema_diff=dict(schema_diff),
        row_diff=dict(row_diff),
        pipeline=pipeline,
    )
