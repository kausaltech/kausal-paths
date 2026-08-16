from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import NAMESPACE_URL, uuid5

from django.utils.translation import gettext_lazy as _

import polars as pl

from kausal_common.i18n.pydantic import TranslatedString

from common import polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.constraints.port_roles import PortRoleInferenceResult
from nodes.constraints.rules import AnyShapeRule, MissingPortRoleError, ProductShapeRule, SameShapeRule
from nodes.defs.port_def import InputPortDeclaration, InputPortDef, OutputPortDeclaration
from nodes.units import Quantity
from params.param import BoolParameter, NumberParameter, StringParameter

from .constants import FORECAST_COLUMN, MIX_QUANTITY, VALUE_COLUMN, YEAR_COLUMN
from .exceptions import NodeError
from .node import InputPortMultiplicityHint, Node, NodeMetric, NodeStatus
from .operands import (
    Operand,
    declared_unit_of,
    empty_output_frame,
    impute_operands,
    multiply_operands,
    resolve_operands,
    sum_operands,
)
from .pipeline.compat import PipelineCompatibleNode

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from typing import Any
    from uuid import UUID

    import pandas as pd

    from paths.identifiers import NodePortIdentifier

    from nodes.datasets import Dataset
    from nodes.edges import Edge
    from nodes.instance_graph import NodeMeta
    from nodes.pipeline.ir import PipelinePortBinding
    from nodes.pipeline.ops import AnyOperationSpec
    from nodes.units import Unit
    from params.base import Parameter

EMISSION_UNIT = 'kg'


def _runtime_port_id(node_id: str, index: int) -> UUID:
    return uuid5(NAMESPACE_URL, f'kausal-paths:{node_id}:pipeline-input:{index}')


def additive_multiplicity_hint(node_class: type[Node], edge: Edge | None) -> InputPortMultiplicityHint:
    """
    Resolve the multiplicity hint for one edge into a class's additive multiport.

    Shared by the runtime hint and by ``instance_parser``, which has to predict the same
    port layout from class metadata alone, before any node exists.
    """
    if edge is None:
        return InputPortMultiplicityHint()
    declaration = node_class.additive_multiport_declaration(edge.tags)
    if declaration is None:
        return InputPortMultiplicityHint()
    return InputPortMultiplicityHint(
        multi=True,
        group=str(declaration.instance_identifier),
        role=str(declaration.role),
    )


class SimpleNode(Node):
    allowed_parameters: ClassVar[Sequence[Parameter[Any]]] = [
        # FIXME Get rid of many of these parameters and use operations instead.
        BoolParameter(
            local_id='fill_gaps_using_input_dataset',
            label=TranslatedString(en='Fill in gaps in computation using input dataset'),
            is_customizable=False,
        ),
        BoolParameter(
            local_id='replace_output_using_input_dataset',
            label=TranslatedString(en='Replace output using input dataset'),
            is_customizable=False,
        ),
        BoolParameter(
            local_id='drop_nulls',
            description=_('At the end of compute() do you want to drop nulls?'),
            is_customizable=False,
        ),
        NumberParameter(
            local_id='replace_nans',
            description=_('At the end of compute() replace nans with this value'),
            is_customizable=False,
        ),
        StringParameter(
            local_id='reference_category',
            description=_('Category to which all others are compared'),
            is_customizable=False,
        ),
        NumberParameter(
            local_id='reference_year',
            description=_('Year to which all others are compared'),
            is_customizable=False,
        ),
        StringParameter(
            local_id='share_dimension',
            description=_('Dimension over which values are converted to shares'),
            is_customizable=False,
        ),
        NumberParameter(  # FIXME Make sure that the treatment is systematic in all node classes.
            local_id='multiplier',
            description=_('Multiplier to implement after operation and before additions'),
            is_customizable=False,
        ),
        StringParameter(  # FIXME Is this the same functionality as variant?
            local_id='filter_categories',
            description=_('Categories to filter in format dimension:category,category2'),
            is_customizable=False,
        ),
    ]

    def replace_output_using_input_dataset_pl(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        # If we have also data from an input dataset, we only fill in the gaps from the
        # calculated data.
        df = df.drop_nulls()

        input_df = self.get_input_dataset_pl(required=False)
        if input_df is None:
            return df
        data_df = input_df

        data_latest_year: int = data_df[YEAR_COLUMN].max()  # type: ignore
        df_latest_year: int = df[YEAR_COLUMN].max()  # type: ignore
        df_meta = df.get_meta()
        data_meta = data_df.get_meta()
        if df_latest_year > data_latest_year:
            for col in data_meta.metric_cols:
                data_df = data_df.ensure_unit(col, df_meta.units[col])
            data_df = data_df.paths.join_over_index(df, how='outer')
            fills = [pl.col(col).fill_null(pl.col(col + '_right')) for col in data_meta.metric_cols]
            data_df = data_df.select([YEAR_COLUMN, *data_meta.dim_ids, FORECAST_COLUMN, *fills], units=df_meta.units)

        return data_df

    def replace_output_using_input_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.replace_output_using_input_dataset_pl(ppl.from_pandas(df)).to_pandas()

    def fill_gaps_using_input_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        ndf = ppl.from_pandas(df)
        out = self.fill_gaps_using_input_dataset_pl(ndf)
        return out.to_pandas()

    def fill_gaps_using_input_dataset_pl(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        data_df = self.get_input_dataset_pl(required=False)
        if data_df is None:
            return df

        meta = df.get_meta()
        df = df.paths.join_over_index(data_df, how='outer')
        for metric_col in meta.metric_cols:
            right = '%s_right' % metric_col  # FIXME Not clear that the right column has same metric name as left
            df = df.ensure_unit(right, meta.units[metric_col])
            df = df.with_columns([
                pl.col(metric_col).fill_null(pl.col(right)),
            ]).drop(right)
        return df

    def maybe_drop_nulls(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        if self.get_parameter_value('drop_nulls', required=False):
            df = df.drop_nulls()
        return df

    def replace_nans(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        rep = self.get_parameter_value('replace_nans', required=False)
        if rep is not None:
            df = df.with_columns(
                pl
                .when(pl.col(VALUE_COLUMN).is_nan() | pl.col(VALUE_COLUMN).is_infinite())
                .then(pl.lit(rep))
                .otherwise(pl.col(VALUE_COLUMN))
                .alias(VALUE_COLUMN)
            )
        return df

    def scale_by_reference_category(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        param = self.get_parameter_value_str('reference_category', required=False)
        if param:
            col, cat = param.split(':')
            reference = df.filter(pl.col(col).eq(cat)).drop(col)
            df = df.paths.join_over_index(reference)
            df = df.divide_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN).drop(VALUE_COLUMN + '_right')

        return df

    def scale_by_reference_year(self, df: ppl.PathsDataFrame, year: int | None = None) -> ppl.PathsDataFrame:
        if not year:
            year = self.get_typed_parameter_value('reference_year', int, required=False)
        if year:
            df = df.paths._scale_by_reference_year(df, year)
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

    def get_shares(self, df: ppl.PathsDataFrame, dim: str | None = None) -> ppl.PathsDataFrame:
        if not dim:
            dim = self.get_parameter_value_str('share_dimension', required=False)
        if dim:
            df = df.paths.calculate_shares(VALUE_COLUMN, VALUE_COLUMN, [dim])

        return df

    # See also sister function in ActionNode
    def apply_multiplier(self, df: ppl.PathsDataFrame, required: bool, units: bool) -> ppl.PathsDataFrame:
        multiplier = self.get_parameter_value('multiplier', required=required, units=units)
        if multiplier is not None:
            if isinstance(multiplier, Quantity):
                df = df.multiply_quantity(VALUE_COLUMN, multiplier)
            else:
                df = df.with_columns((pl.col(VALUE_COLUMN) * pl.lit(multiplier)).alias(VALUE_COLUMN))
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class AdditiveNode(SimpleNode, PipelineCompatibleNode):
    explanation = _("""This is an Additive Node. It performs a simple addition of inputs.
Missing values are assumed to be zero.

Input nodes tagged 'impute' are excluded from the addition; their values overlay the result
afterwards instead, replacing it wherever the tagged node has a value and leaving the rest untouched.""")
    export_additive_input_ports_as_multi: ClassVar[bool] = False
    additive_multi_input_excluded_tags: ClassVar[frozenset[str]] = frozenset({'non_additive'})
    additive_port = InputPortDeclaration(role='additive', multi=True, label=_('Additive inputs'))
    impute_port = InputPortDeclaration(role='impute', multi=True, min_count=0, default_count=0, label=_('Imputed values'))
    output_port = OutputPortDeclaration(role='output', identifier='default', label=_('Output'))
    input_port_declarations = (additive_port, impute_port)
    output_port_declarations = (output_port,)

    @classmethod
    def shape_rules(cls, meta: NodeMeta) -> tuple[AnyShapeRule, ...]:
        output = meta.require_output_port('output')
        inputs = meta.input_port_ids_for_roles('additive', 'impute')
        if not inputs:
            return ()
        return (SameShapeRule(inputs=inputs, output=output.id),)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        try:
            output_unit = meta.require_output_port('output').unit
        except MissingPortRoleError:
            output_unit = None
        for port in candidates:
            tags = {tag for binding in meta.bindings_for_port(port.id) for tag in binding.tags}
            if 'impute' in tags:
                result.classify(port, 'impute', "binding tag 'impute'")
            elif 'non_additive' in tags:
                result.refuse(port, "tag 'non_additive' excludes it from addition")
            elif port.unit is None or output_unit is None:
                result.refuse(port, 'cannot classify without both port and output units')
            elif port.unit.is_compatible_with(output_unit):
                result.classify(port, 'additive', f'unit {port.unit} being compatible with output {output_unit}')
            else:
                result.refuse(port, f'unit {port.unit} is incompatible with output {output_unit} on an additive node')
        return result

    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        BoolParameter(local_id='drop_nans', is_customizable=False),
        StringParameter(local_id='metric', is_customizable=False),
        BoolParameter(
            local_id='inventory_only',
            description=_('Node represents historical (inventory) values only'),
            is_customizable=False,
        ),
        BoolParameter(
            local_id='use_input_node_unit_when_adding',
            description=_('Use input node unit when doing add_nodes_pl()'),
            is_customizable=False,
        ),
    ]

    @classmethod
    def additive_multiport_declaration(cls, tags: Collection[str] = ()) -> InputPortDeclaration | None:
        # The base class always has one; a subclass only if it says so, because a subclass
        # usually consumes its inputs by tag rather than pooling them into a sum.
        if cls is not AdditiveNode and not cls.export_additive_input_ports_as_multi:
            return None
        if any(tag in cls.additive_multi_input_excluded_tags for tag in tags):
            return None
        return cls.additive_port

    def input_port_multiplicity_hint(
        self,
        *,
        edge: Edge | None = None,
        metric: NodeMetric | None = None,
        dataset: Dataset | None = None,
    ) -> InputPortMultiplicityHint:
        return additive_multiplicity_hint(type(self), edge)

    def lower_to_pipeline_ir(self):
        from nodes.pipeline import AddOperationSpec, IdentityOperationSpec, InputNodeBinding, PipelineNodeIR, PortInputRef

        unsupported_params = [
            'drop_nans',
            'metric',
            'inventory_only',
            'use_input_node_unit_when_adding',
            'fill_gaps_using_input_dataset',
            'replace_output_using_input_dataset',
            'drop_nulls',
            'replace_nans',
            'reference_category',
            'reference_year',
            'share_dimension',
            'multiplier',
        ]
        active_unsupported = [
            param_id for param_id in unsupported_params if self.get_parameter_value(param_id, required=False) not in (None, False)
        ]
        if self.input_dataset_instances or active_unsupported:
            raise NotImplementedError(
                'AdditiveNode pipeline lowering currently supports only pure input-node addition '
                + f'(datasets={bool(self.input_dataset_instances)}, unsupported_params={active_unsupported})'
            )

        if not self.input_nodes:
            raise NotImplementedError('AdditiveNode pipeline lowering currently requires at least one input node')

        spec = self.spec
        if spec.input_ports and len(spec.input_ports) == len(self.input_nodes):
            port_ids = [port.id for port in spec.input_ports]
        else:
            port_ids = [_runtime_port_id(self.id, idx) for idx, _ in enumerate(self.input_nodes)]

        port_bindings: dict[NodePortIdentifier, PipelinePortBinding] = {
            port_id: InputNodeBinding(node=input_node.id) for port_id, input_node in zip(port_ids, self.input_nodes, strict=True)
        }
        first_port = PortInputRef(port=port_ids[0])
        operations: list[AnyOperationSpec]
        if len(port_ids) == 1:
            operations = [IdentityOperationSpec(input=first_port, result_id='output')]
        else:
            operations = [
                AddOperationSpec(
                    input=first_port,
                    values=[PortInputRef(port=port_id) for port_id in port_ids[1:]],
                    result_id='output',
                ),
            ]

        return PipelineNodeIR(
            node_id=self.id,
            source_node_class=f'{type(self).__module__}.{type(self).__qualname__}',
            port_bindings=port_bindings,
            operations=operations,
            output_ref='output',
        )

    def add_nodes(self, ndf: pd.DataFrame | None, nodes: list[Node], metric: str | None = None) -> pd.DataFrame:
        if ndf is not None:
            df = ppl.from_pandas(ndf)
        else:
            df = None
        out = self.add_nodes_pl(df, nodes, metric)
        return out.to_pandas()

    def _process_input_dataset_df(self, df: ppl.PathsDataFrame, metric: str | None) -> ppl.PathsDataFrame:  # noqa: PLR0912
        if VALUE_COLUMN not in df.columns:
            if len(df.metric_cols) == 1:
                df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
            elif metric is not None:
                if metric in df.columns:
                    df = df.rename({metric: VALUE_COLUMN})
                    cols = [YEAR_COLUMN, *df.dim_ids, VALUE_COLUMN]
                    if FORECAST_COLUMN in df.columns:
                        cols.append(FORECAST_COLUMN)
                    df = df.select(cols)
                else:
                    raise NodeError(self, 'Metric is not found in metric columns')
            else:
                compatible_cols = [col for col, unit in df.get_meta().units.items() if self.is_compatible_unit(unit, self.unit)]
                if len(compatible_cols) == 1:
                    df = df.rename({compatible_cols[0]: VALUE_COLUMN})
                    cols = [YEAR_COLUMN, *df.dim_ids, VALUE_COLUMN]
                    if FORECAST_COLUMN in df.columns:
                        cols.append(FORECAST_COLUMN)
                    df = df.select(cols)
                else:
                    raise NodeError(self, 'Input dataset has multiple metric columns, but no Value column')
        elif VALUE_COLUMN not in df.metric_cols:
            raise NodeError(self, 'Value column is not a metric')

        df = self.apply_multiplier(df, required=False, units=True)
        df = df.ensure_unit(VALUE_COLUMN, self.single_metric_unit)

        if self.get_parameter_value('inventory_only', required=False):
            df = df.with_columns([pl.lit(value=False).alias(FORECAST_COLUMN)])
        else:
            df = extend_last_historical_value_pl(df, self.get_end_year())
        return df

    def _empty_output(self) -> ppl.PathsDataFrame:
        """
        Build an empty (zero-row) but schema-valid output for an INCOMPLETE node.

        Used when the node has no available inputs (e.g. not wired up yet). The frame
        is dimensionless — a transparent additive node with no inputs has no categorical
        dimensions. See ``docs/architecture/fault-tolerance.md``.
        """
        return empty_output_frame(self)

    def compute(self) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912
        idf = self.get_input_dataset_pl(required=False)
        metric = self.get_parameter_value_str('metric', required=False)
        assert self.unit is not None
        if idf is not None:
            idf = self._process_input_dataset_df(idf, metric)

        na_nodes = self.get_input_nodes(tag='non_additive')
        impute_nodes = self.get_input_nodes(tag='impute')
        input_nodes = [node for node in self.input_nodes if node not in na_nodes and node not in impute_nodes]

        if self.get_parameter_value('use_input_node_unit_when_adding', required=False) and self.input_nodes:
            unit = self.input_nodes[0].unit
        else:
            unit = self.unit

        tolerant = self.context.tolerate_node_failures
        skipped = 0
        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            if tolerant:
                df, skipped = self.add_nodes_tolerant(None, input_nodes, metric, unit=unit)
            else:
                df = self.add_nodes_pl(None, input_nodes, metric, unit=unit)
            if df is not None:
                df = self.fill_gaps_using_input_dataset_pl(df)
        elif tolerant:
            df, skipped = self.add_nodes_tolerant(idf, input_nodes, metric, unit=unit)
        else:
            df = self.add_nodes_pl(idf, input_nodes, metric, unit=unit)

        if df is None:
            # No input dataset and every input node unavailable: the node isn't wired up yet.
            self.mark_status(NodeStatus.INCOMPLETE)
            return self._empty_output()
        if skipped:
            self.mark_status(NodeStatus.DEGRADED)

        df = self.maybe_drop_nulls(df)  # FIXME Check where this should be done.
        if self.get_parameter_value('drop_nans', required=False):  # FIXME: Implement this in the same way as drop_nulls
            df = df.filter(~pl.col(VALUE_COLUMN).is_nan())
        df = self.scale_by_reference_category(df)
        df = self.scale_by_reference_year(df)
        df = self.get_shares(df)

        if impute_nodes:
            df = self.impute_nodes_pl(df, impute_nodes)

        return df


class AdditiveNode2(PipelineCompatibleNode):
    """
    Sum every input, whether it arrived as a node or as a dataset.

    The rebuilt ``AdditiveNode``. Three things differ from the original, all deliberate:

    * **A dataset is just another input.** Any number of them, combined with node inputs on
      equal terms, instead of the old limit of one dataset processed down a separate path.
    * **No implicit extension.** The original carried a dataset's last value to the model
      end year but did nothing of the sort for a node input. Whether a series extends is now
      a property of the binding, not of how the value happens to reach the node.
    * **A ``non_additive`` input is an error.** The original collected them and then silently
      dropped them, so a factor wired into an additive node vanished without a word.

    See ``docs/plans/additive-multiplicative-modernization.md``.
    """

    explanation = _("""This is an Additive Node. It adds up all of its inputs, whether they are
nodes or datasets. Inputs must have the same dimensions and compatible units; a missing value
counts as zero.

Inputs tagged 'impute' are excluded from the addition; their values overlay the result
afterwards instead, replacing it wherever the tagged input has a value.""")

    allowed_parameters: ClassVar[Sequence[Parameter[Any]]] = [
        BoolParameter(
            local_id='inventory_only',
            description=_('Node represents historical (inventory) values only'),
            is_customizable=False,
        ),
        StringParameter(
            local_id='metric',
            description=_('Which column of a multi-metric input dataset carries the values'),
            is_customizable=False,
        ),
    ]

    interpolates_input_datasets_by_default: ClassVar[bool] = True
    export_additive_input_ports_as_multi: ClassVar[bool] = False
    additive_multi_input_excluded_tags: ClassVar[frozenset[str]] = frozenset({'non_additive'})
    additive_port = InputPortDeclaration(role='additive', multi=True, label=_('Additive inputs'))
    impute_port = InputPortDeclaration(role='impute', multi=True, min_count=0, default_count=0, label=_('Imputed values'))
    output_port = OutputPortDeclaration(role='output', identifier='default', label=_('Output'))
    input_port_declarations = (additive_port, impute_port)
    output_port_declarations = (output_port,)

    @classmethod
    def shape_rules(cls, meta: NodeMeta) -> tuple[AnyShapeRule, ...]:
        output = meta.require_output_port('output')
        inputs = meta.input_port_ids_for_roles('additive', 'impute')
        if not inputs:
            return ()
        return (SameShapeRule(inputs=inputs, output=output.id),)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        try:
            output_unit = meta.require_output_port('output').unit
        except MissingPortRoleError:
            output_unit = None
        for port in candidates:
            tags = {tag for binding in meta.bindings_for_port(port.id) for tag in binding.tags}
            if 'impute' in tags:
                result.classify(port, 'impute', "binding tag 'impute'")
            elif 'non_additive' in tags:
                result.refuse(port, "tag 'non_additive' has no meaning on an additive node")
            elif port.unit is None or output_unit is None:
                result.refuse(port, 'cannot classify without both port and output units')
            elif port.unit.is_compatible_with(output_unit):
                result.classify(port, 'additive', f'unit {port.unit} being compatible with output {output_unit}')
            else:
                result.refuse(port, f'unit {port.unit} is incompatible with output {output_unit} on an additive node')
        return result

    @classmethod
    def additive_multiport_declaration(cls, tags: Collection[str] = ()) -> InputPortDeclaration | None:
        if cls is not AdditiveNode2 and not cls.export_additive_input_ports_as_multi:
            return None
        if any(tag in cls.additive_multi_input_excluded_tags for tag in tags):
            return None
        return cls.additive_port

    def input_port_multiplicity_hint(
        self,
        *,
        edge: Edge | None = None,
        metric: NodeMetric | None = None,
        dataset: Dataset | None = None,
    ) -> InputPortMultiplicityHint:
        return additive_multiplicity_hint(type(self), edge)

    def lower_to_pipeline_ir(self):
        from nodes.pipeline import AddOperationSpec, IdentityOperationSpec, InputNodeBinding, PipelineNodeIR, PortInputRef

        if self.input_dataset_instances:
            raise NotImplementedError('AdditiveNode2 pipeline lowering does not yet support input datasets')
        if not self.input_nodes:
            raise NotImplementedError('AdditiveNode2 pipeline lowering requires at least one input node')

        spec = self.spec
        if spec.input_ports and len(spec.input_ports) == len(self.input_nodes):
            port_ids = [port.id for port in spec.input_ports]
        else:
            port_ids = [_runtime_port_id(self.id, idx) for idx, _ in enumerate(self.input_nodes)]

        port_bindings: dict[NodePortIdentifier, PipelinePortBinding] = {
            port_id: InputNodeBinding(node=input_node.id) for port_id, input_node in zip(port_ids, self.input_nodes, strict=True)
        }
        first_port = PortInputRef(port=port_ids[0])
        operations: list[AnyOperationSpec]
        if len(port_ids) == 1:
            operations = [IdentityOperationSpec(input=first_port, result_id='output')]
        else:
            operations = [
                AddOperationSpec(
                    input=first_port,
                    values=[PortInputRef(port=port_id) for port_id in port_ids[1:]],
                    result_id='output',
                ),
            ]

        return PipelineNodeIR(
            node_id=self.id,
            source_node_class=f'{type(self).__module__}.{type(self).__qualname__}',
            port_bindings=port_bindings,
            operations=operations,
            output_ref='output',
        )

    def compute(self) -> ppl.PathsDataFrame:
        assert self.unit is not None
        operands = resolve_operands(self, metric=self.get_parameter_value_str('metric', required=False))

        if operands.factors:
            raise NodeError(
                self,
                'An additive node cannot multiply. These inputs are factors, by tag or by unit: %s. '
                'Use a MultiplicativeNode2, or fix the unit.' % ', '.join(str(op) for op in operands.factors),
            )
        if operands.claimed_elsewhere:
            raise NodeError(
                self,
                'These inputs are tagged for an operation an additive node does not have: %s'
                % ', '.join(operands.claimed_elsewhere),
            )

        if not operands.additive:
            # Nothing usable arrived: not wired up yet, or every upstream is unavailable.
            self.mark_status(NodeStatus.INCOMPLETE)
            return empty_output_frame(self)
        if operands.unavailable:
            self.mark_status(NodeStatus.DEGRADED)

        df = sum_operands(self, operands.additive, self.unit)
        if self.get_parameter_value('inventory_only', required=False):
            df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        if operands.impute:
            df = impute_operands(self, df, operands.impute)
        return df


class DataAvailabilityNode(AdditiveNode):
    """
    Report where the input dataset has a value, as 1.0 (value present) and 0.0 (no value).

    The test is made on the dataset as it arrives from its source, before interpolation or
    extension can fill in the missing years, so the output describes what the data actually
    covers rather than what the pipeline is able to fabricate. Interpolation configured on
    the bindings (``interpolate: true`` or ``input_dataset_processors: [LinearInterpolation]``)
    is therefore switched off for this node's datasets.

    The output covers the whole model year range (``minimum_historical_year`` ..
    ``model_end_year``) for every dimension category combination that occurs in the dataset;
    cells outside the data's own span are 0.0. The combinations are the ones the data uses,
    not the cross product of each dimension's categories, so that a dataset whose categories
    are ragged (a category of one dimension going together with only some categories of
    another) is not reported as permanently incomplete. Each metric column of the dataset becomes a
    0/1 column of its own. When the dataset has a single metric column, it is renamed to the
    node's own metric column, so the usual ``unit: dimensionless`` node definition is enough;
    a multi-metric dataset needs matching ``output_metrics`` on the node.

    Taking the combinations from the data means that a combination nobody supplied is not
    reported as missing — it is simply absent from the output, and a downstream flag computed
    over it silently loses a term. Where the set of combinations is prescribed rather than
    discovered, declare it in a second dataset tagged ``template``: its dimension columns list
    the combinations that must exist, and they become the combinations reported on. A required
    combination the data never mentions then reads 0.0 for every year instead of vanishing.
    Keeping that dataset in DVC is the point of it — a user who edits the data being checked
    cannot edit the requirement. The template's own values and years are ignored; one year's
    worth of rows is enough, because a requirement of this kind is structural rather than
    temporal, and it is checked in every year of the model range.

    With ``flag_unexpected``, a combination that carries a value the template does not ask for
    reads -1.0 instead of 1.0. It is off by default because these outputs are meant to be
    combined arithmetically downstream (``prod_dim`` over a conformity flag, say), and a
    negative term would corrupt that rather than register as a complaint.
    """

    explanation = _("""This is a Data Availability Node. Instead of using the values of its input dataset, it
reports whether a value exists in each cell: 1.0 where the dataset has a value and 0.0 where it does not.
The check is made on the original data, before interpolation or extension fill in the missing years.
The output covers the whole model period; the years and categories that the dataset does not reach are 0.0.
When a template dataset declares which category combinations have to exist, those are the ones reported on,
so that a combination missing from the data altogether is reported as missing rather than passed over.""")

    allowed_parameters: ClassVar[list[Parameter[Any]]] = [
        *AdditiveNode.allowed_parameters,
        BoolParameter(
            local_id='flag_unexpected',
            label=_('Report values that the template does not require as -1'),
        ),
    ]

    TEMPLATE_TAG = 'template'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The node reports what the source data covers, so gap-filling must not run before
        # the check. The dataset instances belong to this node alone, so this affects nobody
        # else (and the dataset cache key follows `interpolate`, so it stays separate too).
        for ds in self.input_dataset_instances:
            ds.interpolate = False

    def compute(self) -> ppl.PathsDataFrame:
        from nodes.datasets import FixedDataset

        if self.input_nodes:
            raise NodeError(
                self,
                'DataAvailabilityNode only inspects its input dataset; it has %d input nodes. Combine '
                'availability flags in a downstream node instead (e.g. with the min/max edge tags).' % len(self.input_nodes),
            )
        for ds in self.input_dataset_instances:
            if isinstance(ds, FixedDataset) and ds.use_interpolation:
                raise NodeError(
                    self,
                    "Dataset '%s' is interpolated already when it is built, so the gaps of the original data "
                    'cannot be seen any more. Remove the LinearInterpolation dataset processor.' % ds.id,
                )
        dfs = self.get_input_datasets_pl(exclude_tags=[self.TEMPLATE_TAG])
        if len(dfs) != 1:
            raise NodeError(
                self,
                'Expecting one dataset to check the availability of, besides any tagged %r; got %d.'
                % (self.TEMPLATE_TAG, len(dfs)),
            )
        return self.check_availability(dfs[0])

    def _get_template(self) -> ppl.PathsDataFrame | None:
        templates = self.get_input_datasets_pl(tag=self.TEMPLATE_TAG)
        if not templates:
            return None
        if len(templates) > 1:
            raise NodeError(self, 'Expecting at most one dataset tagged %r; got %d.' % (self.TEMPLATE_TAG, len(templates)))
        return templates[0].paths.cast_index_to_str()

    def _get_required_combinations(self, dim_ids: list[str]) -> pl.DataFrame | None:
        """Return one row per combination the template requires, or None when there is no template."""
        template = self._get_template()
        if template is None:
            return None
        if sorted(template.dim_ids) != sorted(dim_ids):
            raise NodeError(
                self,
                'The template requires combinations of (%s), but the data has dimensions (%s). '
                'They have to be the same dimensions for the requirement to mean anything.'
                % (', '.join(sorted(template.dim_ids)) or '-', ', '.join(sorted(dim_ids)) or '-'),
            )
        if not dim_ids:
            raise NodeError(
                self,
                'A template declares which dimension category combinations must exist, but the data has no dimensions.',
            )
        # The template's values and years carry no meaning; the combinations are the whole of it.
        combinations = pl.DataFrame({dim_id: template[dim_id] for dim_id in dim_ids})
        return combinations.unique(subset=dim_ids)

    def _all_required_missing(self, template: ppl.PathsDataFrame, metric_cols: list[str]) -> ppl.PathsDataFrame:
        """
        Report every required combination as missing, for a dataset that has no rows at all.

        A dataset nobody has started filling in cannot say what it should have contained: rows
        whose every metric is null are dropped on the way here, and with the last row go the
        dimension columns. That is exactly the case the template exists for -- "nobody has
        looked" has to read as missing rather than as an error -- so the combinations come from
        the template instead, and every one of them is 0.0 in every year.
        """
        dim_ids = sorted(template.dim_ids)
        combinations = pl.DataFrame({dim_id: template[dim_id] for dim_id in dim_ids}).unique(subset=dim_ids)
        years = pl.DataFrame({
            YEAR_COLUMN: range(self.context.instance.minimum_historical_year, self.get_end_year() + 1),
        })
        grid = years.join(combinations, how='cross')
        grid = grid.with_columns(
            [pl.lit(0.0).alias(col) for col in metric_cols]
            # Dimension columns are Categorical everywhere else, and a Utf8 column here would
            # move the failure downstream to the first join against a normal node output.
            + [pl.col(dim_id).cast(pl.Categorical) for dim_id in dim_ids],
        )
        meta = ppl.DataFrameMeta(
            units=dict.fromkeys(metric_cols, self.context.unit_registry.parse_units('dimensionless')),
            primary_keys=[YEAR_COLUMN, *dim_ids],
        )
        return ppl.to_ppdf(grid, meta=meta)

    def _report_on_required_combinations(self, out: ppl.PathsDataFrame, required: pl.DataFrame) -> ppl.PathsDataFrame:
        """Give every required combination a row in every year, whether or not the data mentions it."""
        required_col = '_Required'
        meta = out.get_meta()
        dim_ids = out.dim_ids
        flag_cols = out.metric_cols
        # Join on plain strings, then put the dimension columns back as they were. Category
        # columns from two different frames are not joinable, and leaving the output as Utf8
        # where it used to be Categorical would move the same failure downstream instead.
        dim_dtypes = {dim_id: out.schema[dim_id] for dim_id in dim_ids}
        out = out.with_columns([pl.col(dim_id).cast(pl.Utf8) for dim_id in dim_ids])

        years = pl.DataFrame({YEAR_COLUMN: out[YEAR_COLUMN].unique()})
        grid = years.join(required.with_columns(pl.lit(value=True).alias(required_col)), how='cross')
        # The union: a required combination the data never mentions gets its zeros, and a
        # combination the data has that nothing requires is still reported rather than dropped.
        joined = out.join(grid, on=[YEAR_COLUMN, *dim_ids], how='full', coalesce=True)
        joined = joined.with_columns([pl.col(col).fill_null(0.0) for col in flag_cols])

        if self.get_typed_parameter_value('flag_unexpected', bool, required=False):
            unexpected = ~pl.col(required_col).fill_null(value=False)
            joined = joined.with_columns([
                pl.when(unexpected & (pl.col(col) > 0)).then(pl.lit(-1.0)).otherwise(pl.col(col)).alias(col) for col in flag_cols
            ])

        joined = joined.drop(required_col).with_columns([pl.col(dim_id).cast(dtype) for dim_id, dtype in dim_dtypes.items()])
        return ppl.to_ppdf(joined, meta=meta)

    def check_availability(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Convert the values of the raw dataset into 1.0 (value exists) and 0.0 (no value)."""
        metric_cols = df.metric_cols
        if not metric_cols:
            raise NodeError(self, 'The input dataset has no metric columns whose availability could be tested.')
        if len(metric_cols) == 1:
            out_cols = [self.get_default_output_metric().column_id]
        else:
            out_cols = list(metric_cols)
        declared = {metric.column_id: metric.unit for metric in self.output_metrics.values()}
        units = {col: declared.get(col, self.context.unit_registry.parse_units('dimensionless')) for col in out_cols}
        with_dimension = {col: str(unit) for col, unit in units.items() if not unit.dimensionless}
        if with_dimension:
            raise NodeError(
                self,
                'The output columns tell whether a value exists, so their units must be dimensionless: %s.'
                % ', '.join("'%s' is '%s'" % item for item in with_dimension.items()),
            )

        template = self._get_template()
        if df.is_empty() or not df.dim_ids:
            if template is not None and template.dim_ids:
                return self._finish_availability(
                    self._all_required_missing(template, metric_cols),
                    metric_cols,
                    out_cols,
                    units,
                )
            if df.is_empty():
                raise NodeError(
                    self,
                    'The dataset has no rows, so there is nothing to report the availability of, and no '
                    'dataset tagged %r declares what it should have contained. Add a template.' % self.TEMPLATE_TAG,
                )

        # Missing data is mostly missing *rows*, not null cells, so the years and the
        # dimension category combinations have to be materialised before they can be
        # reported as zeros. Projecting wide gives one column per combination that the
        # dataset actually uses; joining the model timeline onto it then turns every
        # absent year into nulls, which the presence test below reads as 'no value'.
        # (Crossing the categories of each dimension separately instead would invent
        # cells that the dataset never means to have -- the BISKO district heating data
        # uses different energy carriers for fuel input than for heat output -- and those
        # would look like missing data forever.)
        wide = df.drop(FORECAST_COLUMN) if FORECAST_COLUMN in df.columns else df
        wide = wide.paths.to_wide()
        years = pl.DataFrame({YEAR_COLUMN: range(self.context.instance.minimum_historical_year, self.get_end_year() + 1)})
        years_pdf = ppl.to_ppdf(
            years.with_columns(pl.col(YEAR_COLUMN).cast(df.schema[YEAR_COLUMN])),
            ppl.DataFrameMeta(units={}, primary_keys=[YEAR_COLUMN]),
        )
        wide = years_pdf.paths.join_over_index(wide)

        # A null cell has no value; for float columns NaN counts as no value, too.
        # (`is_not_nan()` is null for null cells, but the `is_not_null()` term makes the
        # conjunction false there anyway.)
        flags: list[pl.Expr] = []
        for col in wide.metric_cols:
            has_value = pl.col(col).is_not_null()
            if wide.schema[col].is_float():
                has_value = has_value & pl.col(col).is_not_nan()
            flags.append(has_value.cast(pl.Float64).alias(col))
        out = wide.with_columns(flags).paths.to_narrow()

        required = self._get_required_combinations(out.dim_ids)
        if required is not None:
            out = self._report_on_required_combinations(out, required)

        return self._finish_availability(out, metric_cols, out_cols, units)

    def _finish_availability(
        self,
        out: ppl.PathsDataFrame,
        metric_cols: list[str],
        out_cols: list[str],
        units: dict[str, Unit],
    ) -> ppl.PathsDataFrame:
        """Mark the forecast years and give the flag columns the node's own metric names."""
        max_hist_year = self.context.instance.maximum_historical_year
        is_forecast = pl.lit(value=False) if max_hist_year is None else pl.col(YEAR_COLUMN) > max_hist_year
        out = out.with_columns(is_forecast.alias(FORECAST_COLUMN)).sort(out.primary_keys)

        for col_id, out_col in zip(metric_cols, out_cols, strict=True):
            if out_col != col_id:
                out = out.rename({col_id: out_col})
            out = out.set_unit(out_col, 'dimensionless', force=True)
            out = out.ensure_unit(out_col, units[out_col])
        return out


class SubtractiveNode(Node):  # FIXME Remove, when you clean Longmont.
    explanation = _(
        'This is a Subtractive Node. It takes the first input node and subtracts all other input nodes from it.',
    )  # FIXME Is this needed? Edge process arithmetic_inverse could be used instead.
    allowed_parameters = [
        BoolParameter(
            local_id='only_historical',
            description=_('Perform subtraction on only historical data'),
            is_customizable=False,
        ),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        nodes = list(self.input_nodes)
        mults = [1.0 if i == 0 else -1.0 for i, _ in enumerate(nodes)]
        df = self.add_nodes_pl(None, nodes, node_multipliers=mults)
        only_historical = self.get_parameter_value('only_historical', required=False)
        if only_historical:
            df = df.filter(~pl.col(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class SectorEmissions(AdditiveNode):
    explanation = _('This is a Sector Emissions Node. It is like Additive Node but for subsector emissions')
    export_additive_input_ports_as_multi = True
    # FIXME Is this needed?
    quantity = 'emissions'

    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(
            local_id='category', description=_('Category id for the emission sector dimension'), is_customizable=False
        ),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        val = self.get_parameter_value('category', required=False)
        if val is not None:
            df = self.get_input_dataset_pl()
            df_dims = df.dim_ids
            for dim_id in self.input_dimensions.keys():
                if dim_id not in df_dims:
                    raise NodeError(self, "Dataset doesn't have dimension %s" % dim_id)
                df_dims.remove(dim_id)
            if len(df_dims) != 1:
                raise NodeError(self, 'Emission sector dimension missing')
            sector_dim = df_dims[0]
            df = df.filter(pl.col(sector_dim).eq(val))
            if not len(df):
                raise NodeError(self, 'Emission sector %s not found in input data' % val)
            df = df.drop(sector_dim)
            m = self.get_default_output_metric()
            if len(df.metric_cols) != 1:
                raise NodeError(self, 'Input dataset has more than 1 metric')
            df = df.rename({df.metric_cols[0]: m.column_id})
            df = extend_last_historical_value_pl(df, self.get_end_year())
            df = df.drop_nulls()
            return super().add_nodes_pl(df, self.input_nodes)

        return super().compute()


class MultiplicativeNode(SimpleNode, PipelineCompatibleNode):
    explanation = _("""This is a Multiplicative Node. It multiplies nodes together with potentially adding other input nodes.

    Multiplication and addition is determined based on the input node units.

    Input nodes tagged 'impute' take no part in the multiplication or addition; their values
    overlay the result afterwards, replacing it wherever the tagged node has a value.
    """)

    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        BoolParameter(
            local_id='only_historical',
            description=_('Process only historical rows'),
            is_customizable=False,
        ),
        BoolParameter(
            local_id='extend_rows',
            description=_('Extend last row to future years'),
            is_customizable=False,
        ),
    ]
    operation_label = 'multiplication'
    factors_port = InputPortDeclaration(role='factors', repeatable=True, min_count=1, default_count=2, label=_('Factor'))
    additive_port = InputPortDeclaration(role='additive', multi=True, min_count=0, default_count=1, label=_('Additive inputs'))
    impute_port = InputPortDeclaration(role='impute', multi=True, min_count=0, default_count=0, label=_('Imputed values'))
    output_port = OutputPortDeclaration(role='output', identifier='default', label=_('Output'))
    input_port_declarations = (factors_port, additive_port, impute_port)
    output_port_declarations = (output_port,)

    @classmethod
    def shape_rules(cls, meta: NodeMeta) -> tuple[AnyShapeRule, ...]:
        output = meta.require_output_port('output')
        rules: list[AnyShapeRule] = []
        factors = meta.input_port_ids_for_roles('factors')
        if factors:
            rules.append(ProductShapeRule(inputs=factors, output=output.id))
        same_shaped = meta.input_port_ids_for_roles('additive', 'impute')
        if same_shaped:
            rules.append(SameShapeRule(inputs=same_shaped, output=output.id))
        return tuple(rules)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        from nodes.defs.binding_def import DatasetBindingDef

        result = PortRoleInferenceResult()
        try:
            output_unit = meta.require_output_port('output').unit
        except MissingPortRoleError:
            output_unit = None
        for port in candidates:
            bindings = meta.bindings_for_port(port.id)
            tags = {tag for binding in bindings for tag in binding.tags}
            if 'impute' in tags:
                result.classify(port, 'impute', "binding tag 'impute'")
            elif 'non_additive' in tags:
                result.classify(port, 'factors', "binding tag 'non_additive'")
            elif any(isinstance(binding, DatasetBindingDef) for binding in bindings):
                result.refuse(port, 'dataset-bound port on a multiplicative node has no explicit role')
            elif port.unit is None or output_unit is None:
                result.refuse(port, 'cannot classify without both port and output units')
            elif port.unit.is_compatible_with(output_unit):
                result.classify(port, 'additive', f'unit {port.unit} being compatible with output {output_unit}')
            else:
                result.classify(port, 'factors', f'unit {port.unit} being incompatible with output {output_unit}')
        return result

    def lower_to_pipeline_ir(self):
        from nodes.pipeline import InputNodeBinding, MultiplyOperationSpec, PipelineNodeIR, PortInputRef

        unsupported_params = [
            'only_historical',
            'extend_rows',
            'fill_gaps_using_input_dataset',
            'replace_output_using_input_dataset',
            'drop_nulls',
            'replace_nans',
            'reference_category',
            'reference_year',
            'share_dimension',
            'multiplier',
        ]
        active_unsupported = [
            param_id for param_id in unsupported_params if self.get_parameter_value(param_id, required=False) not in (None, False)
        ]
        if self.input_dataset_instances or active_unsupported:
            raise NotImplementedError(
                'MultiplicativeNode pipeline lowering currently supports only pure input-node multiplication '
                + f'(datasets={bool(self.input_dataset_instances)}, unsupported_params={active_unsupported})'
            )

        assert self.unit is not None
        additive_nodes: list[Node] = []
        operation_nodes: list[Node] = []
        non_additive_nodes = self.get_input_nodes(tag='non_additive')
        for node in self.input_nodes:
            if node.unit is None:
                raise NotImplementedError(f'Input node {node.id} does not have a unit')
            if node in non_additive_nodes:
                operation_nodes.append(node)
            elif self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        if additive_nodes:
            raise NotImplementedError(
                'MultiplicativeNode pipeline lowering does not yet support additive side inputs '
                + f'({[node.id for node in additive_nodes]})'
            )
        if len(operation_nodes) < 2:
            raise NotImplementedError(
                'MultiplicativeNode pipeline lowering currently requires at least two multiplicative inputs '
                + f'({[node.id for node in operation_nodes]})'
            )

        spec = self.spec
        if spec.input_ports and len(spec.input_ports) == len(self.input_nodes):
            port_ids = [port.id for port in spec.input_ports]
        else:
            port_ids = [_runtime_port_id(self.id, idx) for idx, _ in enumerate(self.input_nodes)]

        port_bindings: dict[NodePortIdentifier, PipelinePortBinding] = {
            port_id: InputNodeBinding(node=input_node.id) for port_id, input_node in zip(port_ids, self.input_nodes, strict=True)
        }
        operations: list[AnyOperationSpec] = [
            MultiplyOperationSpec(
                input=PortInputRef(port=port_ids[0]),
                values=[PortInputRef(port=port_id) for port_id in port_ids[1:]],
                result_id='output',
            ),
        ]

        return PipelineNodeIR(
            node_id=self.id,
            source_node_class=f'{type(self).__module__}.{type(self).__qualname__}',
            port_bindings=port_bindings,
            operations=operations,
            output_ref='output',
        )

    def operate_pairwise(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = df.multiply_cols(['_Left', '_Right'], '_Left').drop('_Right')
        return df

    def perform_operation(self, nodes: Sequence[Node | None], outputs: list[ppl.PathsDataFrame]) -> ppl.PathsDataFrame:
        for n in nodes:
            if n is None:
                continue
            assert n.unit is not None
        assert self.unit is not None

        df = None
        for n, ndf in zip(nodes, outputs, strict=False):
            if df is None:
                # First output in the list
                df = ndf
                if n is not None:
                    m = n.get_default_output_metric()
                    col = m.column_id
                else:
                    assert len(df.metric_cols) == 1
                    col = df.metric_cols[0]
                df = df.rename({col: '_Left'})
                continue

            if n is not None:
                m = n.get_default_output_metric()
                col = m.column_id
            else:
                assert len(ndf.metric_cols) == 1
                col = df.metric_cols[0]

            ndf_new = ndf.rename({col: '_Right'})
            df = df.paths.join_over_index(ndf_new, how='left', index_from='union')
            df = self.operate_pairwise(df)

        assert df is not None
        df = df.rename({'_Left': VALUE_COLUMN})
        df = df.drop_nulls(VALUE_COLUMN)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

    def _compute(self, input_df: ppl.PathsDataFrame | None = None) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912, PLR0915
        additive_nodes: list[Node] = []
        operation_nodes: list[Node] = []
        assert self.unit is not None
        non_additive_nodes = self.get_input_nodes(tag='non_additive')
        impute_nodes = self.get_input_nodes(tag='impute')
        for node in self.input_nodes:
            if node in impute_nodes:
                continue
            if node.unit is None:
                raise NodeError(self, 'Input node %s does not have a unit' % str(node))
            if node in non_additive_nodes:
                operation_nodes.append(node)
            elif self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        if len(operation_nodes) < 2 and input_df is None:
            raise NodeError(
                self,
                'Must receive at least two inputs to operate %s on. Now received %s.'
                % (self.operation_label, [node.id for node in operation_nodes]),
            )

        outputs: list[ppl.PathsDataFrame] = []
        for idx, n in enumerate(operation_nodes):
            ndf = n.get_output_pl(target_node=self)
            if self.debug:
                print('%s: %s input from node %d (%s):' % (self.operation_label, self.id, idx, str(n)))
                print(ndf)
            outputs.append(ndf)

        if outputs:
            df = self.perform_operation(operation_nodes, outputs)
            if input_df is not None:
                input_df = input_df.rename({VALUE_COLUMN: '_InputSum'})
                assert input_df.dim_ids == df.dim_ids
                df = df.paths.join_over_index(input_df)
                df = df.ensure_unit('_InputSum', df.get_unit(VALUE_COLUMN))
                df = df.with_columns((pl.col(VALUE_COLUMN) + pl.col('_InputSum')).alias(VALUE_COLUMN)).drop('_InputSum')
        else:
            assert input_df is not None
            df = input_df

        if self.get_parameter_value('only_historical', required=False):
            outputs = [df.filter(~pl.col(FORECAST_COLUMN)) for df in outputs]

        if self.get_parameter_value('extend_rows', required=False):
            df = extend_last_historical_value_pl(df, self.get_end_year())

        df = self.add_nodes_pl(df, additive_nodes)
        fill_gaps = self.get_parameter_value('fill_gaps_using_input_dataset', required=False)
        if fill_gaps:
            df = self.fill_gaps_using_input_dataset_pl(df)
        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset_pl(df)
        df = self.replace_nans(df)
        if impute_nodes:
            df = self.impute_nodes_pl(df, impute_nodes)
        if self.debug:
            print('%s: Output:' % str(self))
            self.print(df)

        return df

    def compute(self) -> ppl.PathsDataFrame:
        return self._compute()


class MultiplicativeNode2(PipelineCompatibleNode):
    """
    Multiply the factors, then add whatever is additive to the product.

    The rebuilt ``MultiplicativeNode``. Differences from the original, all deliberate:

    * **Datasets can be factors.** The original loaded an input dataset and then never
      looked at it, so a dataset bound here contributed nothing, silently.
    * **A null factor value propagates.** The original dropped the whole row, so a year with
      one unknown factor disappeared from the output rather than being reported as unknown.
    * **The unit test reads what an input actually delivers**, not what it declares. The two
      disagree whenever a node's output unit differs from its declaration.

    See ``docs/plans/additive-multiplicative-modernization.md``.
    """

    explanation = _("""This is a Multiplicative Node. It multiplies its factors together and adds
any additive inputs to the product. Inputs may be nodes or datasets. Whether an input is a factor
or an addend is decided by its tag ('non_additive' or 'additive'), or failing that by whether its
unit is compatible with this node's own unit.

The product spans the union of the factors' dimensions, and a row missing from any factor is
missing from the result. Additive inputs must match the product's dimensions.

Inputs tagged 'impute' take no part in either operation; their values overlay the result
afterwards, replacing it wherever the tagged input has a value.""")

    allowed_parameters: ClassVar[Sequence[Parameter[Any]]] = [
        StringParameter(
            local_id='metric',
            description=_('Which column of a multi-metric input dataset carries the values'),
            is_customizable=False,
        ),
    ]

    interpolates_input_datasets_by_default: ClassVar[bool] = True
    operation_label = 'multiplication'
    factors_port = InputPortDeclaration(role='factors', repeatable=True, min_count=1, default_count=2, label=_('Factor'))
    additive_port = InputPortDeclaration(role='additive', multi=True, min_count=0, default_count=1, label=_('Additive inputs'))
    impute_port = InputPortDeclaration(role='impute', multi=True, min_count=0, default_count=0, label=_('Imputed values'))
    output_port = OutputPortDeclaration(role='output', identifier='default', label=_('Output'))
    input_port_declarations = (factors_port, additive_port, impute_port)
    output_port_declarations = (output_port,)

    @classmethod
    def shape_rules(cls, meta: NodeMeta) -> tuple[AnyShapeRule, ...]:
        output = meta.require_output_port('output')
        rules: list[AnyShapeRule] = []
        factors = meta.input_port_ids_for_roles('factors')
        if factors:
            rules.append(ProductShapeRule(inputs=factors, output=output.id))
        same_shaped = meta.input_port_ids_for_roles('additive', 'impute')
        if same_shaped:
            rules.append(SameShapeRule(inputs=same_shaped, output=output.id))
        return tuple(rules)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        try:
            output_unit = meta.require_output_port('output').unit
        except MissingPortRoleError:
            output_unit = None
        for port in candidates:
            tags = {tag for binding in meta.bindings_for_port(port.id) for tag in binding.tags}
            if 'impute' in tags:
                result.classify(port, 'impute', "binding tag 'impute'")
            elif 'non_additive' in tags:
                result.classify(port, 'factors', "binding tag 'non_additive'")
            elif port.unit is None or output_unit is None:
                result.refuse(port, 'cannot classify without both port and output units')
            elif port.unit.is_compatible_with(output_unit):
                result.classify(port, 'additive', f'unit {port.unit} being compatible with output {output_unit}')
            else:
                result.classify(port, 'factors', f'unit {port.unit} being incompatible with output {output_unit}')
        return result

    def lower_to_pipeline_ir(self):
        from nodes.pipeline import InputNodeBinding, MultiplyOperationSpec, PipelineNodeIR, PortInputRef

        if self.input_dataset_instances:
            raise NotImplementedError('MultiplicativeNode2 pipeline lowering does not yet support input datasets')

        operands = resolve_operands(self, unit_of=declared_unit_of)
        if operands.additive:
            raise NotImplementedError(
                'MultiplicativeNode2 pipeline lowering does not yet support additive side inputs '
                + f'({[operand.source_id for operand in operands.additive]})'
            )
        if len(operands.factors) < 2:
            raise NotImplementedError(
                'MultiplicativeNode2 pipeline lowering requires at least two factors '
                + f'({[operand.source_id for operand in operands.factors]})'
            )

        spec = self.spec
        if spec.input_ports and len(spec.input_ports) == len(self.input_nodes):
            port_ids = [port.id for port in spec.input_ports]
        else:
            port_ids = [_runtime_port_id(self.id, idx) for idx, _ in enumerate(self.input_nodes)]

        port_bindings: dict[NodePortIdentifier, PipelinePortBinding] = {
            port_id: InputNodeBinding(node=input_node.id) for port_id, input_node in zip(port_ids, self.input_nodes, strict=True)
        }
        operations: list[AnyOperationSpec] = [
            MultiplyOperationSpec(
                input=PortInputRef(port=port_ids[0]),
                values=[PortInputRef(port=port_id) for port_id in port_ids[1:]],
                result_id='output',
            ),
        ]

        return PipelineNodeIR(
            node_id=self.id,
            source_node_class=f'{type(self).__module__}.{type(self).__qualname__}',
            port_bindings=port_bindings,
            operations=operations,
            output_ref='output',
        )

    def compute(self) -> ppl.PathsDataFrame:
        assert self.unit is not None
        operands = resolve_operands(self, metric=self.get_parameter_value_str('metric', required=False))

        if operands.claimed_elsewhere:
            raise NodeError(
                self,
                'These inputs are tagged for an operation a multiplicative node does not have: %s'
                % ', '.join(operands.claimed_elsewhere),
            )
        if operands.unavailable:
            # A product cannot be computed from a subset of its factors, so unlike addition
            # there is no degraded answer to give.
            raise NodeError(
                self,
                'Cannot multiply: these inputs are unavailable: %s' % ', '.join(operands.unavailable),
            )

        df = multiply_operands(self, operands.factors, self.unit)
        if operands.additive:
            product = Operand(df=df, role='additive', source_id=self.id, kind='node')
            df = sum_operands(self, [product, *operands.additive], self.unit)
        if operands.impute:
            df = impute_operands(self, df, operands.impute)
        return df


class EmissionFactorActivity(MultiplicativeNode):  # FIXME Does not work with Tampere/other_electricity_consumption_emisisons
    explanation = _("""This is an Emission Factor Activity Node. It multiplies an activity by an emission factor.""")
    # FIXME Do we need a separate node class?
    quantity = 'emissions'
    default_unit = '%s/a' % EMISSION_UNIT
    allowed_parameters = [
        *MultiplicativeNode.allowed_parameters,
        BoolParameter(local_id='convert_missing_values_to_zero'),
    ]

    def _get_dataset_emissions(self) -> ppl.PathsDataFrame | None:
        edfs = self.get_input_datasets_pl(tag='emissions')
        ds_list = self.get_input_datasets_pl(exclude_tags=['emissions'])
        if not ds_list:
            return None
        efdf = None  # emission factors
        adf = None  # activity
        for ds in list(ds_list):
            if 'emission_factor' in ds.metric_cols:
                assert efdf is None
                efdf = ds
            else:
                assert adf is None
                adf = ds

        if efdf is None or adf is None:
            raise NodeError(self, 'Missing either emission factor or activity datasets')

        a_metric = adf.metric_cols[0]

        df = adf.paths.join_over_index(efdf, how='outer', index_from='union')
        df = df.multiply_cols([a_metric, 'emission_factor'], 'Emissions').with_columns(pl.col('Emissions').fill_null(0.0))
        df = df.select_metrics(['Emissions'])

        if 'greenhouse_gases' in df.dim_ids:
            df = convert_to_co2e(df, 'greenhouse_gases')
        output_dims = set(self.output_dimensions.keys())
        df_dims = set(df.dim_ids)
        sum_dims = df_dims - output_dims
        if sum_dims:
            df = df.paths.sum_over_dims(list(sum_dims))

        m = self.get_default_output_metric()
        df = df.rename({'Emissions': m.column_id}).ensure_unit(m.column_id, m.unit)

        for edf in edfs:
            edf = edf.rename({edf.metric_cols[0]: '_Right'}).ensure_unit('_Right', m.unit)  # noqa: PLW2901
            df = df.paths.join_over_index(edf, how='outer', index_from='union')
            df = df.with_columns((pl.col(m.column_id).fill_null(0.0) + pl.col('_Right').fill_null(0.0)).alias(m.column_id)).drop(
                '_Right'
            )

        return df

    def compute(self) -> ppl.PathsDataFrame:
        input_df = self._get_dataset_emissions()
        convert = self.get_parameter_value('convert_missing_values_to_zero', required=False)
        df = super()._compute(input_df)
        if convert:
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_nan(pl.lit(0)))
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(pl.lit(0)))
        return df


class PerCapitaActivity(MultiplicativeNode):  # FIXME Remove. Replace with GenericNode
    pass


class FixedScenarioNode(MultiplicativeNode):  # FIXME Inherit from GenericNode instead.
    def compute(self) -> ppl.PathsDataFrame:
        scenario = self.context.scenarios['baseline']
        with scenario.override():
            df = MultiplicativeNode.compute(self)
        return df


class Activity(AdditiveNode):  # FIXME Are these special classes useful?
    explanation = _("""This is Activity Node. It adds activity amounts together.""")
    pass


class FixedMultiplierNode(SimpleNode):  # FIXME Convert to a generic parameter instead.
    explanation = _("""This is a Fixed Multiplier Node. It multiplies a single input node with a parameter.""")
    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        NumberParameter(local_id='multiplier'),
        StringParameter(local_id='global_multiplier'),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output_pl(target_node=self)
        multiplier_param = self.get_parameter('multiplier')  # FIXME Use get_parameter_value() instead.
        multiplier = multiplier_param.get()
        if multiplier_param.has_unit():
            m_unit = multiplier_param.get_unit()
        else:
            m_unit = self.context.unit_registry.parse_units('dimensionless')

        meta = df.get_meta()
        exprs = [pl.col(col) * multiplier for col in meta.metric_cols]
        units = {col: meta.units[col] * m_unit for col in meta.metric_cols}
        df = df.with_columns(exprs)
        for col, unit in units.items():
            df = df.set_unit(col, unit, force=True)

        for metric in self.output_metrics.values():
            df = df.ensure_unit(metric.column_id, metric.unit)

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset_pl(df)
        return df


class MixNode(AdditiveNode):
    output_metrics = {
        MIX_QUANTITY: NodeMetric(unit='%', quantity=MIX_QUANTITY),
    }
    default_unit = '%'
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
    ]
    skip_normalize: bool = False

    def add_mix_normalized(self, df: ppl.PathsDataFrame, nodes: list[Node], over_dims: list[str] | None = None):
        df = self.add_nodes_pl(df=df, nodes=nodes)
        if len(df.metric_cols) != 1:
            raise NodeError(self, 'Must have exactly one metric column')

        # Fill missing values with zeroes
        df = df.paths.to_wide()
        null_fills = [pl.col(col).fill_null(0.0) for col in df.metric_cols]
        df = df.with_columns(null_fills)
        df = df.paths.to_narrow()

        if over_dims is None:
            over_dims = df.dim_ids
        col = df.metric_cols[0]
        df = df.ensure_unit(col, 'dimensionless')
        if not self.skip_normalize:
            # Normalize so that all values are 0 <= x <= 1.0 and
            # the yearly sum is 1.0
            df = df.with_columns(pl.col(col).clip(0, 1))
            sdf = df.paths.sum_over_dims(over_dims).rename({col: '_YearSum'})
            df = df.paths.join_over_index(sdf)
            df = df.divide_cols([col, '_YearSum'], col).drop('_YearSum')

        df = extend_last_historical_value_pl(df, self.get_end_year())
        m = self.get_default_output_metric()
        df = df.ensure_unit(m.column_id, m.unit)
        return df

    def compute(self) -> ppl.PathsDataFrame:
        anode = self.get_input_node(tag='activity')
        adf = anode.get_output_pl(target_node=self)
        am = anode.get_default_output_metric()
        adf = adf.paths.calculate_shares(am.column_id, '_Share')
        m = self.get_default_output_metric()
        df = adf.select_metrics(['_Share']).ensure_unit('_Share', m.unit).rename({'_Share': m.column_id})
        df = extend_last_historical_value_pl(df, self.get_end_year())
        nodes = list(self.input_nodes)
        nodes.remove(anode)
        return self.add_mix_normalized(df, nodes)


class ImprovementNode(MultiplicativeNode):  # FIXME Remove, when you clean Longmont.
    explanation = _("""First does what MultiplicativeNode does, then calculates 1 - result.
    Can only be used for dimensionless content (i.e., fractions and percentages)
    """)

    def compute(self) -> ppl.PathsDataFrame:
        if len(self.input_nodes) == 1:
            node = self.input_nodes[0]
            df = node.get_output_pl(target_node=self)
        else:
            df = super().compute()
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.from_pandas(df)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        df = df.with_columns((pl.lit(1) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

        return df


class ImprovementNode2(MultiplicativeNode):  # FIXME Remove, when you clean Longmont.
    explanation = _("""First does what MultiplicativeNode does, then calculates 1 + result.
    Can only be used for dimensionless content (i.e., fractions and percentages)
    """)

    def compute(self) -> ppl.PathsDataFrame:
        if len(self.input_nodes) == 1:
            node = self.input_nodes[0]
            df = node.get_output_pl(target_node=self)
        else:
            df = super().compute()
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.from_pandas(df)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        df = df.with_columns((pl.lit(1) + pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

        return df


class RelativeNode(AdditiveNode):  # FIXME Remove. Only Espoo and budget use this.
    explanation = _("""
    First like AdditiveNode, then multiply with a node with "non_additive".
    The relative node is assumed to be the relative difference R = V / N - 1,
    where V is the expected output value and N is the comparison value from
    the other input nodes. So, the output value V = (R + 1)N.
    If there is no "non-additive" node, it will behave like AdditiveNode except
    it never creates a temporary dimension Sectors.
    """)

    def compute(self) -> ppl.PathsDataFrame:
        n = self.get_input_node(tag='non_additive', required=False)
        df = super().compute()
        if n is not None:
            dfn = n.get_output_pl(target_node=self)
            if dfn.get_unit(VALUE_COLUMN).dimensionless:
                dfn = dfn.ensure_unit(VALUE_COLUMN, 'dimensionless')
            df = df.paths.join_over_index(dfn, how='outer', index_from='union')
            rn = VALUE_COLUMN + '_right'
            df = df.with_columns([pl.col(rn).fill_null(0)])
            df = df.with_columns(pl.col(rn) + pl.lit(1))
            df = df.multiply_cols([VALUE_COLUMN, rn], VALUE_COLUMN).drop(rn)
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class FillNewCategoryNode(AdditiveNode):
    explanation = _(
        """This is a Fill New Category Node. It behaves like Additive Node, but in the end of computation
        it creates a new category such that the values along that dimension sum up to 1. The input nodes
        must have a dimensionless unit. The new category in an existing dimension is given as parameter
        'new_category' in format 'dimension:category
        """
    )
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(local_id='new_category'),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        category = self.get_parameter_value_str('new_category', required=True)
        dim, cat = category.split(':')

        df: ppl.PathsDataFrame = self.add_nodes_pl(None, self.input_nodes)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        df2 = df.paths.sum_over_dims(dim)
        df2 = df2.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))
        df2 = df2.with_columns(pl.lit(cat).cast(pl.Categorical).alias(dim))
        df2 = df2.select(df.columns)

        df = df.paths.concat_vertical(df2)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        if self.get_parameter_value('drop_nans', required=False):  # FIXME Not consistent with the parameter name!
            df = df.paths.to_wide()
            for col in df.metric_cols:
                df = df.filter(~pl.col(col).is_null())
            df = df.paths.to_narrow()
        return df


class FillNewCategoryNode2(AdditiveNode):  # FIXME Merge into FillNewCategoryNode
    explanation = _(
        """This is a Fill New Category Node.

        It behaves like Additive Node, but in the end of computation
        it creates a new category such that the values along that dimension sum up to 1. The input nodes
        must have a dimensionless unit. The new category in an existing dimension is given as parameter
        'new_category' in format 'dimension:category
        """
    )
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(local_id='new_category'),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df: ppl.PathsDataFrame = self.add_nodes_pl(None, self.input_nodes)

        df = self.fill_new_category(df)
        return df

    def fill_new_category(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        category = self.get_parameter_value_str('new_category', required=True)
        dim, cat = category.split(':')

        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        df2 = df.paths.sum_over_dims(dim)
        df2 = df2.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))
        df2 = df2.with_columns(pl.lit(cat).cast(pl.Categorical).alias(dim))
        df2 = df2.select(df.columns)

        df = df.paths.concat_vertical(df2)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        if self.get_parameter_value('drop_nans', required=False):  # FIXME Not consistent with the parameter name!
            df = df.paths.to_wide()
            for col in df.metric_cols:
                df = df.filter(~pl.col(col).is_null())
            df = df.paths.to_narrow()
        return df


class ChooseInputNode(AdditiveNode):
    explanation = _(
        """
        This is a ChooseInputNode. It can have several input nodes, and it selects the one that has the same
        tag as given in the parameter node_tag. The idea of the node is that you can change the parameter value
        in the scenario and thus have different nodes used in different contexts.
        """
    )
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(local_id='node_tag', label=_('Tag to use as selecting the input node')),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        node_tag = self.get_parameter_value_str('node_tag', required=True)
        df = self.get_input_node(tag=node_tag).get_output_pl(target_node=self)
        return df


class RelativeYearScaledNode(AdditiveNode):
    explanation = _(
        """
        This is RelativeYearScaledNode. First it acts like additive node.
        In the end, everything is scaled by the values of the reference year.
        The reference year is either the instance reference year or from parameter.
        """
    )
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        NumberParameter(local_id='reference_year', label=_('The year whose values are used for scaling')),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = AdditiveNode.compute(self)
        year = self.get_parameter_value_int('reference_year', required=False)
        if not year:
            year = self.context.instance.reference_year
            assert year is not None
        df = df.paths._scale_by_reference_year(df, year)
        return df


class AnnuityNode(AdditiveNode):
    def compute(self) -> ppl.PathsDataFrame:
        targetyear = self.get_target_year()
        outputdf = ppl.PathsDataFrame()

        discountnode = self.get_input_node(tag='discount_rate')
        discountdf = discountnode.get_output_pl(target_node=self)
        discountdf = (
            discountdf
            .drop(FORECAST_COLUMN)
            .rename({VALUE_COLUMN: 'discount_rate'})
            .with_columns(pl.col('discount_rate') / pl.lit(100.0))
        )

        inputnodes = self.get_input_nodes()
        inputnodes.remove(discountnode)
        inputdf = inputnodes[0].get_output_pl(target_node=self)
        meta = inputdf.get_meta()
        for node in inputnodes[1:]:
            nodedf = node.get_output_pl(target_node=self)
            inputdf.extend(nodedf.select(inputdf.columns))

        dimensions = list(self.input_dimensions.keys())
        partitions = inputdf.partition_by(dimensions)
        for partdf in partitions:
            annuitydf = (
                partdf
                .join(discountdf, on=YEAR_COLUMN, how='left')
                .with_columns(((pl.lit(1.0) + pl.col('discount_rate')) ** pl.col('term')).alias('compound_factor'))
                .with_columns(
                    ((pl.col('discount_rate') * pl.col('compound_factor')) / (pl.col('compound_factor') - pl.lit(1.0))).alias(
                        'capital_recovery_factor'
                    )
                )
                .with_columns(
                    (
                        pl
                        .when(pl.col('capital_recovery_factor').is_not_nan())
                        .then(pl.col('currency') * pl.col('capital_recovery_factor'))
                        .otherwise(pl.col('currency') / pl.col('term'))
                    ).alias('annual_payment')
                )
                .with_columns(pl.int_ranges(pl.col(YEAR_COLUMN), pl.col(YEAR_COLUMN) + pl.col('term')).alias('payment_years'))
                .explode('payment_years')
                .group_by('payment_years')
                .agg(pl.col('annual_payment').sum())
                .rename({'payment_years': YEAR_COLUMN})
                .sort(YEAR_COLUMN)
            )

            joindf = (
                partdf
                .join(annuitydf, on=YEAR_COLUMN, how='outer', coalesce=True)
                .with_columns(pl.col('annual_payment').alias('currency'))
                .drop(['term', 'annual_payment'])
                .filter(pl.col(YEAR_COLUMN).le(targetyear))
                .fill_null(strategy='forward')
            )

            pathsdf = ppl.to_ppdf(joindf, meta=meta)
            pathsdf = pathsdf.rename({'currency': VALUE_COLUMN})
            if outputdf.is_empty():
                outputdf = pathsdf
            else:
                outputdf.extend(pathsdf)

        return outputdf


class DiscountNode(AdditiveNode):
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        NumberParameter(local_id='start_year', label=_('The first year in which the discount rate is applied.')),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        minyear = self.get_parameter_value_int('start_year', required=True) - 1

        ratenode = self.get_input_node(tag='discount_rate')
        ratedf = ratenode.get_output_pl(target_node=self)
        ratedf = (
            ratedf
            .drop(FORECAST_COLUMN)
            .rename({VALUE_COLUMN: 'discount_rate'})
            .with_columns((pl.col('discount_rate') + pl.lit(100.0)) / pl.lit(100.0))
        )

        currencynode = self.get_input_node(tag='currency')
        df = currencynode.get_output_pl(target_node=self)
        df = (
            df.paths
            .join_over_index(ratedf)
            .with_columns((pl.col(YEAR_COLUMN) - pl.lit(minyear)).clip(lower_bound=0).alias('year_count'))
            .with_columns(pl.col(VALUE_COLUMN) / (pl.col('discount_rate') ** pl.col('year_count')))
            .drop(['discount_rate', 'year_count'])
        )

        return df
