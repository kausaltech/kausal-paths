from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.calc import convert_to_co2e
from nodes.constraints.port_roles import PortRoleInferenceResult
from nodes.defs.port_def import InputPort, InputPortDeclaration
from nodes.operands import Operand, sum_operands
from nodes.simple import AdditiveNode, MultiplicativeNode, SimpleNode
from params.param import NumberParameter, StringParameter

from .constants import FORECAST_COLUMN, TIME_INTERVAL, VALUE_COLUMN, YEAR_COLUMN
from .exceptions import NodeError
from .node import Node
from .units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nodes.defs.port_def import InputPortDef
    from nodes.instance_graph import NodeMeta


class FloorAreaNode(MultiplicativeNode):  # FIXME Rebuild this with modern tools
    explanation = _('Floor area node takes in actions and calculates the floor area impacted.')
    output_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']  # FIXME Generalise and remove emission_sectors
    input_dimension_ids = ['building_energy_class', 'emission_sectors']

    floor_area_port = InputPort.one('floor_area', label=_('Floor area'))
    triggered_port = InputPort.multi('triggered', label=_('Share of floor area an action triggers'))
    compliant_port = InputPort.multi('compliant', label=_('Share of triggered area that complies'))
    input_port_declarations: ClassVar[tuple[InputPortDeclaration, ...]] = (
        floor_area_port,
        triggered_port,
        compliant_port,
    )
    consumes_all_inputs_through_ports = True

    metric_roles: ClassVar[dict[str, str]] = {'triggered': 'triggered', 'compliant': 'compliant'}
    """Source metric column -> this class's role for it. CfNode reads a different metric."""

    base_role: ClassVar[str] = 'floor_area'
    """Role for the single-metric operand the class multiplies the action shares against."""

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        """
        Classify by the source metric each port already selects.

        The parser expands one legacy ``input_nodes`` entry against a multi-metric action
        into one port per metric, each bound to that metric's output port, so the split
        this class used to do with ``isinstance(node, CfFloorAreaAction)`` and column
        names is already present in the graph. Reading ``source_port.column_id`` recovers
        it without the class having to know which node classes are actions.
        """
        from nodes.defs.binding_def import EdgeBindingDef

        result = PortRoleInferenceResult()
        for port in candidates:
            edges = [binding for binding in meta.bindings_for_port(port.id) if isinstance(binding, EdgeBindingDef)]
            if not edges:
                result.refuse(port, 'only a node input can be an operand of this class')
                continue
            columns = {str(edge.source_port.column_id) for edge in edges}
            if len(columns) != 1:
                result.refuse(port, f'port mixes source metrics {sorted(columns)}')
                continue
            column = columns.pop()
            if column in cls.metric_roles:
                result.classify(port, cls.metric_roles[column], f'source metric {column!r}')
            elif column == VALUE_COLUMN:
                result.classify(port, cls.base_role, 'a single-metric source')
            else:
                result.refuse(port, f'source metric {column!r} is not an operand of this class')
        return result

    def _metric_by_source(self, port: InputPortDeclaration, column: str) -> dict[str, ppl.PathsDataFrame]:
        """
        Resolve a per-action metric role, keyed by the source node that supplied it.

        Each action reaches this node as one binding per metric, so the metrics that used
        to arrive together in one frame have to be paired back up by source. Insertion
        order follows binding position, which is the order the old single pass over
        ``input_nodes`` used.
        """
        frames: dict[str, ppl.PathsDataFrame] = {}
        for binding in self.iter_input_bindings(port):
            source = binding.source
            assert isinstance(source, Node)
            frame = self.resolve_input_binding(binding)
            frames[source.id] = frame.rename({frame.metric_cols[0]: column})
        return frames

    def include_custom_dimension(self, df: ppl.PathsDataFrame):  # Dimension must be explained in column name in the right syntax
        df = df.paths.to_wide()  # Make column names consistent
        for s in df.columns:
            arr = s.split('@')
            if len(arr) > 2:
                s2 = '@'.join(arr[:2]) + '/' + ''.join(arr[2:])
                df = df.rename({s: s2})
        df = df.paths.to_narrow()

        return df

    def compute(self):
        df: ppl.PathsDataFrame = self.require_input(self.floor_area_port)

        # Existing (old) and new floor area in baseline
        flhv = df.get_last_historical_values()
        flhv = flhv.rename({flhv.metric_cols[0]: 'floor_old'})
        df_bau = df.paths.join_over_index(flhv.drop([YEAR_COLUMN, FORECAST_COLUMN]))
        df_bau = df_bau.with_columns(
            pl.when(pl.col(FORECAST_COLUMN)).then(pl.col('floor_old')).otherwise(pl.col(VALUE_COLUMN)).alias('floor_old')
        )
        df_bau = df_bau.with_columns((pl.col(VALUE_COLUMN) - pl.col('floor_old')).alias('floor_new'))
        # FIXME Bubblegum fix for wrong unit treatment in diff:
        df_bau = df_bau.set_unit('floor_new', df_bau.get_unit('floor_old') * unit_registry.parse_units(TIME_INTERVAL))
        df_bau = df_bau.diff('floor_new').with_columns(pl.col('floor_new').fill_null(0))
        df_bau = df_bau.drop(VALUE_COLUMN)

        # Add or update dimension building_energy_class
        if 'building_energy_class' in df.dim_ids:
            df_bau = df_bau.with_columns(
                pl
                .when(pl.col('building_energy_class').eq('existing'))
                .then(pl.col('floor_old'))
                .otherwise(pl.col('floor_new'))
                .alias('floor_old')
            )
            df_bau = df_bau.drop('floor_new')
            df_bau = df_bau.rename({'floor_old': 'floor_area'})
        else:
            col = 'floor_area@building_energy_class:'
            df_bau = df_bau.rename({'floor_old': col + 'existing', 'floor_new': col + 'new'})
            df_bau = self.include_custom_dimension(df_bau)

        triggered = self._metric_by_source(self.triggered_port, 'triggered')
        compliant = self._metric_by_source(self.compliant_port, 'compliant')
        if set(triggered) != set(compliant):
            raise NodeError(
                self,
                'Every action must supply both a triggered and a compliant share; got %s against %s.'
                % (sorted(triggered), sorted(compliant)),
            )

        df_out = None
        for action_id, triggered_df in triggered.items():
            # The two shares arrive as separate single-metric bindings; rejoining them
            # reconstructs the frame the action's multi-metric output used to deliver.
            df = triggered_df.paths.join_over_index(compliant[action_id])
            df = df.ensure_unit('triggered', 'dimensionless')
            df = df.ensure_unit('compliant', 'dimensionless')

            df = df.paths.join_over_index(df_bau)
            df = df.with_columns(
                pl
                .when(pl.col('building_energy_class').eq(pl.lit('new')))
                .then(pl.lit(1.0))
                .otherwise(pl.col('triggered'))
                .alias('triggered')
            )

            df = df.multiply_cols(['floor_area', 'triggered', 'compliant'], 'floor_area')

            df = df.rename({'floor_area': 'floor_area@action:' + action_id})
            df = df.drop(['triggered', 'compliant'])
            df = self.include_custom_dimension(df)

            if df_out is None:
                df_out = df
            else:
                meta = df.get_meta()
                df_out = ppl.to_ppdf(pl.concat([df_out, df], rechunk=True), meta)

        assert df_out is not None
        df_out = df_out.ensure_unit('floor_area', self.unit)
        df_out = df_out.with_columns(pl.col('floor_area').alias(VALUE_COLUMN))

        m = self.get_default_output_metric()
        df_out = df_out.select_metrics(['floor_area']).rename({'floor_area': m.column_id})

        return df_out


class CfNode(FloorAreaNode):
    """
    Consumption factor (CF) describes the energy saving caused by the action.

    There must be at least one action of type energy_saving.CfFloorAreaAction.
    """

    output_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']
    input_dimension_ids = ['building_energy_class', 'emission_sectors']

    # A CfNode reads the actions' improvement metric, and unlike FloorAreaNode it may take
    # any number of baseline inputs — or none — and adds them together.
    baseline_port = InputPort.multi('baseline', required=False, aggregation='sum', label=_('Baseline inputs'))
    improvement_port = InputPort.multi('improvement', label=_('Consumption factor improvement per action'))
    input_port_declarations: ClassVar[tuple[InputPortDeclaration, ...]] = (baseline_port, improvement_port)

    metric_roles: ClassVar[dict[str, str]] = {'improvement': 'improvement'}
    base_role: ClassVar[str] = 'baseline'

    def compute(self):
        improvements = self._metric_by_source(self.improvement_port, VALUE_COLUMN)
        assert len(improvements) > 0

        df = None
        for action_id, df_a in improvements.items():
            if df is None:
                df = df_a
            else:
                df = df_a.paths.join_over_index(df, index_from='union')
                df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(pl.lit(0)))

            col = VALUE_COLUMN + '@action:' + action_id
            df = df.with_columns(pl.col(VALUE_COLUMN).alias(col))
            df = df.drop(VALUE_COLUMN)

        assert df is not None
        df = self.include_custom_dimension(df)

        # Inputs nodes are baseline but not required.
        # If actions are not in the same units as the baseline, they are assumed to be relative values.
        baseline = [
            Operand(
                df=self.resolve_input_binding(binding),
                role='additive',
                source_id=binding.source_id or str(binding.id),
                kind=binding.source_kind,
            )
            for binding in self.iter_input_bindings(self.baseline_port)
        ]
        if baseline:
            assert self.unit is not None
            df_bau = sum_operands(self, baseline, self.unit)
            df = df.paths.join_over_index(df_bau)
            sub = self.is_compatible_unit(df.get_unit(VALUE_COLUMN), df.get_unit(VALUE_COLUMN + '_right'))
            if sub:
                assert sub is False  # Because the use case is unclear
                df = df.subtract_cols([VALUE_COLUMN + '_right', VALUE_COLUMN], VALUE_COLUMN)
            else:
                df = df.multiply_cols([VALUE_COLUMN + '_right', VALUE_COLUMN], VALUE_COLUMN)
            df = df.drop(VALUE_COLUMN + '_right')

        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df


class EnergyNode(MultiplicativeNode):
    """
    Takes the floor area and consumption factor categorized by building energy class and action.

    This energy saving is accumulated over time to reflect the situation that
    the energy use of a building stays constant after renovation.
    However, accumulation can be prevented by using the parameter not_cumulated.
    """

    allowed_parameters = [
        *MultiplicativeNode.allowed_parameters,
        StringParameter(
            local_id='not_cumulated',
            description='Action that is not cumulated',
            is_customizable=False,
        ),
    ]

    input_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']
    output_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']

    def compute(self):
        df = super().compute()
        df = df.with_columns([pl.col(VALUE_COLUMN).alias('cumulated')])
        df = df.cumulate('cumulated')

        not_cumulated = self.get_parameter_value('not_cumulated', required=False)
        if not_cumulated is not None:
            df = df.with_columns(  # FIXME fails to pick non-cumulated action
                pl
                .when(pl.col('action') == not_cumulated)
                .then(pl.col(VALUE_COLUMN))
                .otherwise(pl.col('cumulated'))
                .alias(VALUE_COLUMN)
            )

        df = df.drop(['cumulated'])

        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class HistoricalNode(AdditiveNode):
    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        df = df.filter(~pl.col(FORECAST_COLUMN))
        return df


class CCSNode(SimpleNode):
    emissions_port = InputPortDeclaration(role='emissions')
    ccs_share_port = InputPortDeclaration(role='ccs_share')
    input_port_declarations: ClassVar[tuple[InputPortDeclaration, ...]] = (emissions_port, ccs_share_port)
    legacy_input_port_roles_by_tag = {'emissions': 'emissions', 'ccs_share': 'ccs_share'}
    allowed_parameters = [
        NumberParameter(local_id='capture_efficiency', unit_str='%', is_customizable=True),
        NumberParameter(local_id='storage_efficiency', unit_str='%', is_customizable=True),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.emissions_port)
        df = df.rename({VALUE_COLUMN: 'Emissions'})

        sdf = self.require_input(self.ccs_share_port)
        sdf = sdf.rename({VALUE_COLUMN: 'CCSShare'}).ensure_unit('CCSShare', 'dimensionless')

        df = df.paths.join_over_index(sdf)
        df = df.with_columns(pl.col('CCSShare').fill_null(0.0))

        capt_eff = self.get_parameter_value('capture_efficiency', units=True).to('dimensionless').m
        df = df.multiply_cols(['Emissions', 'CCSShare'], 'Captured')
        df = df.with_columns(
            pl.when(pl.col('greenhouse_gases').is_in(('co2', 'co2_biogen'))).then(pl.col('Captured') * capt_eff).otherwise(0.0)
        )

        storage_eff = self.get_parameter_value('storage_efficiency', units=True).to('dimensionless').m
        u = df.get_unit('Captured')
        df = (
            df
            .with_columns([
                (pl.col('Captured') * storage_eff).alias('Stored'),
                (pl.col('Emissions') - pl.col('Captured')).alias('Remaining'),
            ])
            .with_columns([(pl.col('Captured') - pl.col('Stored')).alias('StorageLoss')])
            .set_unit('Remaining', u)
            .set_unit('StorageLoss', u)
        )

        m = self.get_default_output_metric()
        rdf = df.select_metrics('Remaining', rename=m.column_id).with_columns(pl.lit('scope1').alias('emission_scope'))
        sdf = (
            df
            .select_metrics('Stored', rename=m.column_id)
            .filter(pl.col('greenhouse_gases').eq('co2_biogen'))
            .with_columns([
                pl.lit('negative_emissions').alias('emission_scope'),
                # use co2 to be able to convert to GWP
                pl.lit('co2', dtype=pl.Categorical).alias('greenhouse_gases'),
                (-pl.col(m.column_id)).alias(m.column_id),
            ])
        )
        ldf = df.select_metrics('StorageLoss', rename=m.column_id).with_columns(pl.lit('scope3').alias('emission_scope'))

        df = ppl.to_ppdf(pl.concat([rdf, sdf, ldf]), rdf.get_meta()).add_to_index('emission_scope')
        df = convert_to_co2e(df, 'greenhouse_gases')
        df = df.ensure_unit(m.column_id, m.unit)
        return df
