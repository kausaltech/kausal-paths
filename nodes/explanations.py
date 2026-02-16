from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from common.i18n import gettext as _

from .constants import TIME_INTERVAL
from .formula import (
    FormulaSpec,
    UnitOverride,
    analyze_formula_dimensions,
    analyze_formula_units,
    build_name_dimension_map,
    build_name_unit_map,
    collect_term_names,
    make_identifier,
    normalize_formula_identifiers,
)
from .units import Unit, unit_registry

if TYPE_CHECKING:
    from nodes.context import Context

TAG_TO_BASKET = {
    'additive': 'add',
    'add_from_incoming_dims': 'add_from_incoming_dims',
    'add_to_existing_dims': 'add_to_existing_dims',
    'base': 'other',
    # These operations only handle existing df and do not take new inputs:
    # apply_multipliet, do_correction, drop_nand, drop_infs, extend_values, extrapolate
    # [concat_datasets: generalize to use the port theory]
    # inventory_only, other, select_variant
    'non_additive': 'multiply',
    'other_node': 'other',
    'primary': 'coalesce',
    'rate': 'other',
    'secondary': 'coalesce',
    'skip_dim_test': 'skip_dim_test',
    'split_by_existing_shares': 'split_by_existing_shares',
    'split_evenly_to_cats': 'split_evenly_to_cats',
    'use_as_shares': 'use_as_shares',
    'use_as_totals': 'use_as_totals',
}

BASKET_DISPLAY_NAMES = { # FIXME We may not need explicit basket names.
    'add': _('addition'),
    'add_from_incoming_dims': _('addition from incoming dimensions'),
    'add_to_existing_dims': _('addition to existing dimensions'),
    'coalesce': _('coalesce'),
    'multiply': _('multiplication'),
    'other': _('other operations'),
    'skip_dim_test': _('skip dimension test'),
    'split_by_existing_shares': _('split by existing shares'),
    'split_evenly_to_cats': _('split evenly to categories'),
    'use_as_shares': _('use as shares'),
    'use_as_totals': _('use as totals'),
    'unknown': _('unknown operation'),
    'skip': _('skip'),
}

BASKET_OPERATION_LABEL = {
    'add': ' + ',
    'add_from_incoming_dims': ' + ',
    'add_to_existing_dims': ' + ',
    'coalesce': ', ',
    'multiply': ' * ',
}

TAG_DESCRIPTIONS = {
    'add_datasets': _('Get and prepare each dataset, then add them together.'),
    'additive': _("Add input node values (even if the units don't match with the node units)."),
    'and': _('Logical AND: min(a, b). Warns if inputs deviate from 0 or 1 (see node explanation).'),
    'arithmetic_inverse': _('Take the arithmetic inverse of the values (-x).'),
    'bring_to_maximum_historical_year': _('Makes all years up to maximum historical year non-forecasts.'),
    'complement': _('Take the complement of the dimensionless values (1-x).'),
    'complement_cumulative_product': _('Take the cumulative product of the dimensionless complement values over time.'),
    'concat_datasets': _('Get and concatenate datasets vertically, only then prepare the output.'),
    'cumulative': _('Take the cumulative sum over time.'),
    'cumulative_product': _('Take the cumulative product of the dimensionless values over time.'),
    'difference': _('Take the difference over time (i.e. annual changes)'),
    'empty_to_zero': _('Convert NaNs to zeros.'),
    'expectation': _('Take the expected value over the uncertainty dimension.'),
    'extend_all': _('Extend the values to all the remaining missing years.'),
    'extend_both_ways': _('Extend the values beyond the first and last values, but do not interpolate.'),
    'extend_forecast_values': _('Extend the last forecast values to the remaining missing years.'),
    'extend_to_history': _('Extend the first values to the years after the minimum historical year.'),
    'extend_values': _('Extend the last historical values to the remaining missing years.'),
    'geometric_inverse': _('Take the geometric inverse of the values (1/x).'),
    'get_single_dataset': _('Get a single dataset if it exists.'),
    'goal': _('The node is used as the goal for the action.'),
    'historical': _('The node is used as the historical starting point.'),
    'existing': _('This is used as the baseline.'),
    'incoming': _('This is used for the incoming stock.'),
    'ignore_content': _('Show edge on graphs but ignore upstream content.'),
    'inserting': _('This is the rate of new stock coming in.'),
    'inventory_only': _('Truncate the forecast values.'),
    'make_nonnegative': _('Negative result values are replaced with 0.'),
    'make_nonpositive': _('Positive result values are replaced with 0.'),
    'max': _('Element-wise maximum of two values; max(a, b). For 0/1 inputs this is logical OR.'),
    'min': _('Element-wise minimum of two values; min(a, b). For 0/1 inputs this is logical AND.'),
    'non_additive': _('Input node values are not added but operated despite matching units.'),
    'observed_only_extend_all': _('Extend the observed data only based on the observed data points.'),
    'or': _('Logical OR: max(a, b). Warns if inputs deviate from 0 or 1 (see node explanation).'),
    'prepare_gpc_dataset': _('Prepare a GPC-styledataset for use.'),
    'primary': _('Use data as primary values even if a secondary value exists.'),
    'ratio_to_last_historical_value': _('Take the ratio of the values compared with the last historical value.'),
    'removing': _('This is the rate of stock removal.'),
    'round_to_five': _('Round values to 5 significant digits rather than 5 decimal places.'),
    'secondary': _('Use data only if a primary value does not exist.'),
    'select_port': _('If condition is True, select the first option, otherwise the second.'),
    'truncate_before_start': _('Truncate values before the reference year. There may be some from data'),
    'truncate_beyond_end': _('Truncate values beyond the model end year. There may be some from data'),
}

def _unit_dimensionless(_unit: Unit | None) -> Unit:
    return unit_registry.parse_units('dimensionless')


def _unit_mul_time(unit: Unit | None) -> Unit | None:
    if unit is None:
        return None
    return cast("Unit", unit * unit_registry(TIME_INTERVAL))


def _unit_div_time(unit: Unit | None) -> Unit | None:
    if unit is None:
        return None
    return cast("Unit", unit / unit_registry(TIME_INTERVAL))


def _unit_geometric_inverse(unit: Unit | None) -> Unit | None:
    if unit is None:
        return None
    return cast("Unit", unit_registry.parse_units('dimensionless') / unit)


def _unit_passthrough(unit: Unit | None) -> Unit | None:
    return unit


FORMULA_FUNCTION_UNIT_OVERRIDES = {
    # Produces a unitless ratio even when input has units.
    'ratio_to_last_historical_value': _unit_dimensionless,
    # Cumulative sum/diff adjust by timestep.
    'cumulative': _unit_mul_time,
    'difference': _unit_div_time,
    # Invert unit (1 / unit).
    'geometric_inverse': _unit_geometric_inverse,
    # Keep unit unchanged.
    'ignore_content': _unit_passthrough,
}


@dataclass
class NodeInfo:
    description: str
    deprecated: bool = False


# FIXME Make descriptions concise.
NODE_CLASS_DESCRIPTIONS: dict[str, NodeInfo] = {
    'AdditiveAction': NodeInfo(_("""Simple action that produces an additive change to a value.""")),
    'AdditiveNode': NodeInfo(_("")),
    'AlasEmissions': NodeInfo(
        _("""AlasEmissions is a specified node to handle emissions from the ALas model by Syke."""),
        deprecated=True),
    'AlasNode': NodeInfo(
        _("""AlasNode is a specified node to handle data from the ALas model by Syke."""),
        deprecated=True),
    'AttributableFractionRR': NodeInfo(_(
        """
        Calculate attributable fraction when the ERF function is relative risk.

        AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
        is smaller than 0, we should use r instead. It can be converted from the result:
        r/(r+1)=s <=> r=s/(1-s)
        """)),
    'BuildingEnergySavingAction': NodeInfo(_(
        """
        Action that has an energy saving effect on building stock (per floor area).

        The output values are given per TOTAL building floor area,
        not per RENOVATEABLE building floor area. This is useful because
        the costs and savings from total renovations sum up to a meaningful
        impact on nodes that are given per floor area.
        """)),
    'BuildingEnergySavingActionUs': NodeInfo(_(
        """BuildingEnergySavingAction with U.S. units and natural gas instead of heat.""")),
    'CfFloorAreaAction': NodeInfo(_(
        """
        Action that has an energy saving effect on building stock (per floor area).

        The output values are given per TOTAL building floor area,
        not per RENOVATEABLE building floor area. This is useful because
        the costs and savings from total renovations sum up to a meaningful
        impact on nodes that are given per floor area.

        Outputs:
        # fraction of existing buildings triggering code updates
        # compliance of new buildings to the more active regulations
        # improvement in energy consumption factor
        """)),
    'ChpNode': NodeInfo(_(
        """
        Calculates the attribution of emissions of combined heat and power (CHP) producion according to the
        <a href="https://ghgprotocol.org/sites/default/files/CHP_guidance_v1.0.pdf">GPC protocol</a>.
        There are several methods but for each method, the average emission factor of the fuel mix used
        is split into two parts by using fractions a_heat and a_electricity that sum up to 1. They are calculated as
        <br/>a<sub>i</sub> = z<sub>i</sub> * f<sub>i</sub> / sum<sub>i</sub>(z<sub>i</sub> * f<sub>i</sub>),
        <br/>where a<sub>i</sub> is the fraction for each product (i = electricity, heat)
        z<sub>i</sub> is a method-specific multiplier (see below)
        f<sub>i</sub> is the fraction of product i from the total energy produced.

        <ol><li><b>Energy method</b>
        Logic: Energy products are treated equally based on the energy content.
        All z<sub>i</sub> = 1

        </li><li><b>Work potential method</b> (aka Carnot method, or exergetic method)
        Logic: Energy products are treated equally based on the potential of doing work (i.e., exergy content).
        This moves emissions toward electricity.
        z<sub>heat</sub> = 1 - T<sub>return</sub> / T<sub>supply</sub>, and
        z<sub>electricity</sub> = 1.
        T<sub>return</sub> and T<sub>supply</sub> are the output and input process temperatures, respectively.

        </li><li><b>Bisko method</b>
        Bisko method is a variant of the work potential method.
        The only difference is that Bisko assumes T<sub>return</sub> = 283 K.

        </li><li><b>Efficiency method</b>
        Logic: What emissions would have occured if each energy product had been produced separately?
        z<sub>i</sub> = 1 / n<sub>i</sub>,
        where n<sub>i</sub> is the reference efficiency for producing the energy type separately.
        Typical values are z<sub>heat</sub> = 0.9, z<sub>electricity</sub> = 0.4.</li></ol>
        """)),
    'CoalesceNode': NodeInfo(_(
        "Uses 'primary' tagged data when available, otherwise 'secondary' tagged data. One of the tags must be given.")),
    'CohortNode': NodeInfo(_(
        """
        Cohort node takes in initial age structure (inventory) and follows the cohort in time as it ages.

        Harvest describes how much is removed from the cohort.
        """)),
    'ConstantNode': NodeInfo(_(
        """
        Constant node returns a constant value spread over the timeline.
        """)),
    'CumulativeAdditiveAction': NodeInfo(_(
        """Additive action where the effect is cumulative and remains in the future.""")),
    'DatasetDifferenceAction': NodeInfo(_(
        """
        Receive goal input from a dataset or node and cause an effect.

        The output will be a time series with the difference to the
        predicted baseline value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """)),
    'DatasetDifferenceAction2': NodeInfo(_(
        """
        Receive goal input from a dataset or node and cause an effect.

        The output will be a time series with the difference to the
        predicted baseline value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """)),
    'DatasetNode': NodeInfo(_(
        """
        This is a DatasetNode. It takes in a specifically formatted dataset and
        converts the relevant part into a node output.
        """)),
    'DatasetReduceAction': NodeInfo(_(
        """
        Receive goal input from a dataset or node and cause a linear effect.

        The output will be a time series with the difference to the
        last historical value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """)),
    'DatasetReduceNode': NodeInfo(_(
        """
        Receive goal input from a dataset or node and cause a linear effect.

        The output will be a time series with the difference to the
        last historical value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """)),
    'DatasetRelationAction': NodeInfo(_(
        """
        ActionRelationshipNode enforces a logical relationship with another action node.

        This node monitors an upstream action node (A) and automatically sets its own
        enabled state (B) according to the relationship specified in the edge tags.
        """)),
    'DilutionNode': NodeInfo(_(
        """
        This is Dilution Node. It has exactly four input nodes which are marked by tags: 1) existing is the current,
        non-diluted variable. 2) Incoming is the variable which diluted the existing one with its different values. 3)
        Removing is the fraction that is removed from the existing stock each year. 4) Incoming is the ratio compared
        with the existing stock that is inserted into the system. (Often the removed and incoming values are the same,
        and then the stock size remains constant.)
        """)),
    'DimensionalSectorEmissionFactor': NodeInfo(_(
        "Filters emissions and energy according to the <i>sector</i> parameter and calculates emission factor.")),
    'DimensionalSectorEmissions': NodeInfo(_("Filters emissions according to the <i>sector</i> parameter.")),
    'DimensionalSectorEnergy': NodeInfo(_("Filters energy use according to the <i>sector</i> parameter.")),
    'DimensionalSectorNode': NodeInfo(_(
        "Reads in a dataset and filters and interprets its content according to the <i>sector</i> parameter.")),
    'EnergyAction': NodeInfo(_("""Simple action with several energy metrics.""")),
    'ExponentialNode': NodeInfo(_(
        """
        This is Exponential Node.
        Takes in either input nodes as AdditiveNode, or builds a dataframe from current_value.
        Builds an exponential multiplier based on annual_change and multiplies the VALUE_COLUMN.
        Optionally, touches also historical values.
        Parameter is_decreasing_rate is used to give discount rates instead.
        """)),
    'FillNewCategoryNode': NodeInfo(_(
        """This is a Fill New Category Node. It behaves like Additive Node, but in the end of computation
        it creates a new category such that the values along that dimension sum up to 1. The input nodes
        must have a dimensionless unit. The new category in an existing dimension is given as parameter
        'new_category' in format 'dimension:category
        """)),
    'FixedMultiplierNode': NodeInfo(_(
        """This is a Fixed Multiplier Node. It multiplies a single input node with a parameter.""")),
    'FloorAreaNode': NodeInfo(_('Floor area node takes in actions and calculates the floor area impacted.')),
    'FormulaNode': NodeInfo(_('')),
    'GenerationCapacityNode': NodeInfo(_("""
        Calculates generation of energy when new capacity is installed. Includes scope 3 emissions from installation,
        and emissions avoided from the capacity that gets replaced.
        """)),
    'GenericNode': NodeInfo(_("")),
    'GenericAction': NodeInfo(_("")),
    'GpcTrajectoryAction': NodeInfo(_(
        """
        GpcTrajectoryAction is a trajectory action that uses the DatasetNode to fetch the dataset.
        """)),
    'InternalGrowthModel': NodeInfo(_(
        """
        Calculates internal growth of e.g. a forest, accounting for forest cuts. Takes in additive and
        non-additive nodes and a dataset.
        Parameter annual_change is used where the rate node(s) have null values.
        """)),
    'IterativeNode2': NodeInfo(_(
        """
        This is IterativeNode. It calculates one year at a time based on previous year's value and inputs and outputs.
        In addition, it must have a feedback loop (otherwise it makes no sense to use this node class), which is given
        as a growth rate per year from the previous year's value.
        """), deprecated=True),  # FIXME Remove old
    'IterativeNode': NodeInfo(_(
        """
        This is generic IterativeNode for calculating values year by year.
        It calculates one year at a time based on previous year's value and inputs and outputs
        starting from the first forecast year. In addition, it must have a feedback loop (otherwise it makes
        no sense to use this node class), which is given as a growth rate per year from the previous year's value.
        """)),
    'LeverNode': NodeInfo(_(
        """LeverNode replaces the upstream computation completely, if the lever is enabled.""")),
    'LinearCumulativeAdditiveAction': NodeInfo(_(
        """
        Cumulative additive action where a yearly target is set and the effect is linear.
        This can be modified with these parameters:
        target_year_level is the value to be reached at the target year.
        action_delay is the year when the implementation of the action starts.
        multiplier scales the size of the impact (useful between scenarios).
        """)),
    'LogitNode': NodeInfo(_(
        """
        LogitNode gives a probability of event given a baseline and several determinants.

        The baseline is given as a dataset of observed values. The determinants are linearly
        related to the logit of the probability:
        ln(y / (1 - y)) = a + sum<sub>i</sub>(b<sub>i</sub> * X<sub>i</sub>,)
        where y is the probability, a is baseline, X<sub>i</sub> determinants and b<sub>i</sub> coefficients.
        The node expects that a comes from dataset and sum<sub>i</sub>(b<sub>i</sub> * X<sub>i</sub>,) is given by the input nodes
        when operated with the GenericNode compute(). The probability is calculated as
        ln(y / (1 - y)) = b <=> y = 1 / (1 + exp(-b)).
        """)),
    'MultiplicativeNode': NodeInfo(_("")),
    'Population': NodeInfo(_("Population is a specific node about Finnish population."), deprecated=True),
    'ReduceAction': NodeInfo(_("""Define action with parameters <i>reduce</i> and <i>multiplier</i>.""")),
    'SCurveAction': NodeInfo(_(
        """
        This is S Curve Action. It calculates non-linear effect with two parameters,
        max_impact = A and max_year (year when 98 per cent of the impact has occurred).
        The parameters come from Dataset. In addition, there
        must be one input node for background data. Function for
        S-curve y = A/(1+exp(-k*(x-x0)). A is the maximum value, k is the steepness
        of the curve, and x0 is the midpoint year.
        Newton-Raphson method is used to numerically estimate slope and medeian year.
        """)),
    'SectorEmissions': NodeInfo(_("")),
    'ShiftAction': NodeInfo(_("ShiftAction moves activity from one category to others.")),
    'TrajectoryAction': NodeInfo(_(
        """
        TrajectoryAction uses select_category() to select a category from a dimension
        and then possibly do some relative or absolute conversions.
        """)),
    'ValueAction': NodeInfo(_("""
        Value action outputs a constant (e.g. weight for a moral value) over time.
        Adjust the weight parameter to change how much this value contributes to priorities.
        """
    )),
    'WeightedSumNode': NodeInfo(_(
        """
        WeightedSumNode: Combines additive inputs using weights from a multidimensional weights DataFrame.
        """)),
    'Unknown': NodeInfo(_("Node class does not have description."))
}


@dataclass
class GraphRepresentation:
    """Normalized representation of the complete node graph."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)  # node_id -> node_config
    inputs: dict[str, list[str]] = field(default_factory=dict)  # node_id -> list of input_node_ids
    outputs: dict[str, list[str]] = field(default_factory=dict)  # node_id -> list of output_node_ids
    edges: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)  # (from_node, to_node) -> edge_properties


@dataclass
class ValidationResult:
    method: str
    is_valid: bool
    level: Literal['error', 'warning', 'info']
    message: str


class GraphBuilder:
    @staticmethod
    def build_graph(all_node_configs: list[dict[str, Any]]) -> GraphRepresentation:
        """Build normalized graph from all node configs."""

        # Create node_id set for quick lookups
        node_ids = {node['id'] for node in all_node_configs}
        nodes_dict = {node['id']: node for node in all_node_configs}

        inputs: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        outputs: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        edges: dict[tuple[str, str], dict[str, Any]] = {}

        for node in all_node_configs:
            node_id: str = node['id']

            # Handle input_nodes - don't validate existence yet
            for input_spec in node.get('input_nodes', []):
                input_node_id, edge_props = GraphBuilder._parse_edge_spec(input_spec)
                inputs[node_id].append(input_node_id)
                if input_node_id in node_ids:
                    outputs[input_node_id].append(node_id)
                edges[(input_node_id, node_id)] = edge_props

            # Handle output_nodes
            for output_spec in node.get('output_nodes', []):
                output_node_id, edge_props = GraphBuilder._parse_edge_spec(output_spec)
                outputs[node_id].append(output_node_id)
                if output_node_id in node_ids:
                    inputs[output_node_id].append(node_id)

                # Check for duplicate edges
                edge_key = (node_id, output_node_id)
                if edge_key in edges:
                    raise ValueError(f"Duplicate edge definition: {edge_key}")
                edges[edge_key] = edge_props

        return GraphRepresentation(
            nodes=nodes_dict,
            inputs=inputs,
            outputs=outputs,
            edges=edges
        )

    @staticmethod
    def _parse_edge_spec(input_spec) -> tuple[str, dict[str, Any]]:
        """Extract node_id and edge properties from input specification."""
        if isinstance(input_spec, str):
            return input_spec, {}

        if isinstance(input_spec, dict):
            spec_copy = input_spec.copy()
            node_id = spec_copy.pop('id', None)
            if not node_id:
                raise KeyError(f"No node id found in input spec: {input_spec}")
            return node_id, spec_copy

        raise ValueError(f"Invalid input specification: {input_spec}")


@dataclass
class NodeExplanationSystem:

    context: Context

    graph: GraphRepresentation = field(init=False)

    all_node_configs: InitVar[list[dict[str, Any]]]

    explanations: dict[str, list[str]] = field(default_factory=dict)
    """Static explanations generated from node configurations."""

    validations: dict[str, list[ValidationResult]] = field(default_factory=dict)

    baskets: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def __post_init__(self, all_node_configs: list[dict[str, Any]]):
        self.rules = [
            NodeClassRule(),
            DatasetRule(),
            EdgeRule(),
            BasketRule(),
            FormulaDimensionRule(),
            FormulaUnitRule(),
        ]
        self.generate_graph(all_node_configs)

    def generate_graph(self, node_configs: list[dict[str, Any]]) -> NodeExplanationSystem:
        """Validate all nodes with complete graph information."""

        import copy
        all_node_configs = copy.deepcopy(node_configs)
        all_results: dict[str, list[ValidationResult]] = {}

        # Step 1: Build complete graph representation
        try:
            graph = GraphBuilder.build_graph(all_node_configs)
            self.graph = graph
        except KeyError as e:
            # Return graph-level errors for all nodes
            graph_error = ValidationResult(
                method='graph_rule',
                is_valid=False,
                level='error',
                message=str(e)
            )
            all_results = {node['id']: [graph_error] for node in all_node_configs}
            self.validations = all_results # FIXME edge dimensions don't end up here if 'output_nodes' used.(?)
            self.graph = GraphRepresentation()

        return self

    def generate_validations(self) -> dict[str, list[ValidationResult]]:
        """Validate all nodes with complete graph information."""
        all_results: dict[str, list[ValidationResult]] = {}

        # Step 2: Validate each node with complete graph context
        for node_id, node_config in self.graph.nodes.items():

            # Run all validation rules
            node_results = []
            for rule in self.rules:
                if isinstance(node_config, dict):
                    results = rule.validate(node_config, self.context)
                    node_results.extend(results)

            all_results[node_id] = node_results

        self.validations = all_results
        return all_results

    def generate_explanations(self) -> dict[str, list[str]]: # FIXME output_nodes.from_dimension does not show up in explanations
        """Generate explanations for all nodes."""

        all_results = {}
        all_node_configs = self.graph.nodes

        for node_id, node_config in all_node_configs.items():
            # node_id = node_config['id'] if isinstance(node_config, dict) else node_config

            # Run all explanation rules
            node_results = []
            for rule in self.rules:
                if isinstance(node_config, dict):
                    results = rule.explain(node_config, self.context)
                    node_results.extend(results)

            all_results[node_id] = node_results

        self.explanations = all_results
        return all_results

    def generate_input_baskets(self) -> dict[str, dict[str, list[str]]]:
        """Return a dictionary of node 'baskets' categorized by type."""
        baskets: dict[str, dict[str, list[str]]] = {}
        # Special tags that should be skipped completely
        skip_tags = {'ignore_content'}

        for node_id, node_config in self.graph.nodes.items():
            baskets[node_id] = {}

            # Categorize nodes by tags
            assert isinstance(node_config, dict)
            for input_id in self.graph.inputs.get(node_id, []):
                basket = 'unknown'
                input_node = self.graph.nodes[input_id]
                edge_props = self.graph.edges.get((input_id, node_id), {})
                edge_tags = edge_props.get('tags', []) if isinstance(edge_props, dict) else []
                assigned = False
                if 'tags' in input_node:
                    if any(tag in input_node['tags'] or tag in edge_tags for tag in skip_tags): #  or tag in node.tags
                        basket = 'skip'
                        assigned = True
                    else:
                        for tag, basket in TAG_TO_BASKET.items():  # noqa: B007
                            if tag in input_node['tags'] or tag in edge_tags: # TODO or tag in node.tags:
                                assigned = True
                                break

                if not assigned:
                    node_unit = node_config.get('unit') # FIXME Does not Work with multi-metric nodes.
                    input_unit = input_node.get('unit') if isinstance(input_node, dict) else None
                    if node_unit is None or input_unit is None:
                        basket = 'unknown'
                    else:
                        n_dim = unit_registry(node_unit).dimensionality
                        i_dim = unit_registry(input_unit).dimensionality
                        basket = 'add' if n_dim == i_dim else 'multiply'

                if basket not in baskets[node_id]:
                    baskets[node_id][basket] = []
                baskets[node_id][basket].append(input_id)

        self.baskets = baskets
        return baskets

    def has_errors(self) -> bool:
        """Check if any validation results are errors."""
        validation_results = self.validations
        return any(
            any(rule.level == 'error' and not rule.is_valid for rule in node_rules)
            for node_rules in validation_results.values()
        )

    def show_messages(
            self,
            level: Literal['error', 'warning', 'info'] = 'error',
            valid_also: bool = False,
        ) -> dict[str, list[ValidationResult]]:
        """Show all validation results that have messages worse than level."""

        validation_results = self.validations
        severity = {'error': 3, 'warning': 2, 'info': 1}
        min_severity = severity[level]

        messages: dict[str, list[ValidationResult]] = {}
        for node, node_rules in validation_results.items():
            messages[node] = []
            for rule in node_rules:
                if severity[rule.level] >= min_severity and (not rule.is_valid or valid_also):
                    messages[node].append(rule)

        return {node_id: message for node_id, message in messages.items() if len(message) > 0}


class GraphValidator:
    @staticmethod
    def validate_graph(graph: GraphRepresentation) -> list[ValidationResult]:
        """Validate the complete graph for structural issues."""
        results = []

        # Check for missing node references
        for (from_node, to_node) in graph.edges.keys():
            if from_node not in graph.nodes:
                results.append(ValidationResult(
                    method='missing_node_test',
                    is_valid=False,
                    level='error',
                    message=f"Input node reference exists for '{from_node}' but node is missing."
                ))

            if to_node not in graph.nodes:
                results.append(ValidationResult(
                    method='missing_node_test',
                    is_valid=False,
                    level='error',
                    message=f"Output node reference exists for '{from_node}' but node is missing."
                ))

        # Check for circular dependencies
        if GraphValidator._has_cycles(graph):
            results.append(ValidationResult(
                method='cyclic_graph_test',
                is_valid=False,
                level='error',
                message="Graph contains circular dependencies"
            ))

        return results

    @staticmethod
    def _has_cycles(graph: GraphRepresentation) -> bool:
        """Detect cycles by tracking the current visiting path."""
        visited = set()
        visiting = set()  # Currently in the path

        def visit_iterative(start_node: str) -> bool:
            # Use stack to simulate recursion
            stack = [(start_node, 'enter')]

            while stack:
                node_id, action = stack.pop()

                if action == 'enter':
                    if node_id in visiting:
                        return True  # Cycle found!

                    if node_id in visited:
                        continue  # Already processed

                    visiting.add(node_id)
                    # Add exit action first (will be processed after children)
                    stack.append((node_id, 'exit'))

                    # Add children
                    for child_id in graph.outputs.get(node_id, []):
                        if child_id in graph.nodes:  # Only process existing nodes
                            stack.append((child_id, 'enter'))  # noqa: PERF401

                elif action == 'exit':
                    visiting.discard(node_id)
                    visited.add(node_id)

            return False

        # Check all unvisited nodes
        return any(node_id not in visited and visit_iterative(node_id) for node_id in graph.nodes)


class ValidationRule(ABC):
    """Base class for validation rules that also generate explanations."""

    @abstractmethod
    def explain(self, node_config: dict[str, Any], context: Context) -> list[str]:
        """Generate explanation text from node config."""
        pass

    @abstractmethod
    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        """Validate the node configuration."""
        pass

    def get_param(self, node_config: dict[str, Any], param_id: str) -> str:
        if 'params' not in node_config:
            return ''
        params = node_config['params']
        if isinstance(params, dict):
            for id, v in params.items():
                if id == param_id:
                    return v
        else:
            for param in params:
                assert isinstance(param, dict)
                v = param.get('value')
                u = param.get('unit', '')
                if 'id' in param and param['id'] == param_id:
                    return f"{v} {u}"
        return ''

    def get_all_params(self, node_config: dict[str, Any], drop: list[str]) -> list[list[str] | None]:
        out: list[list[str] | None] = []
        if 'params' not in node_config:
            return out
        params = node_config['params']
        if isinstance(params, dict):
            for id, v in params.items():
                if id in drop:
                    continue
                out.append([id, v])
        else: # Assumes list of dicts
            for param in params:
                assert isinstance(param, dict)
                id = param.get('id')
                if id in drop:
                    continue
                assert isinstance(id, str)
                v = param.get('value') or _("referencing to <i>%s</i>") % param.get('ref')
                u = param.get('unit', '')
                out.append([id, f"{v} {u}"])
        return out

class NodeClassRule(ValidationRule):

    def explain(self, node_config: dict[str, Any], context: Context) -> list[str]:
        typ: str = node_config.get('type') or ''
        typ = typ.split('.')[-1]
        html: list[str] = [f"{node_config['id']} ({typ})<br>"]
        desc = NODE_CLASS_DESCRIPTIONS.get(typ) or NODE_CLASS_DESCRIPTIONS['Unknown']
        html.append(f"{desc.description}<ul>")
        operations = self.get_param(node_config, 'operations')
        other = self.get_all_params(node_config, drop = ['operations', 'formula'])
        if operations:
            html.append(f"<li>{_('The order of operations is %s.') % operations}</li>")
        if other:
            for p in other:
                assert p is not None
                text = _('Has parameter <i>%(parameter)s</i> with value %(value)s.') % {'parameter': p[0], 'value': p[1]}
                html.append(f"<li>{text}</li>")

        html.append('</ul>')
        return html

    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        typ = node_config.get('type')
        if isinstance(typ, str):
            typ = typ.split('.')[-1]

            if typ not in NODE_CLASS_DESCRIPTIONS.keys():
                results.append(ValidationResult(
                    method='node_class_rule',
                    is_valid=False,
                    level='warning',
                    message=f'Node class {typ} does not have a description.'
                ))

            elif NODE_CLASS_DESCRIPTIONS[typ].deprecated:
                results.append(ValidationResult(
                    method='node_depreciation_rule',
                    is_valid=False,
                    level='warning',
                    message=f'Node class {typ} is depreciated.'
                ))

        return results

class DatasetRule(ValidationRule):

    def explain(self, node_config: dict[str, Any], context: Context) -> list[str]:
        dataset_html: list[str] = []

        # Terms (including datasets) are handled by BasketRule to keep inputs in one place.
        return dataset_html

        input_datasets: list[dict[str, Any]] = node_config.get('input_datasets', [])

        if not input_datasets:
            return dataset_html

        dataset_html.append(f"{_('Datasets')}:<ul>")

        for dataset_config in input_datasets:
            if isinstance(dataset_config, dict):
                dataset_html.extend(self._explain_single_dataset(dataset_config, context))
            else:
                dataset_html.append(dataset_config)

        dataset_html.append("</ul>")

        return dataset_html

    def _explain_single_dataset(self, dataset_config: dict[str, Any], context: Context) -> list[str]:
        """Explain a single dataset configuration."""
        if isinstance(dataset_config, str):
            return [f"<li><i>{dataset_config}</i></li>"]
        tags: list[Any] = dataset_config.get('tags', [])
        tag_str = ', '.join(tags) + ': ' if tags else ''
        html = [f"<li>{tag_str}{dataset_config['id']}<ul>"]

        col = dataset_config.get('column')
        if col is not None:
            html.append(f'<li>{_("Metric: %(name)s") % {'name': col}}</li>')

        year = dataset_config.get('forecast_from')
        if year is not None:
            html.append(f'<li>{_("Has forecast values from: %(year)s") % {'year': year}}</li>')

        dropna = dataset_config.get('dropna')
        if dropna:
            html.append(f'<li>{_("Rows with missing values are dropped.")}</li>')

        # Handle filters
        filters = dataset_config.get('filters')
        if filters:
            html.extend(self._explain_filters(filters, context))

        html.append('</ul></li>')
        return html

    def _explain_filters(self, filters: list[dict[str, Any]], context: Context) -> list[str]:
        """Explain dataset filters."""
        html = []
        renames = [rename for rename in filters if 'rename_col' in rename]
        if renames:
            html.append(f"<li>{_('Renames the following columns:')}<ul>")
            for d in renames:
                col = d['rename_col']
                val = d.get('value', '')
                html.append(f'<li>{col} &rarr; {val}.</li>')
            html.append('</ul></li>')
        true_filters = [d for d in filters if 'rename_col' not in d]
        if true_filters:
            html.append(f"<li>{_('Has the following filters:')}<ol>")
            for d in true_filters:
                if 'column' in d:
                    html.append(self._explain_column_filter(d, context))
                if 'dimension' in d:
                    html.append(self._explain_dim_filter(d, context))
                if 'rename_item' in d:
                    html.append(self._explain_rename_item_filter(d))

            html.append("</ol></li>")
        return html

    def _explain_column_filter(self, d: dict[str, Any], context: Context) -> str:
        col = d['column']
        v: str = d.get('value', '')
        vals: list[str] = d.get('values', [])
        ref: str = d.get('ref', '')
        if v:
            vals.append(v)
        if ref:
            param = context.global_parameters[ref]
            vals.append(_('global parameter %(label)s') % {'label': param.label})
        drop: bool = d.get('drop_col', True)
        exclude: bool = d.get('exclude', False)
        if ''.join(vals):
            if exclude:
                text = _('Filter column <i>%(name)s</i> by excluding <i>%(values)s</i>.') % {
                    'name': col, 'values': ', '.join(vals)
                }
            else:
                text = _('Filter column <i>%(name)s</i> by including <i>%(values)s</i>.') % {
                    'name': col, 'values': ', '.join(vals)
                }
            out = f"<li>{text}</li>"
        else:
            out = ''
        if  d.get('flatten', False):
            out += f"<li>{_('Sum up column <i>%(name)s</i>.') % {'name': col}}</li>"
        elif drop:
            out += f"<li>{_('Drop column <i>%(name)s</i>.') % {'name': col}}</li>"
        return out

    def _explain_dim_filter(self, d: dict[str, Any], context: Context) -> str:
        dim_id = d['dimension']
        dim = context.dimensions[dim_id]
        if 'assign_category' in d:
            cat_id = d['assign_category']
            dim = context.dimensions[dim_id]
            cat_label = next(str(cat.label) for cat in dim.categories if cat.id == cat_id)
            text = _('Assign dataset to category <i>%(cat_label)s</i> on dimension <i>%(dim_label)s</i>.') % {
                'cat_label': cat_label, 'dim_label': dim.label
            }
            return f"<li>{text}</li>"

        if 'groups' in d:
            grp_ids = d['groups']
            items = [str(group.label) for group in dim.groups if group.id in grp_ids]
        elif 'categories' in d:
            cat_ids = d['categories']
            items = [str(cat.label) for cat in dim.categories if cat.id in cat_ids]
        else:
            items = []
        if items:
            text = _('Filter dimension <i>%s(dim_label)s</i> by categories <i>%(cat_labels)s</i>.') % {
                'dim_label': dim.label, 'cat_labels': ', '.join(items)
            }
            out = f"<li>{text}</li>"
        else:
            out = ''
        if d.get('flatten', False):
            out += f"<li>{_('Sum up the dimension <i>%(label)s</i>.') % {'label': dim.label}}</li>"
        return out

    def _explain_rename_item_filter(self, d: dict[str, Any]) -> str:
        old = d['rename_item'].split('|')
        col = old[0]
        item = old[1]
        new_item = d.get('value', '')
        return _('Rename item <i>%(old_string)s</i> to <i>%(new_string)s</i> in column <i>%(column)s</i>.') % {
            'old_string': item,
            'new_string': new_item,
            'column': col,
        }

    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        input_datasets = node_config.get('input_datasets', [])

        for i, dataset_config in enumerate(input_datasets):
            if isinstance(dataset_config, dict):
                dataset_results = self._validate_single_dataset(dataset_config, i)
                results.extend(dataset_results)

        return results

    def _validate_single_dataset(self, dataset_config: dict[str, Any], index: int) -> list[ValidationResult]:
        """Validate a single dataset configuration."""
        results = []

        # Check for required fields
        if 'id' not in dataset_config:
            results.append(ValidationResult(
                method='dataset_id_check',
                is_valid=False,
                level='error',
                message=f"Dataset {index} is missing required 'id' field"
            ))

        # Validate column specification
        if 'column' in dataset_config:
            column = dataset_config['column']
            if not isinstance(column, str) or not column.strip():
                results.append(ValidationResult(
                    method='dataset_column_check',
                    is_valid=False,
                    level='error',
                    message=f"Dataset {index} is missing column: {column}"
                ))

        # Validate forecast_from year
        if 'forecast_from' in dataset_config:
            year = dataset_config['forecast_from']
            if not isinstance(year, int) or year < 1900 or year > 2100:
                results.append(ValidationResult(
                    method='dataset_forecast_year_check',
                    is_valid=False,
                    level='warning',
                    message=f"Dataset {index} has questionable forecast year: {year}"
                ))

        return results

class EdgeRule(ValidationRule):

    def explain(self, node_config: dict[str, Any], context: Context) -> list[str]:
        # Terms are handled by BasketRule to keep inputs in one place.
        return []

    def get_explanation_for_tag(
        self,
        node: dict[str, Any] | str,
        skip_tags: set[str] | None = None,
    ) -> list[str]:
        html: list[str] = []
        if isinstance(node, str):
            return html

        for tag in node.get('tags', []):
            if skip_tags and tag in skip_tags:
                continue
            if tag in TAG_TO_BASKET.keys(): # These show up in basket explanations
                continue
            description = TAG_DESCRIPTIONS.get(tag, _('Has tag <i>%s</i>.') % tag)
            html.append(f"<li>{description}</li>")
        return html

    def get_explanation_for_edge_from(self, node: dict[str, Any] | str, context: Context) -> list[str]:
        edge_html: list[str] = []
        if isinstance(node, str):
            return edge_html

        for dim in node.get('from_dimensions', []):
            if 'id' not in dim:
                return edge_html
            dimlabel = str(context.dimensions[dim['id']].label)
            cats = dim.get('categories', [])

            if cats:
                category_dict = {cat.id: cat for cat in context.dimensions[dim['id']].categories}
                cats_str = ', '.join([str(category_dict[c].label) for c in cats])
                if dim.get('exclude', False):
                    text = _("From dimension <i>%(dimension)s</i>, exclude categories: <i>%(categories)s</i>") % {
                        'dimension': dimlabel, 'categories': cats_str
                    }
                else:
                    text = _("In dimension <i>%(dimension)s</i>, include categories: <i>%(categories)s</i>") % {
                        'dimension': dimlabel, 'categories': cats_str
                    }
                edge_html.append(f'<li>{text}</li>')

            if dim.get('flatten', False):
                edge_html.append(
                    _("<li>Sum over dimension <i>%(dim)s</i></li>") % {'dim': dimlabel}
                )
        return edge_html

    def get_explanation_for_edge_to(self, node: dict[str, Any] | str, context: Context) -> list[str]:
        edge_html: list[str] = []
        if isinstance(node, str):
            return edge_html

        for dim in node.get('to_dimensions', []):
            if 'id' not in dim:
                return edge_html
            dimlabel = str(context.dimensions[dim['id']].label)
            cats = dim.get('categories', [])

            if cats:
                category_dict = {cat.id: cat for cat in context.dimensions[dim['id']].categories}
                cats_str = ', '.join([str(category_dict[c].label) for c in cats])
                text = _('Categorize the values to <i>%(categories)s</i> in a new dimension <i>%(dimension)s</i>.') % {
                    'categories': cats_str, 'dimension': dimlabel
                }
                edge_html.append(f"<li>{text}</li>")
        return edge_html

    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        return [ValidationResult(
            method='edge_rule',
            is_valid=True,
            level='info',
            message='There is no validation rule for edges.'
        )]


class BasketRule(ValidationRule):

    def explain(self, node_config: dict[str, Any] | str, context: Context) -> list[str]:  # noqa: C901, PLR0912, PLR0915
        assert isinstance(node_config, dict)
        node_id = node_config['id']

        nes = context.node_explanation_system
        assert nes is not None
        baskets = nes.baskets[node_id]
        html: list[str] = []
        operation_list = self.get_param(nes.graph.nodes[node_id], 'operations')
        if not operation_list:
            operation_list = context.nodes[node_id].DEFAULT_OPERATIONS
        operations = [o.strip() for o in operation_list.split(',')]
        terms = self._collect_terms(node_config, context, node_id)
        formula = self._build_formula_from_config(node_config, operations, baskets, terms)
        if not formula and not terms:
            return html
        if formula:
            html.append(f"<p>{_('Formula:')} <b>{formula}</b>,</p>")
        if terms:
            html.append(_("Terms:") + "<ul>")
            for term in terms:
                label = term['label']
                kind = term['kind']
                name = term['name']
                details = term['details']
                suffix_parts: list[str] = []
                if kind == 'constant':
                    value = term.get('value')
                    if value is not None:
                        suffix_parts.append(str(value))
                unit = term.get('unit')
                if unit:
                    suffix_parts.append(str(unit))
                dims = term.get('output_dimensions') or []
                if dims:
                    suffix_parts.append(_('dims: %(dims)s') % {'dims': ', '.join(dims)})
                suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
                if details:
                    html.append(f"<li><b>{label}</b> ({kind}): {name}{suffix}<ul>")
                    html.extend(details)
                    html.append("</ul></li>")
                else:
                    html.append(f"<li><b>{label}</b>: {name}{suffix}</li>")
            html.append("</ul>")

        has_dataset_terms = any(term['kind'] == 'dataset' for term in terms)
        filtered_ops = [
            op for op in operations
            if not (op == 'get_single_dataset' and not has_dataset_terms)
        ]
        functions = self._collect_functions(filtered_ops, terms, node_config)
        if functions:
            html.append(_("Functions:") + "<ul>")
            html.extend(functions)
            html.append("</ul>")

        remaining_baskets = [
            basket for basket in baskets.keys()
            if basket not in operations and basket not in {'skip'}
        ]
        if remaining_baskets:
            html.append(_("These groups are left over without an operation:") + "<ol>")
            for basket in remaining_baskets:
                input_nodes = baskets.get(basket, [])
                basket_display = BASKET_DISPLAY_NAMES.get(basket, basket)
                nodes_str = '</li><li>'.join(input_nodes)
                html.append(f"<li>{_('Group %(basket)s with nodes:') % {'basket': basket_display}}<ul>")
                html.append(f"<li>{nodes_str}</li></ul></li>")
            html.append('</ol>')

        return html

    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        return [ValidationResult(
            method='basket_rule',
            is_valid=True,
            level='info',
            message='There is no validation rule for baskets.'
        )]
        # Each input node must belong to some basket
        # If an input node belongs to the unknown basket, give a warning

    def _collect_terms(  # noqa: C901, PLR0912, PLR0915
        self, # FIXME Somehow does not show edge function list correctly
        node_config: dict[str, Any],
        context: Context,
        node_id: str,
    ) -> list[dict[str, Any]]:
        terms: list[dict[str, Any]] = []
        input_specs = list(node_config.get('input_nodes', []))
        nes = context.node_explanation_system
        assert nes is not None
        merged_specs: list[dict[str, Any]] = []
        index_by_id: dict[str, int] = {}
        for input_spec in input_specs:
            input_id = input_spec if isinstance(input_spec, str) else input_spec.get('id')
            if not input_id:
                continue
            if isinstance(input_spec, dict) and 'ignore_content' in (input_spec.get('tags') or []):
                continue
            spec = {'id': input_id}
            if isinstance(input_spec, dict):
                spec.update(input_spec)
            index_by_id[input_id] = len(merged_specs)
            merged_specs.append(spec)
        for input_id in nes.graph.inputs.get(node_id, []):
            edge_props = nes.graph.edges.get((input_id, node_id), {})
            if 'ignore_content' in (edge_props.get('tags') or []):
                continue
            if input_id in index_by_id:
                spec = merged_specs[index_by_id[input_id]]
                for key, value in edge_props.items():
                    if key not in spec or not spec.get(key):
                        spec[key] = value
                        continue
                    if key == 'tags' and value:
                        tags = list(spec.get('tags', []))
                        for tag in value:
                            if tag not in tags:
                                tags.append(tag)
                        spec['tags'] = tags
                continue
            merged_specs.append({'id': input_id, **edge_props})
            index_by_id[input_id] = len(merged_specs) - 1
        input_specs = merged_specs
        input_datasets = node_config.get('input_datasets', [])
        params = node_config.get('params', [])

        for input_spec in input_specs:
            input_id = input_spec if isinstance(input_spec, str) else input_spec.get('id')
            if not input_id:
                continue
            input_node = context.node_explanation_system.graph.nodes.get(input_id, {})  # type: ignore[union-attr]
            label_tag = None
            func_tags: list[str] = []
            if isinstance(input_spec, dict):
                func_tags = [
                    tag for tag in input_spec.get('tags', [])
                    if tag in TAG_DESCRIPTIONS and tag not in TAG_TO_BASKET
                ]
                label_tag = next(
                    (
                        tag for tag in input_spec.get('tags', [])
                        if tag not in TAG_TO_BASKET and tag not in TAG_DESCRIPTIONS
                    ),
                    None,
                )
            if not func_tags or label_tag is None:
                node_tags = input_node.get('tags', []) if isinstance(input_node, dict) else []
                if not func_tags:
                    func_tags = [
                        tag for tag in node_tags
                        if tag in TAG_DESCRIPTIONS and tag not in TAG_TO_BASKET
                    ]
                if label_tag is None:
                    label_tag = next(
                        (
                            tag for tag in node_tags
                            if tag not in TAG_TO_BASKET and tag not in TAG_DESCRIPTIONS
                        ),
                        None,
                    )
            output_dimensions = input_node.get('output_dimensions')
            adjusted_dims: list[Any] | None = None
            from_dims = input_spec.get('from_dimensions', [])
            to_dims = input_spec.get('to_dimensions', [])
            if output_dimensions is not None or from_dims or to_dims:
                adjusted_dims = []
                dim_ids = []
                for dim in output_dimensions or []:
                    if isinstance(dim, dict):
                        dim_id = dim.get('id')
                    else:
                        dim_id = dim
                    if isinstance(dim_id, str):
                        dim_ids.append(dim_id)
                for dim in from_dims or []:
                    dim_id = dim.get('id') if isinstance(dim, dict) else dim
                    if not isinstance(dim_id, str):
                        continue
                    if isinstance(dim, dict) and dim.get('flatten'):
                        if dim_id in dim_ids:
                            dim_ids.remove(dim_id)
                        continue
                    if dim_id not in dim_ids:
                        dim_ids.append(dim_id)
                for dim in to_dims or []:
                    dim_id = dim.get('id') if isinstance(dim, dict) else dim
                    if isinstance(dim_id, str) and dim_id not in dim_ids:
                        dim_ids.append(dim_id)
                for dim_id in dim_ids:
                    adjusted_dims.append(dim_id)

            term = {
                'kind': 'node',
                'key': input_id,
                'label': label_tag,
                'name': context.nodes[input_id].name,
                'var_names': self._term_var_names(input_spec, input_id),
                'unit': input_node.get('unit'),
                'output_dimensions': adjusted_dims if adjusted_dims is not None else output_dimensions,
                'functions': func_tags,
                'details': self._node_term_details(input_spec, context, label_tag),
            }
            terms.append(term)

        for dataset_config in input_datasets:
            if not isinstance(dataset_config, dict):
                continue
            ds_id = dataset_config.get('id')
            if not ds_id:
                continue
            ds_output_dimensions = dataset_config.get('output_dimensions')
            if ds_output_dimensions is None:
                ds_output_dimensions = node_config.get('output_dimensions')
            ds_unit = dataset_config.get('unit')
            if ds_unit is None:
                ds_unit = node_config.get('unit')
            tags = [tag for tag in dataset_config.get('tags', []) if tag != 'cleaned']
            label_tag = tags[0] if tags else None
            term = {
                'kind': 'dataset',
                'key': ds_id,
                'label': label_tag,
                'name': ds_id,
                'var_names': self._dataset_var_names(dataset_config, ds_id),
                'unit': ds_unit,
                'output_dimensions': ds_output_dimensions,
                'functions': [],
                'details': self._dataset_term_details(dataset_config, context),
            }
            terms.append(term)

        if isinstance(params, dict):
            params = [dict(id=param_id, value=value) for param_id, value in params.items()]
        for param in params:
            if not isinstance(param, dict):
                continue
            param_id = param.get('id')
            if param_id in ['formula', 'operations']:
                continue
            value = param.get('value')
            unit = param.get('unit', '')
            if param_id and value is not None:
                term = {
                    'kind': 'constant',
                    'key': param_id,
                    'label': param_id,
                    'name': _('Constant'),
                    'var_names': [param_id],
                    'unit': unit or None,
                    'value': value,
                    'functions': [],
                    'details': [],
                }
                terms.append(term)

        counter = 1
        for term in terms:
            if not term['label']:
                term['label'] = f"t{counter}"
                counter += 1

        formula_param = self.get_param(node_config, 'formula')
        if formula_param:
            formula_param = self._apply_term_functions(formula_param, terms)
            used_names = self._extract_formula_identifiers(formula_param)
            for term in terms:
                if term['kind'] != 'node':
                    continue
                if term['key'] in used_names and str(term['label']).startswith('t'):
                    term['label'] = term['key']

        return terms

    def _collect_functions(  # noqa: C901
        self,
        operations: list[str],
        terms: list[dict[str, Any]],
        node_config: dict[str, Any],
    ) -> list[str]:
        functions: list[str] = []
        ops_seen: set[str] = set()
        formula_param = self.get_param(node_config, 'formula')
        if formula_param:
            for func in self._extract_formula_functions(formula_param):
                if func in ops_seen:
                    continue
                description = TAG_DESCRIPTIONS.get(func)
                if description:
                    functions.append(f"<li><b>{func}</b>: {description}</li>")
                    ops_seen.add(func)
        for op in operations:
            if op in ops_seen:
                continue
            description = TAG_DESCRIPTIONS.get(op)
            if description:
                functions.append(f"<li><b>{op}</b>: {description}</li>")
                ops_seen.add(op)
        for term in terms:
            for func in term.get('functions', []):
                if func in ops_seen:
                    continue
                description = TAG_DESCRIPTIONS.get(func)
                if description:
                    functions.append(f"<li><b>{func}</b>: {description}</li>")
                    ops_seen.add(func)
        return functions

    def _extract_formula_identifiers(self, formula: str) -> set[str]:
        import ast

        try:
            tree = ast.parse(formula, "<string>", mode="eval")
        except SyntaxError:
            return set()
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _extract_formula_functions(self, formula: str) -> set[str]:
        import ast

        try:
            tree = ast.parse(formula, "<string>", mode="eval")
        except SyntaxError:
            return set()
        functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                functions.add(node.func.id)
        return functions

    def _apply_term_functions(self, formula: str, terms: list[dict[str, Any]]) -> str:
        import re

        updated = formula
        for term in terms:
            funcs = term.get('functions', [])
            if not funcs:
                continue
            for var in term.get('var_names', []) or []:
                for func in funcs:
                    wrapped = f"{func}({var})"
                    if wrapped in updated:
                        continue
                    updated = re.sub(rf"\\b{re.escape(var)}\\b", wrapped, updated)
        return updated

    def _term_var_names(self, input_spec: dict[str, Any] | str, input_id: str) -> list[str]:
        if isinstance(input_spec, str):
            return [input_id]
        tags = [
            tag for tag in input_spec.get('tags', [])
            if tag not in TAG_TO_BASKET and tag not in TAG_DESCRIPTIONS
        ]
        if tags:
            return tags
        return [input_id]

    def _dataset_var_names(self, dataset_config: dict[str, Any], ds_id: str) -> list[str]:
        tags = [tag for tag in dataset_config.get('tags', []) if tag != 'cleaned']
        if tags:
            return tags
        return [ds_id]

    def _build_formula_from_config(  # noqa: C901
        self,
        node_config: dict[str, Any],
        operations: list[str],
        baskets: dict[str, list[str]],
        terms: list[dict[str, Any]],
    ) -> str:
        formula_param = self.get_param(node_config, 'formula')
        label_by_id: dict[str, str] = {}
        for term in terms:
            label = make_identifier(term['key'])
            for func in term.get('functions', []):
                label = f"{func}({label})"
            label_by_id[term['key']] = label
        if formula_param:
            formula_param = self._apply_term_functions(formula_param, terms)
            used_names = self._extract_formula_identifiers(formula_param)
            unused_labels = []
            for term in terms:
                if term['kind'] != 'node':
                    continue
                var_names = set(term.get('var_names', []))
                if not var_names or var_names.isdisjoint(used_names):
                    unused_labels.append(label_by_id[term['key']])
            if unused_labels:
                return f"({formula_param} + {' + '.join(unused_labels)})"
            return formula_param
        typ = node_config.get('type') or ''
        if isinstance(typ, str):
            typ = typ.split('.')[-1]
        if typ == 'AdditiveNode':
            add_terms = [term['label'] for term in terms if term['kind'] != 'constant']
            if not add_terms:
                return ''
            return f"({BASKET_OPERATION_LABEL['add'].join(add_terms)})"
        if not operations:
            return ''
        return self._build_formula_from_operations(
            operations,
            baskets,
            label_by_id,
            terms,
            has_dataset_terms=any(term['kind'] == 'dataset' for term in terms),
        )

    def _build_formula_from_operations(  # noqa: C901
        self,
        operations: list[str],
        baskets: dict[str, list[str]],
        label_by_id: dict[str, str],
        terms: list[dict[str, Any]],
        has_dataset_terms: bool,
    ) -> str:
        expr = ''
        fallback_term = next(
            (term['label'] for term in terms if term['kind'] == 'dataset'),
            '',
        )
        if not fallback_term:
            fallback_term = next(
                (term['label'] for term in terms if term['kind'] != 'constant'),
                '',
            )
        multiplier = label_by_id.get('multiplier')
        for operation in operations:
            input_nodes = baskets.get(operation, [])
            if input_nodes:
                seen: set[str] = set()
                deduped = []
                for n in input_nodes:
                    if n in seen:
                        continue
                    seen.add(n)
                    deduped.append(n)
                input_nodes = deduped
            if input_nodes:
                op_terms = [label_by_id[n] for n in input_nodes if n in label_by_id]
                if not op_terms:
                    continue
                if expr and expr not in op_terms:
                    op_terms = [expr, *op_terms]
                expr = self._render_operation(operation, op_terms)
                continue

            if operation == 'get_single_dataset' and has_dataset_terms and not expr and fallback_term:
                expr = fallback_term
            elif operation == 'apply_multiplier' and expr and multiplier:
                expr = self._render_operation('multiply', [expr, multiplier])

        if not expr and fallback_term:
            expr = fallback_term
        return expr

    def _build_formula_from_operations_for_validation(  # noqa: C901
        self,
        operations: list[str],
        baskets: dict[str, list[str]],
        label_by_id: dict[str, str],
        terms: list[dict[str, Any]],
        has_dataset_terms: bool,
    ) -> str:
        def _render_operation(operation: str, op_terms: list[str]) -> str:
            op_label = BASKET_OPERATION_LABEL.get(operation, ' + ')
            if operation in ['add', 'multiply']:
                return f"({op_label.join(op_terms)})"
            return f"{operation}({op_label.join(op_terms)})"

        expr = ''
        fallback_term = next(
            (term['label'] for term in terms if term['kind'] == 'dataset'),
            '',
        )
        if not fallback_term:
            fallback_term = next(
                (term['label'] for term in terms if term['kind'] != 'constant'),
                '',
            )
        multiplier = label_by_id.get('multiplier')
        for operation in operations:
            input_nodes = baskets.get(operation, [])
            if input_nodes:
                seen: set[str] = set()
                deduped = []
                for n in input_nodes:
                    if n in seen:
                        continue
                    seen.add(n)
                    deduped.append(n)
                input_nodes = deduped
            if input_nodes:
                op_terms = [label_by_id[n] for n in input_nodes if n in label_by_id]
                if not op_terms:
                    continue
                if expr and expr not in op_terms:
                    op_terms = [expr, *op_terms]
                expr = _render_operation(operation, op_terms)
                continue

            if operation == 'get_single_dataset' and has_dataset_terms and not expr and fallback_term:
                expr = fallback_term
            elif operation == 'apply_multiplier' and expr and multiplier:
                expr = _render_operation('multiply', [expr, multiplier])

        if not expr and fallback_term:
            expr = fallback_term
        return expr

    def _render_operation(self, operation: str, terms: list[str]) -> str:
        op_label = BASKET_OPERATION_LABEL.get(operation, ' + ')
        no_name = ['add', 'multiply']
        op_name = '' if operation in no_name else BASKET_DISPLAY_NAMES.get(operation, operation)
        joined = op_label.join(terms)
        if op_name:
            return f"{op_name}({joined})"
        return f"({joined})"

    def _node_term_details(
        self,
        input_spec: dict[str, Any] | str,
        context: Context,
        label_tag: str | None,
    ) -> list[str]:
        if isinstance(input_spec, str):
            return []
        details: list[str] = []
        metrics = input_spec.get('metrics', [])
        if metrics:
            metrics_str = ', '.join(metrics)
            details.append(f"<li>{_('Metrics: %(metrics)s') % {'metrics': metrics_str}}</li>")
        func_tags = [
            tag for tag in input_spec.get('tags', [])
            if tag in TAG_DESCRIPTIONS and tag not in TAG_TO_BASKET
        ]
        details.extend(
            EdgeRule().get_explanation_for_tag(
                input_spec,
                skip_tags=set(filter(None, [label_tag, *func_tags])) or None,
            )
        )
        details.extend(EdgeRule().get_explanation_for_edge_from(input_spec, context))
        details.extend(EdgeRule().get_explanation_for_edge_to(input_spec, context))
        return details

    def _dataset_term_details(self, dataset_config: dict[str, Any], context: Context) -> list[str]:
        html: list[str] = []
        col = dataset_config.get('column')
        if col is not None:
            text = _("Metric: %(name)s") % {'name': col}
            html.append(f"<li>{text}</li>")
        year = dataset_config.get('forecast_from')
        if year is not None:
            text = _("Has forecast values from: %(year)s") % {'year': year}
            html.append(f"<li>{text}</li>")
        dropna = dataset_config.get('dropna')
        if dropna:
            html.append(f'<li>{_("Rows with missing values are dropped.")}</li>')
        filters = dataset_config.get('filters')
        if filters:
            html.extend(DatasetRule()._explain_filters(filters, context))
        return html


class FormulaValidationMixin(ValidationRule, ABC):
    def explain(self, _node_config: dict[str, Any], _context: Context) -> list[str]:
        return []

    def _ensure_baskets(self, context: Context) -> GraphRepresentation:
        nes = context.node_explanation_system
        assert nes is not None
        if not nes.baskets:
            nes.generate_input_baskets()
        return nes.graph

    def _get_operations(
        self,
        node_id: str,
        context: Context,
        graph: GraphRepresentation,
    ) -> list[str]:
        operation_list = self.get_param(graph.nodes[node_id], 'operations')
        if not operation_list:
            operation_list = context.nodes[node_id].DEFAULT_OPERATIONS
        return [o.strip() for o in operation_list.split(',') if o.strip()]

    def _build_formula_spec(  # noqa: PLR0912, C901
        self,
        node_config: dict[str, Any],
        context: Context,
        operations: list[str],
    ) -> FormulaSpec:
        node_id = node_config['id']
        nes = context.node_explanation_system
        assert nes is not None

        basket_rule = BasketRule()
        terms = basket_rule._collect_terms(node_config, context, node_id)
        baskets = nes.baskets.get(node_id, {})

        display_expression = basket_rule._build_formula_from_config(
            node_config,
            operations,
            baskets,
            terms,
        )

        formula_param = self.get_param(node_config, 'formula')
        label_by_id: dict[str, str] = {}
        for term in terms:
            label = term['key']
            for func in term.get('functions', []):
                label = f"{func}({label})"
            label_by_id[term['key']] = label

        if formula_param:
            formula_param = basket_rule._apply_term_functions(formula_param, terms)
            formula_param = normalize_formula_identifiers(
                formula_param,
                collect_term_names(terms),
            )
            used_names = basket_rule._extract_formula_identifiers(formula_param)
            unused_labels = []
            for term in terms:
                if term['kind'] != 'node':
                    continue
                var_names = set(term.get('var_names', []))
                if not var_names or var_names.isdisjoint(used_names):
                    unused_labels.append(label_by_id[term['key']])
            if unused_labels:
                expression = f"({formula_param} + {' + '.join(unused_labels)})"
            else:
                expression = formula_param
        else:
            typ = node_config.get('type') or ''
            if isinstance(typ, str):
                typ = typ.split('.')[-1]
            if typ == 'AdditiveNode':
                add_terms = [
                    label_by_id[term['key']]
                    for term in terms
                    if term['kind'] != 'constant' and term['key'] in label_by_id
                ]
                expression = f"({BASKET_OPERATION_LABEL['add'].join(add_terms)})" if add_terms else ''
            elif not operations:
                expression = ''
            else:
                expression = basket_rule._build_formula_from_operations_for_validation(
                    operations,
                    baskets,
                    label_by_id,
                    terms,
                    has_dataset_terms=any(term['kind'] == 'dataset' for term in terms),
                )

        if expression:
            expression = normalize_formula_identifiers(
                expression,
                collect_term_names(terms),
            )

        return FormulaSpec(
            expression=expression,
            display_expression=display_expression,
            terms=terms,
        )

    @staticmethod
    def _normalize_dims(dims: list[Any] | None) -> set[str]:
        out: set[str] = set()
        for dim in dims or []:
            if isinstance(dim, dict) and isinstance(dim_id := dim.get('id'), str) and dim_id:
                out.add(dim_id)
            elif isinstance(dim, str):
                out.add(dim)
        return out


class FormulaDimensionRule(FormulaValidationMixin):
    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        node_id = node_config['id']
        nes = context.node_explanation_system
        if nes is None:
            return results
        graph = self._ensure_baskets(context)
        operations = self._get_operations(node_id, context, graph)
        spec = self._build_formula_spec(node_config, context, operations)
        if not spec.expression:
            return results

        results.extend(self._validate_dataset_dims(node_config, spec))
        results.extend(self._validate_formula_dims(spec))
        results.extend(self._validate_output_dims(node_config, spec))
        return results

    def _validate_dataset_dims(
        self,
        node_config: dict[str, Any],
        spec: FormulaSpec,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        used_names = BasketRule()._extract_formula_identifiers(spec.expression)
        node_output_dimensions = node_config.get('output_dimensions')
        for term in spec.terms:
            if term.get('kind') != 'dataset':
                continue
            if term.get('output_dimensions') is not None:
                continue
            if node_output_dimensions is not None:
                continue
            term_names = set(term.get('var_names', []) or [])
            for field_name in ('label', 'key'):
                val = term.get(field_name)
                if isinstance(val, str) and val:
                    term_names.add(val)
            if not used_names or term_names.intersection(used_names):
                label = term.get('label') or term.get('key') or term.get('name')
                label_str = f" '{label}'" if label else ""
                results.append(ValidationResult(
                    method='formula_dimension_rule',
                    is_valid=False,
                    level='info',
                    message=(
                        f"Dataset term{label_str} is missing output_dimensions; "
                        "dimension validation may be incomplete."
                    ),
                ))
        return results

    def _validate_formula_dims(self, spec: FormulaSpec) -> list[ValidationResult]:
        analysis = analyze_formula_dimensions(
            spec.expression,
            build_name_dimension_map(spec.terms)[0],
            passthrough_functions=set(TAG_DESCRIPTIONS.keys()),
        )
        results: list[ValidationResult] = []
        results.extend([
            ValidationResult(
                method='formula_dimension_rule',
                is_valid=False,
                level='error',
                message=message,
            )
            for message in analysis.errors
        ])
        results.extend([
            ValidationResult(
                method='formula_dimension_rule',
                is_valid=True,
                level='warning',
                message=message,
            )
            for message in analysis.warnings
        ])
        return results

    def _validate_output_dims(
        self,
        node_config: dict[str, Any],
        spec: FormulaSpec,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        expected_dims = self._normalize_dims(node_config.get('output_dimensions'))
        analysis = analyze_formula_dimensions(
            spec.expression,
            build_name_dimension_map(spec.terms)[0],
            passthrough_functions=set(TAG_DESCRIPTIONS.keys()),
        )
        if analysis.dims is not None and expected_dims and analysis.dims != expected_dims:
            results.append(ValidationResult(
                method='formula_dimension_rule',
                is_valid=False,
                level='error',
                message=(
                    "Formula output dimensions do not match node.output_dimensions: "
                    f"{sorted(analysis.dims)} vs {sorted(expected_dims)}. "
                    f"Expression: {spec.expression}"
                ),
            ))
        return results


class FormulaUnitRule(FormulaValidationMixin):
    def validate(self, node_config: dict[str, Any], context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        node_unit = node_config.get('unit')
        if not node_unit:
            return results
        node_id = node_config['id']
        nes = context.node_explanation_system
        if nes is None:
            return results
        graph = self._ensure_baskets(context)
        operations = self._get_operations(node_id, context, graph)
        spec = self._build_formula_spec(node_config, context, operations)
        if not spec.expression:
            return results

        name_units = build_name_unit_map(spec.terms)
        results.extend(self._apply_multiplier_unit_inference(node_config, operations, spec, name_units))

        unit_analysis = analyze_formula_units(
            spec.expression,
            name_units,
            passthrough_functions=set(TAG_DESCRIPTIONS.keys()),
            unit_overrides=cast("dict[str, UnitOverride]", FORMULA_FUNCTION_UNIT_OVERRIDES),
        )
        results.extend([
            ValidationResult(
                method='formula_unit_rule',
                is_valid=False,
                level='error',
                message=message,
            )
            for message in unit_analysis.errors
        ])
        results.extend([
            ValidationResult(
                method='formula_unit_rule',
                is_valid=True,
                level='warning',
                message=message,
            )
            for message in unit_analysis.warnings
        ])

        expected_unit = unit_registry.parse_units(node_unit)
        if unit_analysis.unit is not None:
            if unit_analysis.unit.dimensionality != expected_unit.dimensionality:
                results.append(ValidationResult(
                    method='formula_unit_rule',
                    is_valid=False,
                    level='error',
                    message=(
                        "Formula output unit does not match node.unit: "
                        f"{unit_analysis.unit} vs {expected_unit}"
                    ),
                ))
            elif unit_analysis.unit != expected_unit:
                results.append(ValidationResult(
                    method='formula_unit_rule',
                    is_valid=True,
                    level='info',
                    message=(
                        "Formula output unit differs from node.unit but is compatible: "
                        f"{unit_analysis.unit} vs {expected_unit}"
                    ),
                ))
        return results

    def _apply_multiplier_unit_inference(  # noqa: C901, PLR0912
        self,
        node_config: dict[str, Any],
        operations: list[str],
        spec: FormulaSpec,
        name_units: dict[str, Unit | None],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        params = node_config.get('params', [])
        multiplier_unit = None
        if isinstance(params, dict):
            multiplier_unit = params.get('multiplier_unit')
        else:
            for param in params:
                if isinstance(param, dict) and param.get('id') == 'multiplier':
                    multiplier_unit = param.get('unit')
                    break
        if not multiplier_unit or 'apply_multiplier' not in operations:
            return results

        explicit_ds_units: set[str] = set()
        for ds in node_config.get('input_datasets', []) or []:
            if isinstance(ds, dict) and ds.get('id') and ds.get('unit') is not None:
                explicit_ds_units.add(ds['id'])
        inferred_unit = cast("Unit", unit_registry.parse_units(node_config['unit']) / unit_registry.parse_units(multiplier_unit))
        used_multiplier_inference = False
        for term in spec.terms:
            if term.get('kind') != 'dataset':
                continue
            if term.get('key') in explicit_ds_units:
                continue
            term_names = set(term.get('var_names', []) or [])
            for field_name in ('label', 'key'):
                val = term.get(field_name)
                if isinstance(val, str) and val:
                    term_names.add(val)
            for name in term_names:
                name_units[name] = inferred_unit
                name_units[make_identifier(name)] = inferred_unit
            used_multiplier_inference = True
        if used_multiplier_inference:
            results.append(ValidationResult(
                method='formula_unit_rule',
                is_valid=True,
                level='warning',
                message=(
                    "Dataset unit inferred from node.unit and multiplier; "
                    "consider setting dataset.unit explicitly."
                ),
            ))
        return results
