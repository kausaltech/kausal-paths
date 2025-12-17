from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from common.i18n import gettext as _

from .units import unit_registry

if TYPE_CHECKING:
    from nodes.context import Context

TAG_TO_BASKET = {
    'additive': 'add', # FIXME Values should be translatable because they are used in explanations
    'add_from_incoming_dims': 'add_from_incoming_dims',
    'add_to_existing_dims': 'add_to_existing_dims',
    'base': 'other',
    # These operations only handle existing df and do not take new inputs:
    # apply_multipliet, do_correction, drop_nand, drop_infs, extend_values, extrapolate
    # [get_datasets: generalize to use the port theory]
    # inventory_only, other, select_variant
    'non_additive': 'multiply',
    'other_node': 'other',
    'primary': 'coalesce',
    'rate': 'other',
    'secondary': 'coalesce',
    'skip_dim_test': 'skip_dim_test',
    'split_by_existing_shares': 'split_by_existing_shares',
    'split_evenly_to_cats': 'split_evenly_to_cats',
    'threshold': 'apply_threshold',
    'use_as_shares': 'use_as_shares',
    'use_as_totals': 'use_as_totals',
}

BASKET_DISPLAY_NAMES = {
    'add': _('addition'),
    'add_from_incoming_dims': _('addition from incoming dimensions'),
    'add_to_existing_dims': _('addition to existing dimensions'),
    'coalesce': _('coalescing'),
    'multiply': _('multiplication'),
    'other': _('other operations'),
    'skip_dim_test': _('skip dimension test'),
    'split_by_existing_shares': _('split by existing shares'),
    'split_evenly_to_cats': _('split evenly to categories'),
    'apply_threshold': _('apply threshold'),
    'use_as_shares': _('use as shares'),
    'use_as_totals': _('use as totals'),
    'unknown': _('unknown operation'),
    'skip': _('skip'),
}

TAG_DESCRIPTIONS = {
    'additive': _("Add input node values (even if the units don't match with the node units)."),
    'arithmetic_inverse': _('Take the arithmetic inverse of the values (-x).'),
    'bring_to_maximum_historical_year': _('Makes all years up to maximum historical year non-forecasts.'),
    'complement': _('Take the complement of the dimensionless values (1-x).'),
    'complement_cumulative_product': _('Take the cumulative product of the dimensionless complement values over time.'),
    'cumulative': _('Take the cumulative sum over time.'),
    'cumulative_product': _('Take the cumulative product of the dimensionless values over time.'),
    'difference': _('Take the difference over time (i.e. annual changes)'),
    'empty_to_zero': _('Convert NaNs to zeros.'),
    'expectation': _('Take the expected value over the uncertainty dimension.'),
    'extend_forecast_values': _('Extend the last forecast values to the remaining missing years.'),
    'extend_to_history': _('Extend the first values to the years after the minimum historical year.'),
    'extend_values': _('Extend the last historical values to the remaining missing years.'),
    'geometric_inverse': _('Take the geometric inverse of the values (1/x).'),
    'goal': _('The node is used as the goal for the action.'),
    'historical': _('The node is used as the historical starting point.'),
    'existing': _('This is used as the baseline.'),
    'incoming': _('This is used for the incoming stock.'),
    'ignore_content': _('Show edge on graphs but ignore upstream content.'),
    'inserting': _('This is the rate of new stock coming in.'),
    'inventory_only': _('Truncate the forecast values.'),
    'make_nonnegative': _('Negative result values are replaced with 0.'),
    'make_nonpositive': _('Positive result values are replaced with 0.'),
    'non_additive': _('Input node values are not added but operated despite matching units.'),
    'primary': _('Use data as primary values even if a secondary value exists.'),
    'ratio_to_last_historical_value': _('Take the ratio of the values compared with the last historical value.'),
    'removing': _('This is the rate of stock removal.'),
    'secondary': _('Use data only if a primary value does not exist.'),
    'truncate_before_start': _('Truncate values before the reference year. There may be some from data'),
    'truncate_beyond_end': _('Truncate values beyond the model end year. There may be some from data'),
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
    'AssociationNode': NodeInfo(_(
        """
        Association nodes connect to their upstream nodes in a loose way:
        Their values follow the relative changes of the input nodes but
        their quantities and units are not dependent on those of the input nodes.
        The node MUST have exactly one dataset, which is the prior estimate.
        Fractions 1..3 can be used to tell how much the input node should adjust
        the output node. The default relation is "increase", if "decrease" is used,
        that must be explicitly said in the tags.
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
        <br/>a_i = z_i * f_i / sum_j(z_j * f_j),
        <br/>where a_i is the fraction for each product (i = electricity, heat)
        z_i is a method-specific multiplier (see below)
        f_i is the fraction of product i from the total energy produced.

        <ol><li><b>Energy method</b>
        Logic: Energy products are treated equally based on the energy content.
        All z_i = 1

        </li><li><b>Work potential method</b> (aka Carnot method, or exergetic method)
        Logic: Energy products are treated equally based on the potential of doing work (i.e., exergy content).
        This moves emissions toward electricity.
        z_heat = 1 - T_return / T_supply, and
        z_electricity = 1.
        T_supply and T_return are the input and output process temperatures, respectively.

        </li><li><b>Bisko method</b>
        Bisko method is a variant of the work potential method.
        The only difference is that Bisko assumes T_return = 283 K.

        </li><li><b>Efficiency method</b>
        Logic: What emissions would have occured if each energy product had been produced separately?
        z_i = 1 / n_i,
        where n_i is the reference efficiency for producing the energy type separately.
        Typical values are z_heat = 0.9, z_electricity = 0.4.</li></ol>
        """)),
    'CoalesceNode': NodeInfo(_(
        "Uses 'primary' tagged data when available, otherwise 'secondary' tagged data. One of the tags must be given.")),
    'CohortNode': NodeInfo(_(
        """
        Cohort node takes in initial age structure (inventory) and follows the cohort in time as it ages.

        Harvest describes how much is removed from the cohort.
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
    'FormulaNode': NodeInfo(_('This is a Formula Node. It uses a specified formula to calculate the output.')),
    'GenerationCapacityNode': NodeInfo(_("""
        Calculates generation of energy when new capacity is installed. Includes scope 3 emissions from installation,
        and emissions avoided from the capacity that gets replaced.
        """)),
    'GenericNode': NodeInfo(_("")),
    'GenericAction': NodeInfo(_("Action with GenericNode properties.")),
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
    'LogicalNode': NodeInfo(_(
        """
        LogicalNode processes logical inputs (values 0 or 1).

        It applies Boolean AND to multiplicative nodes (nodes are ANDed together)
        and Boolean OR to additive nodes (nodes are ORed together).

        AND operations are performed first, then OR operations. For more complex
        logical structures, use several subsequent nodes.
        """), deprecated=True),  # FIXME There are several versions. Remove redundant.
    'LogitNode': NodeInfo(_(
        """
        LogitNode gives a probability of event given a baseline and several determinants.

        The baseline is given as a dataset of observed values. The determinants are linearly
        related to the logit of the probability:
        ln(y / (1 - y)) = a + sum_i(b_i * X_i,)
        where y is the probability, a is baseline, X_i determinants and b_i coefficients.
        The node expects that a comes from dataset and sum_i(b_i * X_i,) is given by the input nodes
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
    'ThresholdNode': NodeInfo(_(
        """
        ThresholdNode computes a preliminary result using standard GenericNode operations.

        After computation, it returns True (1) if the result is greater than or equal to
        the threshold parameter, otherwise False (0).
        """), deprecated=True),
    'TrajectoryAction': NodeInfo(_(
        """
        TrajectoryAction uses select_category() to select a category from a dimension
        and then possibly do some relative or absolute conversions.
        """)),
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

    validations: dict[str, list[ValidationResult]] = field(default_factory=dict)

    baskets: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def __post_init__(self, all_node_configs: list[dict[str, Any]]):
        self.rules = [
            NodeClassRule(),
            DatasetRule(),
            EdgeRule(),
            BasketRule(),
            # CategoryRetentionRule(),
            # OperationBasketRule(),
            # UnitCompatibilityRule(),
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
                assigned = False
                if 'tags' in input_node:
                    if any(tag in input_node['tags'] for tag in skip_tags): #  or tag in node.tags
                        basket = 'skip'
                        assigned = True
                    else:
                        for tag, basket in TAG_TO_BASKET.items():  # noqa: B007
                            if tag in input_node['tags']: # TODO or tag in node.tags:
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
                v = param.get('value')
                u = param.get('unit', '')
                if id in drop:
                    continue
                assert isinstance(id, str)
                out.append([id, f"{v} {u}"])
        return out

class NodeClassRule(ValidationRule):

    def explain(self, node_config: dict[str, Any], context: Context) -> list[str]:
        html: list[str] = [f"{node_config['id']}<br>"]

        typ = node_config.get('type')
        if isinstance(typ, str):
            typ = typ.split('.')[-1]
            desc = NODE_CLASS_DESCRIPTIONS.get(typ)
            if desc is None:
                desc = NODE_CLASS_DESCRIPTIONS['Unknown']
            html.append(f"{desc.description}<ul>")
        operations = self.get_param(node_config, 'operations')
        formula = self.get_param(node_config, 'formula')
        other = self.get_all_params(node_config, drop = ['operations', 'formula'])
        if operations:
            html.append(f"<li>{_('The order of operations is %s.') % operations}</li>")
        if formula:
            html.append(f"<li>{_('Has formula %s.') % formula}</li>")
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

    def _explain_single_dataset(self, dataset_config: dict[str, None], context: Context) -> list[str]:
        """Explain a single dataset configuration."""
        if isinstance(dataset_config, str):
            return [f"<li>{_('Dataset with identifier <i>%s</i>') % dataset_config}</li>"]
        html = [f"<li>{_('Dataset with identifier <i>%s</i>') % dataset_config['id']}<ul>"]

        col = dataset_config.get('column')
        if col is not None:
            html.append(f'<li>{_("Uses column: %(name)s") % {'name': col}}</li>')

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


# class DimensionCompatibilityRule(ValidationRule): # FIXME SUggestions by CLaude. Rewrite when you take these into use.
#     def explain(self, config: dict) -> str:
#         operation = config.get('operation', 'unknown')
#         input_dims = config.get('input_dimensions', [])
#         output_dims = config.get('output_dimensions', [])

#         if operation == 'addition':
#             return f"Addition requires identical dimensions: {input_dims} → {output_dims}"
#         elif operation == 'multiplication':
#             return f"Multiplication allows any dimensions: {input_dims}  ... → {output_dims}"
#         return f"Operation {operation}: {input_dims} → {output_dims}"

#     def validate(self, config: dict | None) -> list[ValidationResult]:
#         results = []
#         operation = config.get('operation')
#         input_dims = config.get('input_dimensions', [])
#         output_dims = config.get('output_dimensions', [])

#         if operation == 'addition' and input_dims != output_dims:
#             results.append(ValidationResult(
#                 is_valid=False,
#                 level='error',
#                 message="Addition requires identical input and output dimensions"
#             ))
#         return results

# class CategoryRetentionRule(ValidationRule):
#     def explain(self, config: dict) -> str:
#         operation = config.get('operation', 'unknown')

#         if operation == 'addition':
#             return "Addition retains all categories from inputs"
#         elif operation == 'multiplication':
#             return "Multiplication only retains categories present in both inputs"
#         return f"Operation {operation} has specific category retention rules"

#     def validate(self, config: dict | None) -> list[ValidationResult]:
#         results = []
#         operation = config.get('operation')
#         # Add validation logic for categories
#         return results

# class OperationBasketRule(ValidationRule):
#     def __init__(self):
#         self.basket_requirements = {
#             'historical': {'min_nodes': 1, 'required_tags': ['historical']},
#             'forecast': {'min_nodes': 0, 'required_tags': ['forecast']},
#             'goal': {'min_nodes': 0, 'required_tags': ['goal']},
#         }

#     def explain(self, config: dict) -> str:
#         operation = config.get('operation', 'unknown')
#         explanations = []

#         for basket, req in self.basket_requirements.items():
#             if req['min_nodes'] > 0:
#                 explanations.append(
#                     f"Requires at least {req['min_nodes']} input(s) with tags: {', '.join(req['required_tags'])}"
#                 )

#         return "; ".join(explanations) if explanations else f"Operation {operation} has no special basket requirements"

#     def validate(self, config: dict | None) -> list[ValidationResult]:
#         results = []
#         input_nodes = config.get('input_nodes', [])

#         for basket, requirements in self.basket_requirements.items():
#             matching_nodes = [
#                 node for node in input_nodes
#                 if any(tag in node.get('tags', []) for tag in requirements['required_tags'])
#             ]

#             if len(matching_nodes) < requirements['min_nodes']:
#                 results.append(ValidationResult(
#                     is_valid=False,
#                     level='error',
#                     message=f"Basket {basket} needs at least {requirements['min_nodes']} nodes"
#                 ))

#         return results

# class UnitCompatibilityRule(ValidationRule):
#     def explain(self, config: dict) -> str:
#         operation = config.get('operation', 'unknown')
#         input_units = [node.get('unit', 'unknown') for node in config.get('input_nodes', [])]
#         output_unit = config.get('unit', 'unknown')

#         return f"Operation {operation}: {' + '.join(input_units)} → {output_unit}"

#     def validate(self, config: dict | None) -> list[ValidationResult]:
#         results = []
#         operation = config.get('operation')
#         input_units = [node.get('unit') for node in config.get('input_nodes', [])]
#         output_unit = config.get('unit')

#         # Add your unit compatibility logic here
#         expected_result_unit = self._calculate_resulting_unit(operation, input_units)
#         if expected_result_unit != output_unit:
#             results.append(ValidationResult(
#                 is_valid=False,
#                 level='error',
#                 message=f"Unit mismatch: expected {expected_result_unit}, got {output_unit}"
#             ))

#         return results

#     # FIXME Validation code copied from nodes.py. Rewrite for the new context.
#     def _calculate_resulting_unit(self, operation: str, input_units: list[str]) -> str:
#         # Your unit calculation logic
#         return input_units[0] if input_units else 'unknown'

#         # Add formula if available # TODO Also describe other parameters.
#         if 'formula' in self.parameters.keys():
#             formula = self.get_parameter_value_str('formula', required=False)
#             html.append(f"<p>{'The formula is:'}</p>")
#             html.append(f"<pre>{formula}</pre>")


class EdgeRule(ValidationRule):

    def explain(self, node_config: dict[str, Any], context: Context) -> list[str]:
        txt = _('The inputs are processed in the following way before using for calculations in this node:')
        html = [f"<p>{txt}<ul>"]  # Start the main list for nodes
        edge_html0 = html.copy()

        def _get_node_name_text(node_id: str, context: Context) -> str:
            langs = ['']
            langs.append(f'_{context.instance.default_language}')
            langs.extend([f'_{lang}' for lang in context.instance.supported_languages])
            nes = context.node_explanation_system
            assert nes is not None
            node = nes.graph.nodes[node_id]
            for lang in langs:
                key = f'name{lang}'
                if node.get(key):
                    return _('Node <i>%s</i>') % node[key]
            return _('Node with identifier <i>%s</i>') % node_id

        for input_node in node_config.get('input_nodes', {}):

            tag_html = self.get_explanation_for_tag(input_node)
            tag_html.extend(self.get_explanation_for_edge_from(input_node, context))
            tag_html.extend(self.get_explanation_for_edge_to(input_node, context))

            if tag_html:
                input_id = input_node if not isinstance(input_node, dict) else input_node['id']
                txt = _get_node_name_text(input_id, context)
                html.append(f"<li>{txt}:")
                # Create a list item for the node with nested list
                html.append("<ul>")  # Start nested list for this node
                html.extend(tag_html)
                html.append("</ul></li>")  # Close node list item

        if html == edge_html0:
            return []
        html.append("</ul>")  # Close main nodes list
        return html

    def get_explanation_for_tag(self, node: dict[str, Any] | str) -> list[str]:
        html: list[str] = []
        if isinstance(node, str):
            return html

        for tag in node.get('tags', []):
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

    def explain(self, node_config: dict[str, Any] | str, context: Context) -> list[str]:
        assert isinstance(node_config, dict)
        node_id = node_config['id']

        nes = context.node_explanation_system
        assert nes is not None
        baskets = nes.baskets[node_id]
        html: list[str] = []
        if not baskets:
            return html

        html.append(_("The inputs are used for operations in this order:") + "<ol>")
        operation_list = self.get_param(nes.graph.nodes[node_id], 'operations')
        if not operation_list:
            operation_list = context.nodes[node_id].DEFAULT_OPERATIONS
        operations = [o.strip() for o in operation_list.split(',')]

        for operation in operations:
            input_nodes = baskets.get(operation, [])
            if input_nodes:
                op_display = BASKET_DISPLAY_NAMES.get(operation, operation)
                html.append(f"<li>{_('Operation %(op)s with nodes:') % {'op': op_display}}<ul>")
                nodes_str = '</li><li>'.join(str(context.nodes[n].name) for n in input_nodes)
                html.append(f"<li>{nodes_str}</li></ul></li>")
        html.append('</ol>')

        remaining_baskets = [basket for basket in baskets.keys() if basket not in operations]
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
