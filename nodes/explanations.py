from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from common.i18n import gettext as _

TAG_DESCRIPTIONS = {
    'additive': _("Add input node values (even if the units don't match with the node units)."),
    'arithmetic_inverse': _('Take the arithmetic inverse of the values (-x).'),
    'complement': _('Take the complement of the dimensionless values (1-x).'),
    'complement_cumulative_product': _('Take the cumulative product of the dimensionless complement values over time.'),
    'cumulative': _('Take the cumulative sum over time.'),
    'cumulative_product': _('Take the cumulative product of the dimensionless values over time.'),
    'difference': _('Take the difference over time (i.e. annual changes)'),
    'empty_to_zero': _('Convert NaNs to zeros.'),
    'expectation': _('Take the expected value over the uncertainty dimension.'),
    'extend_values': _('Extend the last historical values to the remaining missing years.'),
    'extend_forecast_values': _('Extend the last forecast values to the remaining missing years.'),
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
    'ratio_to_last_historical_value': _('Take the ratio of the values compared with the last historical value.'),
    'removing': _('This is the rate of stock removal.'),
    'truncate_before_start': _('Truncate values before the reference year. There may be some from data'),
    'truncate_beyond_end': _('Truncate values beyond the model end year. There may be some from data'),
}

NODE_CLASS_DESCRIPTIONS = { # FIXME Make descriptions concise.
    'AdditiveAction': _("""Simple action that produces an additive change to a value."""),
    'AdditiveNode': _(
        """This is an Additive Node. It performs a simple addition of inputs.
        Missing values are assumed to be zero."""),
    'AttributableFractionRR': _(
        """
        Calculate attributable fraction when the ERF function is relative risk.

        AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
        is smaller than 0, we should use r instead. It can be converted from the result:
        r/(r+1)=s <=> r=s/(1-s)
        """),
    'AssociationNode': _(
        """
        Association nodes connect to their upstream nodes in a loose way:
        Their values follow the relative changes of the input nodes but
        their quantities and units are not dependent on those of the input nodes.
        The node MUST have exactly one dataset, which is the prior estimate.
        Fractions 1..3 can be used to tell how much the input node should adjust
        the output node. The default relation is "increase", if "decrease" is used,
        that must be explicitly said in the tags.
        """),
    'BuildingEnergySavingAction': _(
        """
        Action that has an energy saving effect on building stock (per floor area).

        The output values are given per TOTAL building floor area,
        not per RENOVATEABLE building floor area. This is useful because
        the costs and savings from total renovations sum up to a meaningful
        impact on nodes that are given per floor area.
        """),
    'BuildingEnergySavingActionUs': _(
        """BuildingEnergySavingAction with U.S. units and natural gas instead of heat."""),
    'CfFloorAreaAction': _(
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
        """),
    'CoalesceNode': _(
        """Coalesces the empty values with the values from the node with the tag 'coalesce'."""
    ),
    'CohortNode': _(
        """
        Cohort node takes in initial age structure (inventory) and follows the cohort in time as it ages.

        Harvest describes how much is removed from the cohort.
        """),
    'CumulativeAdditiveAction': _("""Additive action where the effect is cumulative and remains in the future."""),
    'DatasetDifferenceAction': _(
        """
        Receive goal input from a dataset or node and cause an effect.

        The output will be a time series with the difference to the
        predicted baseline value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """),
    'DatasetDifferenceAction2': _(
        """
        Receive goal input from a dataset or node and cause an effect.

        The output will be a time series with the difference to the
        predicted baseline value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """),
    'DatasetNode': _(
        """This is a DatasetNode. It takes in a specifically formatted dataset and converts the relevant part into a node output.""",  # noqa: E501
    ),
    'DatasetReduceAction': _(
        """
        Receive goal input from a dataset or node and cause a linear effect.

        The output will be a time series with the difference to the
        last historical value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """),
    'DatasetReduceNode': _(
        """
        Receive goal input from a dataset or node and cause a linear effect.

        The output will be a time series with the difference to the
        last historical value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """),
    'DatasetRelationAction': _(
        """
        ActionRelationshipNode enforces a logical relationship with another action node.

        This node monitors an upstream action node (A) and automatically sets its own
        enabled state (B) according to the relationship specified in the edge tags.
        """),
    'DilutionNode': _(
        """
        This is Dilution Node. It has exactly four input nodes which are marked by tags: 1) existing is the current,
        non-diluted variable. 2) Incoming is the variable which diluted the existing one with its different values. 3)
        Removing is the fraction that is removed from the existing stock each year. 4) Incoming is the ratio compared
        with the existing stock that is inserted into the system. (Often the removed and incoming values are the same,
        and then the stock size remains constant.)
        """),
    'DimensionalSectorEmissionFactor': _(
        "Filters emissions and energy according to the <i>sector</i> parameter and calculates emission factor."),
    'DimensionalSectorEmissions': _("Filters emissions according to the <i>sector</i> parameter."),
    'DimensionalSectorEnergy': _("Filters energy use according to the <i>sector</i> parameter."),
    'DimensionalSectorNode': _(
        "Reads in a dataset and filters and interprets its content according to the <i>sector</i> parameter."),
    'EnergyAction': _("""Simple action with several energy metrics."""),
    'ExponentialNode': _(
        """
        This is Exponential Node.
        Takes in either input nodes as AdditiveNode, or builds a dataframe from current_value.
        Builds an exponential multiplier based on annual_change and multiplies the VALUE_COLUMN.
        Optionally, touches also historical values.
        Parameter is_decreasing_rate is used to give discount rates instead.
        """),
    'FillNewCategoryNode': _(
        """This is a Fill New Category Node. It behaves like Additive Node, but in the end of computation
        it creates a new category such that the values along that dimension sum up to 1. The input nodes
        must have a dimensionless unit. The new category in an existing dimension is given as parameter
        'new_category' in format 'dimension:category
        """),
    'FixedMultiplierNode': _(
        """This is a Fixed Multiplier Node. It multiplies a single input node with a parameter."""),
    'FloorAreaNode': _('Floor area node takes in actions and calculates the floor area impacted.'),
    'FormulaNode': _('This is a Formula Node. It uses a specified formula to calculate the output.'),
    'GenericNode': _("Multiply input nodes whose unit does not match the output. Add the rest."),
    'GpcTrajectoryAction': _(
        """
        GpcTrajectoryAction is a trajectory action that uses the DatasetNode to fetch the dataset.
        """),
    'InternalGrowthModel': _(
        """
        Calculates internal growth of e.g. a forest, accounting for forest cuts. Takes in additive and
        non-additive nodes and a dataset.
        Parameter annual_change is used where the rate node(s) have null values.
        """),
    'IterativeNode2': _( # FIXME Remove old
        """
        This is IterativeNode. It calculates one year at a time based on previous year's value and inputs and outputs.
        In addition, it must have a feedback loop (otherwise it makes no sense to use this node class), which is given
        as a growth rate per year from the previous year's value.
        """),
    'IterativeNode': _(
        """
        This is generic IterativeNode for calculating values year by year.
        It calculates one year at a time based on previous year's value and inputs and outputs
        starting from the first forecast year. In addition, it must have a feedback loop (otherwise it makes
        no sense to use this node class), which is given as a growth rate per year from the previous year's value.
        """),
    'LeverNode': _(
        """LeverNode replaces the upstream computation completely, if the lever is enabled."""
    ),
    'LinearCumulativeAdditiveAction': _(
        """
        Cumulative additive action where a yearly target is set and the effect is linear.
        This can be modified with these parameters:
        target_year_level is the value to be reached at the target year.
        action_delay is the year when the implementation of the action starts.
        multiplier scales the size of the impact (useful between scenarios).
        """),
    'LogicalNode': _( # FIXME There are several versions. Remove redundant.
        """
        LogicalNode processes logical inputs (values 0 or 1).

        It applies Boolean AND to multiplicative nodes (nodes are ANDed together)
        and Boolean OR to additive nodes (nodes are ORed together).

        AND operations are performed first, then OR operations. For more complex
        logical structures, use several subsequent nodes.
        """
    ),
    'LogitNode': _(
        """
        LogitNode gives a probability of event given a baseline and several determinants.

        The baseline is given as a dataset of observed values. The determinants are linearly
        related to the logit of the probability:
        ln(y / (1 - y)) = a + sum_i(b_i * X_i,)
        where y is the probability, a is baseline, X_i determinants and b_i coefficients.
        The node expects that a comes from dataset and sum_i(b_i * X_i,) is given by the input nodes
        when operated with the GenericNode compute(). The probability is calculated as
        ln(y / (1 - y)) = b <=> y = 1 / (1 + exp(-b)).
        """
    ),
    'MultiplicativeNode': _(
        """This is a Multiplicative Node. It multiplies nodes together with potentially adding other input nodes.

        Multiplication and addition is determined based on the input node units.
        """),
    'ReduceAction': _("""Define action with parameters <i>reduce</i> and <i>multiplier</i>."""),
    'SCurveAction': _(
        """
        This is S Curve Action. It calculates non-linear effect with two parameters,
        max_impact = A and max_year (year when 98 per cent of the impact has occurred).
        The parameters come from Dataset. In addition, there
        must be one input node for background data. Function for
        S-curve y = A/(1+exp(-k*(x-x0)). A is the maximum value, k is the steepness
        of the curve, and x0 is the midpoint year.
        Newton-Raphson method is used to numerically estimate slope and medeian year.
        """),
    'SectorEmissions': _("SectorEmissions is like AdditiveNode. It is used when creating nodes from emission_sectors."),
    'ThresholdNode': _( # FIXME Several versions. Remove redundant.
        """
        ThresholdNode computes a preliminary result using standard GenericNode operations.

        After computation, it returns True (1) if the result is greater than or equal to
        the threshold parameter, otherwise False (0).
        """
    ),
    'TrajectoryAction': _(
        """
        TrajectoryAction uses select_category() to select a category from a dimension
        and then possibly do some relative or absolute conversions.
        """),
    'WeightedSumNode': _(
        """
        WeightedSumNode: Combines additive inputs using weights from a multidimensional weights DataFrame.
        """
    ),
}


class GraphBuilder:
    @staticmethod
    def build_graph(all_node_configs: list[dict[str, Any]]) -> GraphRepresentation:
        """Build normalized graph from all node configs."""

        # Create node_id set for quick lookups
        node_ids = {node['id'] for node in all_node_configs}
        nodes_dict = {node['id']: node for node in all_node_configs}

        inputs: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        outputs: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        edges: dict[tuple, dict] = {}

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
    def _parse_edge_spec(input_spec) -> tuple[str, dict]:
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


class NodeExplanationSystem:
    def __init__(self):
        self.rules = [
            DatasetRule(),
            # CategoryRetentionRule(),
            # OperationBasketRule(),
            # UnitCompatibilityRule(),
        ]

    def validate_all_nodes(self, all_node_configs: list[dict[str, Any]]) -> dict[str, list[ValidationResult]]:
        """Validate all nodes with complete graph information."""

        # Step 1: Build complete graph representation
        try:
            graph = GraphBuilder.build_graph(all_node_configs)
        except KeyError as e:
            # Return graph-level errors for all nodes
            graph_error = ValidationResult(
                method='graph_rule',
                is_valid=False,
                level='error',
                message=str(e)
            )
            return {node['id']: [graph_error] for node in all_node_configs}

        # Step 2: Validate each node with complete graph context
        all_results = {}

        for node_id, node_config in graph.nodes.items():

            # Run all validation rules
            node_results = []
            for rule in self.rules:
                results = rule.validate(node_config)
                node_results.extend(results)

            all_results[node_id] = node_results

        return all_results

    def generate_explanation(self, config: dict | str | None) -> list[str]:
        """Generate explanation from all rules."""
        explanations = []
        # # Start with the explanation text # FIXME Requires access to node.py to collect node class explanations.
        # if self.explanation:
        #     html.append(f"<p>{self.explanation}")

        if isinstance(config, dict):
            params: dict | None = config.get('params')
            if params is not None and 'operations' in params.keys():
                operations = params.get('operations')
                explanations.append(f"The order of operations is {operations}.</p>")

        for rule in self.rules:
            explanation = rule.explain(config)
            if explanation:
                explanations.append(f"<li>{explanation}</li>")

        return explanations

    def validate_config(self, config: dict | str | None) -> list[ValidationResult]:
        """Validate config using all rules."""
        all_results = []

        for rule in self.rules:
            results = rule.validate(config)
            all_results.extend(results)

        return all_results

    def has_errors(self, validation_results: dict[str, list[ValidationResult]]) -> bool:
        """Check if any validation results are errors."""
        return any(
            any(rule.level == 'error' and not rule.is_valid for rule in node)
            for node in validation_results.values()
        )


@dataclass
class GraphRepresentation:
    """Normalized representation of the complete node graph."""

    nodes: dict[str, dict]  # node_id -> node_config
    inputs: dict[str, list[str]]  # node_id -> list of input_node_ids
    outputs: dict[str, list[str]]  # node_id -> list of output_node_ids
    edges: dict[tuple, dict]  # (from_node, to_node) -> edge_properties


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


@dataclass
class ValidationResult:
    method: str
    is_valid: bool
    level: str  # 'error', 'warning', 'info'
    message: str


class ValidationRule(ABC):
    """Base class for validation rules that also generate explanations."""

    @abstractmethod
    def explain(self, config: dict | str | None) -> list[str]:
        """Generate explanation text from node config."""
        pass

    @abstractmethod
    def validate(self, config: dict | str | None) -> list[ValidationResult]:
        """Validate the node configuration."""
        pass

class DatasetRule(ValidationRule):

    def explain(self, config: dict | str | None) -> list[str]:
        dataset_html: list[str] = []
        if config is None or isinstance(config, str):
            return dataset_html

        col = config.get('column')
        if col is not None:
            dataset_html.append('<li>' + str(_('Is using column: ')) + col + '</li>')
        year = config.get('forecast_from')
        if year is not None:
            dataset_html.append('<li>' + str(_('Has forecast values from: ')) + str(year) + '</li>')
        dropna = config.get('dropna')
        if dropna is not None and dropna:
            dataset_html.append('<li>' + str(_('Rows with missing valuesa are dropped.')) + '</li>')

        filters = config.get('filters')
        if filters is not None: # FIXME Translations should happen on usage, not on creation.
            dataset_html.append(str(_(" has the following filters:")))
            dataset_html.append("<ul>")
            filter_text = _("Filter")
            filter_no = 1
            for filter_dict in filters:
                dataset_html.append(f"<li>{filter_text} {filter_no}")
                if isinstance(filter_dict, dict):
                    dataset_html.append("<ul>")
                    for key, value in filter_dict.items():
                        v = str(value) # FIXME Should understand i18n category names.
                        dataset_html.append(f"<li><strong>{key}:</strong> {v}</li>")
                    dataset_html.append("</ul>")
                dataset_html.append("</li>")
                filter_no += 1
            dataset_html.append("</ul>")
        dataset_html.append("</li>")

        return dataset_html

    def validate(self, config: dict | str | None) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        if config is None:
            return results

        results.append(ValidationResult(
            method='dataset_test',
            is_valid=True,
            level='info',
            message="There are no validation for datasets yet."
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

#         # # Handle input nodes # FIXME Add operations when the buckets is a node attribute.
#         # operation_nodes = [getattr(n, 'translated_name', n.name) for n in self.input_nodes]
#         # if operation_nodes:
#         #     html.append(f"<p>{'The node has the following input nodes:'}</p>")
#         #     html.append("<ul>")
#         #     html.extend([f"<li>{node_name}</li>" for node_name in operation_nodes])
#         #     html.append("</ul>")
#         # else:
#         #     html.append(f"<p>{'The node does not have input nodes.'}</p>")


#         # Add datasets information
#         dataset_html = []
#         if self.input_dataset_instances:
#             df = self.get_output_pl()
#             dataset_html.append(f"<p>{'The node has the following datasets:'}</p>")
#             dataset_html.append("<ul>")
#             dataset_html.extend(df.explanation)
#             dataset_html.append("</ul>")

#         edge_html = self.get_edge_explanation()
#         # Combine all parts
#         if edge_html:
#             html.extend(edge_html)
#         if dataset_html:
#             html.extend(dataset_html)

#         return "".join(html)

#     def get_edge_explanation(self):
#         edge_html = []
#         edge_html.append(f"<p>{
#             'The input nodes are processed in the following way before using as input for calculations in this node:'
#         }</p>")
#         edge_html.append("<ul>")  # Start the main list for nodes
#         edge_html0 = edge_html.copy()

#         for node in self.input_nodes:
#             for edge in node.edges:
#                 if edge.output_node != self:
#                     continue

#                 tag_html = self.get_explanation_for_edge_tag(edge)
#                 from_html = self.get_explanation_for_edge_from(edge)
#                 to_html = self.get_explanation_for_edge_to(edge)

#             if tag_html or from_html or to_html:

#                 node_name = getattr(node, 'translated_name', node.name)

#                 # Create a list item for the node with nested list
#                 edge_html.append(f"<li>{'Node'} <i>{node_name}</i>:")
#                 edge_html.append("<ul>")  # Start nested list for this node
#                 edge_html.extend(tag_html)
#                 edge_html.extend(from_html)
#                 edge_html.extend(to_html)
#                 edge_html.append("</ul>")  # Close node's nested list
#                 edge_html.append("</li>")  # Close node list item

#         if edge_html == edge_html0:
#             return []
#         edge_html.append("</ul>")  # Close main nodes list
#         return edge_html

#     def get_explanation_for_edge_tag(self, edge):
#         edge_html = []
#         # Process edge tags using the lookup dictionary
#         if edge.tags:
#             for tag in edge.tags:
#                 description = TAG_DESCRIPTIONS.get(tag, _('The tag <i>"%s"</i> is given.') % tag)
#                 edge_html.append(f"<li>{description}</li>")
#         return edge_html

#     def get_explanation_for_edge_from(self, edge):
#         edge_html = []
#         from_dims = edge.from_dimensions
#         if from_dims is not None:
#             for dim in from_dims:
#                 dimlabel = self.context.dimensions[dim].label
#                 cats = [str(cat.label) for cat in from_dims[dim].categories]

#                 if cats:
#                     do = _('exclude') if from_dims[dim].exclude else _('include')
#                     edge_html.append(
#                         f"<li>{_('From dimension <i>%s</i>, %s categories: <i>%s</i>.') % (dimlabel, do, ', '.join(cats))}</li>"
#                     )

#                 if from_dims[dim].flatten:
#                     edge_html.append(f"<li>{_('Sum over dimension <i>%s</i>.') % dimlabel}</li>")
#         return edge_html

#     def get_explanation_for_edge_to(self, edge):
#         edge_html = []
#         to_dims = edge.to_dimensions
#         if to_dims is not None:
#             for dim in to_dims:
#                 dimlabel = self.context.dimensions[dim].label
#                 cats = [str(cat.label) for cat in to_dims[dim].categories]

#                 if cats:
#                     cat_str = ', '.join(cats)
#                     edge_html.append(
#                        f"<li>{_('Categorize the values to <i>%s</i> in a new dimension <i>%s</i>.') % (cat_str, dimlabel)}</li>"
#                     )
#         return edge_html
