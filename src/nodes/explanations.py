from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from kausal_common.i18n.pydantic import gettext_lazy as _

from .constants import TIME_INTERVAL
from .defs.transform_def import (
    AssignDimensionOp,
    DropNullsOp,
    FilterColumnOp,
    FilterDimensionOp,
    RenameColumnOp,
    RenameItemOp,
    SetForecastFromOp,
)
from .formula import (
    FormulaSpec,
    analyze_formula_dimensions,
    analyze_formula_units,
    build_name_dimension_map,
    build_name_unit_map,
    collect_term_names,
    make_identifier,
    normalize_formula_identifiers,
)
from .units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kausal_common.i18n.pydantic import I18nString

    from nodes.context import Context
    from nodes.defs.node_defs import InputDatasetDef
    from nodes.defs.transform_def import PortTransformOp

    from .explanation_inputs import ExplainedEdge, ExplainedNode
    from .formula import (
        UnitOverride,
    )
    from .units import Unit

TAG_TO_BASKET = {
    'additive': 'add',
    'add_from_incoming_dims': 'add_from_incoming_dims',
    'add_to_existing_dims': 'add_to_existing_dims',
    'base': 'other',
    'impute': 'impute',
    # These operations only handle existing df and do not take new inputs:
    # apply_multipliet, do_correction, drop_nand, drop_infs, extend_values, extrapolate
    # [concat_datasets: generalize to use the port theory]
    # inventory_only, other, select_variant
    'non_additive': 'multiply',
    'other_node': 'other',
    'partial_factor': 'multiply',
    'primary': 'coalesce',
    'rate': 'other',
    'secondary': 'coalesce',
    'skip_dim_test': 'skip_dim_test',
    'split_by_existing_shares': 'split_by_existing_shares',
    'split_evenly_to_cats': 'split_evenly_to_cats',
    # `split_dims` resolves these itself; without them here `claimed_by_other_operation`
    # returns False and the input is also swept into the additive bucket, where it fails
    # the dimension test before the operation ever runs. That made the operation the
    # docstring calls preferred unusable from YAML.
    'splittee': 'split_dims',
    'splitter': 'split_dims',
    'use_as_shares': 'use_as_shares',
    'use_as_totals': 'use_as_totals',
}

BASKET_DISPLAY_NAMES = {  # FIXME We may not need explicit basket names.
    'add': _('addition'),
    'add_from_incoming_dims': _('addition from incoming dimensions'),
    'add_to_existing_dims': _('addition to existing dimensions'),
    'coalesce': _('coalesce'),
    'impute': _('imputation'),
    'multiply': _('multiplication'),
    'other': _('other operations'),
    'skip_dim_test': _('skip dimension test'),
    'split_by_existing_shares': _('split by existing shares'),
    'split_dims': _('split across dimensions'),
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
    'impute': ', ',
    'multiply': ' * ',
}

TAG_DESCRIPTIONS = {
    'action_with_history': _('Action node evaluating also its historical impact, not only future impacts.'),
    'add_datasets': _('Get and prepare each dataset, then add them together.'),
    'add_from_incoming_dims': _('Add input values, introducing new dimensions from the input into the result.'),
    'add_to_existing_dims': _('Add input values into the existing dimensions of the result without expanding them.'),
    'additive': _("Add input node values (even if the units don't match with the node units)."),
    'and': _('Logical AND: min(a, b). Warns if inputs deviate from 0 or 1 (see node explanation).'),
    'arithmetic_inverse': _('Take the arithmetic inverse of the values (-x).'),
    'base': _('Use this input as the base value to start from.'),
    'bring_to_maximum_historical_year': _('Makes all years up to maximum historical year non-forecasts.'),
    'bring_to_reference_year': _('Makes all years after reference year forecast years.'),
    'city_data': _('Check if city-specific data exists for this framework model.'),
    'coalesce': _('Use the first non-null value among the inputs (priority order).'),
    'complement': _('Take the complement of the unitless values (1-x).'),
    'complement_cumulative_product': _('Take the cumulative product of the dimensionless complement values over time.'),
    'concat_datasets': _('Get and concatenate datasets vertically, only then prepare the output.'),
    'cumulative': _('Take the cumulative sum over time.'),
    'cumulative_product': _('Take the cumulative product of the dimensionless values over time.'),
    'difference': _('Take the difference over time (i.e. annual changes)'),
    'drop_infs': _('Drop long-format rows with infinite values.'),
    'drop_nans': _('Drop long-format rows with NaN/null values.'),
    'empty_to_zero': _('Convert NaNs and Nulls to zeros in wide format.'),
    'existing': _('This is used as the baseline.'),
    'expectation': _('Take the expected value over the uncertainty dimension.'),
    'extend_all': _('Extend the values to all the remaining missing years.'),
    'extend_both_ways': _('Extend the values beyond the first and last values, but do not interpolate.'),
    'extend_forecast_values': _('Extend the last forecast values to the remaining missing years.'),
    'extend_to_history': _('Extend the first values to the years after the minimum historical year.'),
    'extend_values': _('Extend the last historical values to the remaining missing years.'),
    'geometric_inverse': _('Take the geometric inverse of the values (1/x).'),
    'get_single_dataset': _('Get a single dataset if it exists.'),
    'goal': _('The node is used as the goal for the action.'),
    'goal_gap': _("Compute gap = actual - goal from the single input node's output and goals."),
    'historical': _('The node is used as the historical starting point.'),
    'ignore_content': _('Show edge on graphs but ignore upstream content.'),
    'impute': _(
        'Overlay this input onto the result, outer-joined on dimensions: the input value replaces the '
        "result's own value wherever the input has one, and the result's own value is used elsewhere."
    ),
    'incoming': _('This is used for the incoming stock.'),
    'inserting': _('This is the rate of new stock coming in.'),
    'inventory_only': _('Truncate the forecast values.'),
    'make_nonnegative': _('Negative result values are replaced with 0.'),
    'make_nonpositive': _('Positive result values are replaced with 0.'),
    'max': _('Element-wise maximum of two values; max(a, b). For 0/1 inputs this is logical OR.'),
    'min': _('Element-wise minimum of two values; min(a, b). For 0/1 inputs this is logical AND.'),
    'non_additive': _('Input node values are not added but operated despite matching units.'),
    'partial_factor': _(
        'The factor covers only some of the categories of the node; the rest are left unchanged rather than dropped.'
    ),
    'observed_only_extend_all': _('Extend the observed data only based on the observed data points.'),
    'or': _('Logical OR: max(a, b). Warns if inputs deviate from 0 or 1 (see node explanation).'),
    'other_node': _('Auxiliary input used in a non-standard role not covered by other tags.'),
    'prepare_gpc_dataset': _('Prepare a GPC-style dataset for use.'),
    'primary': _('Use data as primary values even if a secondary value exists.'),
    'rate': _('This input is a rate (per-unit or fractional value) applied to another quantity.'),
    'ratio_to_last_historical_value': _('Take the ratio of the values compared with the last historical value.'),
    'ratio_to_max_hist_year': _(
        'Take the ratio of the forecasted values compared with the maximum historical' + ' year (historical values = 1).'
    ),
    'prefer_by_year': _('Use the first option for every year it has data for, and the second option for the remaining years.'),
    'removing': _('This is the rate of stock removal.'),
    'round_to_five': _('Round values to 5 significant digits rather than 5 decimal places.'),
    'scenario_impact': _('Calculate the total impact of all actions in the current scenario.'),
    'secondary': _('Use data only if a primary value does not exist.'),
    'select_port': _('If condition is True, select the first option, otherwise the second.'),
    'skip_dim_test': _('Add input values while skipping the dimension compatibility check.'),
    'split_by_existing_shares': _('Distribute the total across categories according to existing shares in the result.'),
    'split_dims': _('Distribute the total of splittee across categories according to the splitter.'),
    'splittee': _('The node is used as the values to redistribute into a new dimension.'),
    'splitter': _('The node is used as the distribution source for a new dimension.'),
    'split_evenly_to_cats': _('Distribute the total evenly across all categories of a dimension.'),
    'template': _('The dataset declaring the category combinations that are required to exist.'),
    'trendline': _('Fit a linear trend to the last historical years (trend_years) and extrapolate it over the forecast years.'),
    'truncate_before_start': _('Truncate values before the reference year. There may be some from data'),
    'truncate_beyond_end': _('Truncate values beyond the model end year. There may be some from data'),
    'use_as_shares': _('Treat this input as fractional shares (dimensionless) to scale another quantity.'),
    'use_as_totals': _('Treat this input as a total to be distributed across categories.'),
}


def _unit_dimensionless(_unit: Unit | None) -> Unit:
    return unit_registry.parse_units('dimensionless')


def _unit_mul_time(unit: Unit | None) -> Unit | None:
    if unit is None:
        return None
    return cast('Unit', unit * unit_registry.parse_units(TIME_INTERVAL))


def _unit_div_time(unit: Unit | None) -> Unit | None:
    if unit is None:
        return None
    return cast('Unit', unit / unit_registry.parse_units(TIME_INTERVAL))


def _unit_geometric_inverse(unit: Unit | None) -> Unit | None:
    if unit is None:
        return None
    return cast('Unit', unit_registry.parse_units('dimensionless') / unit)


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
    description: I18nString
    deprecated: bool = False


# FIXME Make descriptions concise.
NODE_CLASS_DESCRIPTIONS: dict[str, NodeInfo] = {
    'AdditiveAction': NodeInfo(_("""Simple action that produces an additive change to a value.""")),
    'AdditiveNode': NodeInfo(_('')),
    'AdditiveNode2': NodeInfo(
        _("""Adds up all of its inputs, whether they arrive as nodes or as datasets. Inputs must have
        the same dimensions and compatible units; a missing value counts as zero.""")
    ),
    'AlasEmissions': NodeInfo(
        _("""AlasEmissions is a specified node to handle emissions from the ALas model by Syke."""), deprecated=True
    ),
    'AlasNode': NodeInfo(_("""AlasNode is a specified node to handle data from the ALas model by Syke."""), deprecated=True),
    'AttributableFractionRR': NodeInfo(
        _(
            """
        Calculate attributable fraction when the ERF function is relative risk.

        AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
        is smaller than 0, we should use r instead. It can be converted from the result:
        r/(r+1)=s <=> r=s/(1-s)
        """
        )
    ),
    'BiskoChpNode': NodeInfo(
        _(
            """
        Splits the emissions of combined heat and power (CHP) production the way BISKO prescribes.
        This is the ChpNode with its method fixed: the exergetic (Carnot) split, with the district
        heating return temperature fixed at 283 K, as required by BISKO criterion 6.

        Those choices belong to the standard rather than to the city, so they cannot be changed here
        or in a scenario. What does vary between cities and between years -- the electricity fraction
        of the plant's output and the supply temperature of the network -- is given per year, from a
        dataset, from an input node, or as a single parameter value.
        """
        )
    ),
    'BiskoExergeticAllocationNode': NodeInfo(
        _(
            """
        Reports whether combined heat and power is allocated the way BISKO criterion 6 requires
        (1 = yes, 0 = no). Two conditions must hold in the same year.

        First, the prescribed method has to be in force: the exergetic (Carnot) split with the
        district heating return temperature fixed at 283 K. This node asks the allocation node
        itself which method it applies, rather than guessing from the numbers that came out of it.

        Second, the allocation must have had something to allocate. In a year where the district
        heating emissions of the balance are zero or missing, the answer is 0, because a method
        applied to an empty balance is not evidence that the method was used.

        The node says nothing about whether the electricity fraction and supply temperature fed
        into the split are plausible. That is a separate test.
        """
        )
    ),
    'BuildingEnergySavingAction': NodeInfo(
        _(
            """
        Action that has an energy saving effect on building stock (per floor area).

        The output values are given per TOTAL building floor area,
        not per RENOVATEABLE building floor area. This is useful because
        the costs and savings from total renovations sum up to a meaningful
        impact on nodes that are given per floor area.
        """
        )
    ),
    'CfFloorAreaAction': NodeInfo(
        _(
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
        """
        )
    ),
    'ChpNode': NodeInfo(
        _(
            """
        Splits the emissions of combined heat and power (CHP) production between electricity and
        district heat, following the
        <a href="https://ghgprotocol.org/sites/default/files/CHP_guidance_v1.0.pdf">GHG Protocol CHP guidance</a>.
        The node outputs the two allocation fractions, which sum up to 1 in every year; multiplying
        them with the average emission factor of the fuel mix gives a factor for each product.
        <br/>a<sub>i</sub> = z<sub>i</sub> * f<sub>i</sub> / sum<sub>i</sub>(z<sub>i</sub> * f<sub>i</sub>),
        <br/>where a<sub>i</sub> is the fraction for each product (i = electricity, heat),
        z<sub>i</sub> is a method-specific multiplier (see below), and
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
        Typical values are n<sub>heat</sub> = 0.9, n<sub>electricity</sub> = 0.4.</li></ol>

        A plant does not run the same way every year, so the electricity fraction and the supply and
        return temperatures are read per year. Each of them comes from an input node tagged with its
        name, or from a column of that name in the input dataset, or -- if the city has only one
        representative value rather than a series -- from the parameter of that name. Annual series
        are interpolated over gaps and held constant beyond the years they cover.
        """
        )
    ),
    'CoalesceNode': NodeInfo(
        _("Uses 'primary' tagged data when available, otherwise 'secondary' tagged data. One of the tags must be given.")
    ),
    'CohortNode': NodeInfo(
        _(
            """
        Cohort node takes in initial age structure (inventory) and follows the cohort in time as it ages.

        Harvest describes how much is removed from the cohort.
        """
        )
    ),
    'ConstantNode': NodeInfo(
        _(
            """
        Constant node returns a constant value spread over the timeline.
        """
        )
    ),
    'CumulativeAdditiveAction': NodeInfo(_("""Additive action where the effect is cumulative and remains in the future.""")),
    'DataAvailabilityNode': NodeInfo(
        _(
            """
        This node does not use the values of its input dataset but reports whether a value exists
        in each cell: 1 where the dataset has a value and 0 where it does not. The check is made on
        the original data, before interpolation or extension fill in the missing years. The output
        covers the whole model period, and the years and categories that the data does not reach get 0.
        """
        )
    ),
    'DatasetDifferenceAction': NodeInfo(
        _(
            """
        Receive goal input from a dataset or node and cause an effect.

        The output will be a time series with the difference to the
        predicted baseline value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """
        )
    ),
    'DatasetDifferenceAction2': NodeInfo(
        _(
            """
        Receive goal input from a dataset or node and cause an effect.

        The output will be a time series with the difference to the
        predicted baseline value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """
        )
    ),
    'DatasetNode': NodeInfo(
        _(
            """
        This is a DatasetNode. It takes in a specifically formatted dataset and
        converts the relevant part into a node output.
        """
        )
    ),
    'DatasetReduceAction': NodeInfo(
        _(
            """
        Receive goal input from a dataset or node and cause a linear effect.

        The output will be a time series with the difference to the
        last historical value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """
        )
    ),
    'DatasetReduceNode': NodeInfo(
        _(
            """
        Receive goal input from a dataset or node and cause a linear effect.

        The output will be a time series with the difference to the
        last historical value of the input node.

        The goal input can also be relative (for e.g. percentage
        reductions), in which case the input will be treated as
        a multiplier.
        """
        )
    ),
    'DatasetRelationAction': NodeInfo(
        _(
            """
        ActionRelationshipNode enforces a logical relationship with another action node.

        This node monitors an upstream action node (A) and automatically sets its own
        enabled state (B) according to the relationship specified in the edge tags.
        """
        )
    ),
    'DilutionNode': NodeInfo(
        _(
            """
        This is Dilution Node. It has exactly four input nodes which are marked by tags: 1) existing is the current,
        non-diluted variable. 2) Incoming is the variable which diluted the existing one with its different values. 3)
        Removing is the fraction that is removed from the existing stock each year. 4) Incoming is the ratio compared
        with the existing stock that is inserted into the system. (Often the removed and incoming values are the same,
        and then the stock size remains constant.)
        """
        )
    ),
    'DimensionalSectorEmissionFactor': NodeInfo(
        _('Filters emissions and energy according to the <i>sector</i> parameter and calculates emission factor.')
    ),
    'DimensionalSectorEmissions': NodeInfo(_('Filters emissions according to the <i>sector</i> parameter.')),
    'DimensionalSectorEnergy': NodeInfo(_('Filters energy use according to the <i>sector</i> parameter.')),
    'DimensionalSectorNode': NodeInfo(
        _('Reads in a dataset and filters and interprets its content according to the <i>sector</i> parameter.')
    ),
    'EnergyAction': NodeInfo(_("""Simple action with several energy metrics.""")),
    'ExponentialNode': NodeInfo(
        _(
            """
        This is Exponential Node.
        Takes in either input nodes as AdditiveNode, or builds a dataframe from current_value.
        Builds an exponential multiplier based on annual_change and multiplies the VALUE_COLUMN.
        Optionally, touches also historical values.
        Parameter is_decreasing_rate is used to give discount rates instead.
        """
        )
    ),
    'FillNewCategoryNode': NodeInfo(
        _(
            """This is a Fill New Category Node. It behaves like Additive Node, but in the end of computation
        it creates a new category such that the values along that dimension sum up to 1. The input nodes
        must have a dimensionless unit. The new category in an existing dimension is given as parameter
        'new_category' in format 'dimension:category
        """
        )
    ),
    'FixedMultiplierNode': NodeInfo(
        _("""This is a Fixed Multiplier Node. It multiplies a single input node with a parameter.""")
    ),
    'FloorAreaNode': NodeInfo(_('Floor area node takes in actions and calculates the floor area impacted.')),
    'FormulaNode': NodeInfo(_('')),
    'GenerationCapacityNode': NodeInfo(
        _("""
        Calculates generation of energy when new capacity is installed. Includes scope 3 emissions from installation,
        and emissions avoided from the capacity that gets replaced.
        """)
    ),
    'GenericNode': NodeInfo(_('')),
    'ScenarioImpactNode': NodeInfo(
        _(
            """Gives the difference between the current scenario and a reference scenario
        for the single input node. Reference scenario is configurable (default: baseline)."""
        )
    ),
    'ActionWithHistoryNode': NodeInfo(
        _(
            """
        Calculates the effects of actions that started already during historical years.
        The scenario historical_actions contains info about which actions were implemented.
        """
        )
    ),
    'GenericAction': NodeInfo(_('')),
    'GpcTrajectoryAction': NodeInfo(
        _(
            """
        GpcTrajectoryAction is a trajectory action that uses the DatasetNode to fetch the dataset.
        """
        )
    ),
    'InternalGrowthModel': NodeInfo(
        _(
            """
        Calculates internal growth of e.g. a forest, accounting for forest cuts. Takes in additive and
        non-additive nodes and a dataset.
        Parameter annual_change is used where the rate node(s) have null values.
        """
        )
    ),
    'IterativeNode2': NodeInfo(
        _(
            """
        This is IterativeNode. It calculates one year at a time based on previous year's value and inputs and outputs.
        In addition, it must have a feedback loop (otherwise it makes no sense to use this node class), which is given
        as a growth rate per year from the previous year's value.
        """
        ),
        deprecated=True,
    ),  # FIXME Remove old
    'IterativeNode': NodeInfo(
        _(
            """
        This is generic IterativeNode for calculating values year by year.
        It calculates one year at a time based on previous year's value and inputs and outputs
        starting from the first forecast year. In addition, it must have a feedback loop (otherwise it makes
        no sense to use this node class), which is given as a growth rate per year from the previous year's value.
        """
        )
    ),
    'LeverNode': NodeInfo(_("""LeverNode replaces the upstream computation completely, if the lever is enabled.""")),
    'LinearCumulativeAdditiveAction': NodeInfo(
        _(
            """
        Cumulative additive action where a yearly target is set and the effect is linear.
        This can be modified with these parameters:
        target_year_level is the value to be reached at the target year.
        action_delay is the year when the implementation of the action starts.
        multiplier scales the size of the impact (useful between scenarios).
        """
        )
    ),
    'LogitNode': NodeInfo(
        _(
            """
        LogitNode gives a probability of event given a baseline and several determinants.

        The baseline is given as a dataset of observed values. The determinants are linearly
        related to the logit of the probability:
        ln(y / (1 - y)) = a + sum<sub>i</sub>(b<sub>i</sub> * X<sub>i</sub>,)
        where y is the probability, a is baseline, X<sub>i</sub> determinants and b<sub>i</sub> coefficients.
        The node expects that a comes from dataset and sum<sub>i</sub>(b<sub>i</sub> * X<sub>i</sub>,) is given by the input nodes
        when operated with the GenericNode compute(). The probability is calculated as
        ln(y / (1 - y)) = b <=> y = 1 / (1 + exp(-b)).
        """
        )
    ),
    'MultiplicativeNode2': NodeInfo(
        _("""Multiplies its factors together and adds any additive inputs to the product. Inputs may be
        nodes or datasets, and are sorted into factors and addends by tag or by unit.""")
    ),
    'MultiplicativeNode': NodeInfo(_('')),
    'Population': NodeInfo(_('Population is a specific node about Finnish population.'), deprecated=True),
    'ReduceAction': NodeInfo(_("""Define action with parameters <i>reduce</i> and <i>multiplier</i>.""")),
    'SCurveAction': NodeInfo(
        _(
            """
        This is S Curve Action. It calculates non-linear effect with two parameters,
        max_impact = A and max_year (year when 98 per cent of the impact has occurred).
        The parameters come from Dataset. In addition, there
        must be one input node for background data. Function for
        S-curve y = A/(1+exp(-k*(x-x0)). A is the maximum value, k is the steepness
        of the curve, and x0 is the midpoint year.
        Newton-Raphson method is used to numerically estimate slope and medeian year.
        """
        )
    ),
    'SectorEmissions': NodeInfo(_('')),
    'ShiftAction': NodeInfo(_('ShiftAction moves activity from one category to others.')),
    'TrajectoryAction': NodeInfo(
        _(
            """
        TrajectoryAction uses select_category() to select a category from a dimension
        and then possibly do some relative or absolute conversions.
        """
        )
    ),
    'ValueAction': NodeInfo(
        _("""
        Value action outputs a constant (e.g. weight for a moral value) over time.
        Adjust the weight parameter to change how much this value contributes to priorities.
        """)
    ),
    'WeightedSumNode': NodeInfo(
        _(
            """
        WeightedSumNode: Combines additive inputs using weights from a multidimensional weights DataFrame.
        """
        )
    ),
    'Unknown': NodeInfo(_('Node class does not have description.')),
}


@dataclass
class GraphRepresentation:
    """Normalized representation of the complete node graph."""

    nodes: dict[str, ExplainedNode] = field(default_factory=dict)
    inputs: dict[str, list[str]] = field(default_factory=dict)  # node_id -> list of input_node_ids
    outputs: dict[str, list[str]] = field(default_factory=dict)  # node_id -> list of output_node_ids
    edges: dict[tuple[str, str], ExplainedEdge] = field(default_factory=dict)


@dataclass
class ValidationResult:
    method: str
    is_valid: bool
    level: Literal['error', 'warning', 'info']
    message: str


@dataclass
class TermInfo:
    label: str
    kind: str  # 'node', 'dataset', 'constant'
    name: str
    unit: str | None = None
    value: Any = None
    output_dimensions: list[str] | None = None
    details: list[str] = field(default_factory=list)  # HTML fragment items for nested detail


@dataclass
class NodeExplanation:
    """Structured explanation for a node, converted to HTML only as a final step."""

    node_id: str = ''
    node_type: str = ''
    description: str = ''
    operations: str | None = None
    params: list[tuple[str, str]] = field(default_factory=list)
    formula: str | None = None
    terms: list[TermInfo] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)  # HTML <li> items
    leftover_html: list[str] = field(default_factory=list)  # catch-all for rules not yet restructured

    def __bool__(self) -> bool:
        return bool(self.node_id or self.formula or self.terms or self.functions or self.leftover_html)


def _format_explanation_value(value: Any) -> str:
    """Format semantically equal YAML and typed-spec values identically."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def explanation_to_html(exp: NodeExplanation) -> list[str]:  # noqa: C901, PLR0912
    """Convert a structured NodeExplanation to a flat list of HTML fragments."""
    html: list[str] = []
    if exp.node_id:
        html.append(f'{exp.node_id} ({exp.node_type})<br>')
    if exp.description:
        html.append(f'{exp.description}<ul>')
    if exp.operations:
        html.append(f'<li>{_("The order of operations is %s.") % exp.operations}</li>')
    for param_id, param_val in exp.params:
        text = _('Has parameter <i>%(parameter)s</i> with value %(value)s.') % {
            'parameter': param_id,
            'value': param_val,
        }
        html.append(f'<li>{text}</li>')
    if exp.description:
        html.append('</ul>')
    if exp.formula:
        html.append(f'<p>{_("Formula:")} <b>{exp.formula}</b>,</p>')
    if exp.terms:
        html.append(_('Terms:') + '<ul>')
        for term in exp.terms:
            suffix_parts: list[str] = []
            if term.kind == 'constant' and term.value is not None:
                suffix_parts.append(_format_explanation_value(term.value))
            if term.unit:
                suffix_parts.append(str(term.unit))
            if term.output_dimensions:
                suffix_parts.append(_('dims: %(dims)s') % {'dims': ', '.join(term.output_dimensions)})
            suffix = f' ({"; ".join(suffix_parts)})' if suffix_parts else ''
            if term.details:
                html.append(f'<li><b>{term.label}</b> ({term.kind}): {term.name}{suffix}<ul>')
                html.extend(term.details)
                html.append('</ul></li>')
            else:
                html.append(f'<li><b>{term.label}</b>: {term.name}{suffix}</li>')
        html.append('</ul>')
    if exp.functions:
        html.append(_('Functions:') + '<ul>')
        html.extend(exp.functions)
        html.append('</ul>')
    html.extend(exp.leftover_html)
    return html


def _merge_node_explanations(parts: list[NodeExplanation]) -> NodeExplanation:
    merged = NodeExplanation()
    for part in parts:
        if part.node_id and not merged.node_id:
            merged.node_id = part.node_id
        if part.node_type and not merged.node_type:
            merged.node_type = part.node_type
        if part.description and not merged.description:
            merged.description = part.description
        if part.operations is not None and merged.operations is None:
            merged.operations = part.operations
        merged.params.extend(part.params)
        if part.formula is not None and merged.formula is None:
            merged.formula = part.formula
        merged.terms.extend(part.terms)
        merged.functions.extend(part.functions)
        merged.leftover_html.extend(part.leftover_html)
    return merged


class GraphBuilder:
    @staticmethod
    def build_graph(nodes: Sequence[ExplainedNode]) -> GraphRepresentation:
        """Index the nodes and their incoming edges; a source outside the set is kept as an input, never validated here."""
        nodes_dict = {node.id: node for node in nodes}
        inputs: dict[str, list[str]] = {node_id: [] for node_id in nodes_dict}
        outputs: dict[str, list[str]] = {node_id: [] for node_id in nodes_dict}
        edges: dict[tuple[str, str], ExplainedEdge] = {}

        for node in nodes:
            for edge in node.inputs:
                inputs[node.id].append(edge.source_id)
                if edge.source_id in nodes_dict:
                    outputs[edge.source_id].append(node.id)
                edges[(edge.source_id, node.id)] = edge

        return GraphRepresentation(nodes=nodes_dict, inputs=inputs, outputs=outputs, edges=edges)


@dataclass
class NodeExplanationSystem:
    context: Context

    graph: GraphRepresentation = field(init=False)

    nodes: InitVar[Sequence[ExplainedNode]]

    explanations: dict[str, NodeExplanation] = field(default_factory=dict)
    """Static explanations generated from node configurations."""

    validations: dict[str, list[ValidationResult]] = field(default_factory=dict)

    baskets: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def __post_init__(self, nodes: Sequence[ExplainedNode]):
        self.rules = [
            NodeClassRule(),
            DatasetRule(),
            EdgeRule(),
            BasketRule(),
            FormulaDimensionRule(),
            FormulaUnitRule(),
        ]
        self.graph = GraphBuilder.build_graph(nodes)

    def generate_validations(self) -> dict[str, list[ValidationResult]]:
        """Validate all nodes with complete graph information."""
        all_results: dict[str, list[ValidationResult]] = {}

        # Step 2: Validate each node with complete graph context
        for node_id, node in self.graph.nodes.items():
            node_results: list[ValidationResult] = []
            for rule in self.rules:
                node_results.extend(rule.validate(node, self.context))
            all_results[node_id] = node_results

        self.validations = all_results
        return all_results

    def generate_explanations(self) -> dict[str, NodeExplanation]:
        """Generate structured explanations for all nodes."""
        all_results: dict[str, NodeExplanation] = {}

        for node_id, node in self.graph.nodes.items():
            parts = [rule.explain(node, self.context) for rule in self.rules]
            all_results[node_id] = _merge_node_explanations(parts)

        self.explanations = all_results
        return all_results

    def generate_input_baskets(self) -> dict[str, dict[str, list[str]]]:
        """Return a dictionary of node 'baskets' categorized by type."""
        baskets: dict[str, dict[str, list[str]]] = {}
        # Special tags that should be skipped completely
        skip_tags = {'ignore_content'}

        for node_id, node in self.graph.nodes.items():
            baskets[node_id] = {}

            # Categorize nodes by tags
            for input_id in self.graph.inputs.get(node_id, []):
                basket = 'unknown'
                input_node = self.graph.nodes[input_id]
                edge = self.graph.edges.get((input_id, node_id))
                edge_tags = edge.tags if edge is not None else ()
                node_tags = input_node.tags
                assigned = False
                if any(tag in node_tags or tag in edge_tags for tag in skip_tags):
                    basket = 'skip'
                    assigned = True
                else:
                    for tag, basket in TAG_TO_BASKET.items():  # noqa: B007
                        if tag in node_tags or tag in edge_tags:
                            assigned = True
                            break

                if not assigned:
                    node_unit = node.unit  # FIXME Does not Work with multi-metric nodes.
                    input_unit = input_node.unit
                    if node_unit is None or input_unit is None:
                        basket = 'unknown'
                    else:
                        n_dim = unit_registry.parse_units(node_unit).dimensionality
                        i_dim = unit_registry.parse_units(input_unit).dimensionality
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
            any(rule.level == 'error' and not rule.is_valid for rule in node_rules) for node_rules in validation_results.values()
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


def build_node_explanation_system(context: Context, nodes: Sequence[ExplainedNode]) -> NodeExplanationSystem:
    """Construct the explanation system and run its generation passes."""
    nes = NodeExplanationSystem(context, nodes)
    # The rules reach back through context.node_explanation_system, so it must
    # be assigned before the generation passes run.
    context.node_explanation_system = nes
    nes.generate_validations()
    nes.generate_input_baskets()
    nes.generate_explanations()
    return nes


class GraphValidator:
    @staticmethod
    def validate_graph(graph: GraphRepresentation) -> list[ValidationResult]:
        """Validate the complete graph for structural issues."""
        results = []

        # Check for missing node references
        for from_node, to_node in graph.edges.keys():
            if from_node not in graph.nodes:
                results.append(
                    ValidationResult(
                        method='missing_node_test',
                        is_valid=False,
                        level='error',
                        message=f"Input node reference exists for '{from_node}' but node is missing.",
                    )
                )

            if to_node not in graph.nodes:
                results.append(
                    ValidationResult(
                        method='missing_node_test',
                        is_valid=False,
                        level='error',
                        message=f"Output node reference exists for '{from_node}' but node is missing.",
                    )
                )

        # Check for circular dependencies
        if GraphValidator._has_cycles(graph):
            results.append(
                ValidationResult(
                    method='cyclic_graph_test', is_valid=False, level='error', message='Graph contains circular dependencies'
                )
            )

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
    def explain(self, node: ExplainedNode, context: Context) -> NodeExplanation:
        """Generate structured explanation from the node's typed description."""

    @abstractmethod
    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        """Validate the node's typed description."""

    def get_param(self, node: ExplainedNode, param_id: str) -> str:
        param = node.param(param_id)
        if param is None:
            return ''
        return f'{param.value} {param.unit}'

    def get_all_params(self, node: ExplainedNode, drop: list[str]) -> list[list[str] | None]:
        out: list[list[str] | None] = []
        for param in node.params:
            if param.id in drop:
                continue
            v = param.value or _('referencing to <i>%s</i>') % param.ref
            out.append([param.id, f'{_format_explanation_value(v)} {param.unit}'])
        return out


class NodeClassRule(ValidationRule):
    def explain(self, node: ExplainedNode, context: Context) -> NodeExplanation:
        typ = node.class_name
        desc = NODE_CLASS_DESCRIPTIONS.get(typ) or NODE_CLASS_DESCRIPTIONS['Unknown']
        operations = self.get_param(node, 'operations') or None
        other = self.get_all_params(node, drop=['operations', 'formula'])
        params: list[tuple[str, str]] = []
        for p in other:
            assert p is not None
            params.append((str(p[0]), str(p[1])))
        return NodeExplanation(
            node_id=node.id,
            node_type=typ,
            description=str(desc.description),
            operations=operations,
            params=params,
        )

    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        typ = node.class_name
        if typ:
            if typ not in NODE_CLASS_DESCRIPTIONS.keys():
                results.append(
                    ValidationResult(
                        method='node_class_rule',
                        is_valid=False,
                        level='warning',
                        message=f'Node class {typ} does not have a description.',
                    )
                )

            elif NODE_CLASS_DESCRIPTIONS[typ].deprecated:
                results.append(
                    ValidationResult(
                        method='node_depreciation_rule',
                        is_valid=False,
                        level='warning',
                        message=f'Node class {typ} is depreciated.',
                    )
                )

        return results


def dataset_pipeline(dataset: InputDatasetDef) -> list[PortTransformOp]:
    """Return the binding pipeline; a definition still carrying the YAML-era flat fields compiles them first."""
    if dataset.transformations is not None:
        return list(dataset.transformations)
    return dataset.to_transformations()


def _forecast_from(pipeline: Sequence[PortTransformOp]) -> int | None:
    year: int | None = None
    for op in pipeline:
        if isinstance(op, SetForecastFromOp):
            year = op.year
    return year


def _drops_nulls(pipeline: Sequence[PortTransformOp]) -> bool:
    return any(isinstance(op, DropNullsOp) for op in pipeline)


class DatasetRule(ValidationRule):
    def explain(self, node: ExplainedNode, context: Context) -> NodeExplanation:
        # Terms (including datasets) are handled by BasketRule to keep inputs in one place.
        return NodeExplanation()

    def explain_pipeline(self, pipeline: Sequence[PortTransformOp], context: Context) -> list[str]:
        """Describe the column renames and the filters of a dataset pipeline, in execution order."""
        html: list[str] = []
        renames = [op for op in pipeline if isinstance(op, RenameColumnOp)]
        if renames:
            html.append(f'<li>{_("Renames the following columns:")}<ul>')
            html.extend(f'<li>{rename.column} &rarr; {rename.new_name or ""}.</li>' for rename in renames)
            html.append('</ul></li>')
        filters = [op for op in pipeline if isinstance(op, (FilterColumnOp, FilterDimensionOp, AssignDimensionOp, RenameItemOp))]
        if filters:
            html.append(f'<li>{_("Has the following filters:")}<ol>')
            for op in filters:
                match op:
                    case FilterColumnOp():
                        html.append(self._explain_column_filter(op, context))
                    case FilterDimensionOp():
                        html.append(self._explain_dim_filter(op, context))
                    case AssignDimensionOp():
                        html.append(self._explain_assign(op, context))
                    case RenameItemOp():
                        html.append(self._explain_rename_item(op))
            html.append('</ol></li>')
        return html

    def _explain_column_filter(self, op: FilterColumnOp, context: Context) -> str:
        vals = list(op.values)
        if op.value:
            vals.append(op.value)
        if op.ref:
            param = context.global_parameters[op.ref]
            label = param.label
            if isinstance(label, dict):
                label = next(iter(label.values()), '')
            vals.append(_('global parameter %(label)s') % {'label': str(label)})
        if ''.join(vals):
            if op.exclude:
                text = _('Filter column <i>%(name)s</i> by excluding <i>%(values)s</i>.') % {
                    'name': op.column,
                    'values': ', '.join(vals),
                }
            else:
                text = _('Filter column <i>%(name)s</i> by including <i>%(values)s</i>.') % {
                    'name': op.column,
                    'values': ', '.join(vals),
                }
            out = f'<li>{text}</li>'
        else:
            out = ''
        if op.flatten:
            out += f'<li>{_("Sum up column <i>%(name)s</i>.") % {"name": op.column}}</li>'
        elif op.drop_col:
            out += f'<li>{_("Drop column <i>%(name)s</i>.") % {"name": op.column}}</li>'
        return out

    def _explain_dim_filter(self, op: FilterDimensionOp, context: Context) -> str:
        dim = context.dimensions[op.dimension]
        if op.groups:
            items = [str(group.label) for group in dim.groups if group.id in op.groups]
        elif op.categories:
            items = [str(cat.label) for cat in dim.categories if cat.id in op.categories]
        else:
            items = []
        if items:
            text = _('Filter dimension <i>%(dim_label)s</i> by categories <i>%(cat_labels)s</i>.') % {
                'dim_label': dim.label,
                'cat_labels': ', '.join(items),
            }
            out = f'<li>{text}</li>'
        else:
            out = ''
        if op.flatten:
            out += f'<li>{_("Sum up the dimension <i>%(label)s</i>.") % {"label": dim.label}}</li>'
        return out

    def _explain_assign(self, op: AssignDimensionOp, context: Context) -> str:
        dim = context.dimensions[op.dimension]
        cat_label = next(str(cat.label) for cat in dim.categories if cat.id == op.category)
        text = _('Assign dataset to category <i>%(cat_label)s</i> on dimension <i>%(dim_label)s</i>.') % {
            'cat_label': cat_label,
            'dim_label': dim.label,
        }
        return f'<li>{text}</li>'

    def _explain_rename_item(self, op: RenameItemOp) -> str:
        return _('Rename item <i>%(old_string)s</i> to <i>%(new_string)s</i> in column <i>%(column)s</i>.') % {
            'old_string': op.old_item,
            'new_string': op.new_item,
            'column': op.column,
        }

    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for index, dataset in enumerate(node.datasets):
            results.extend(self._validate_single_dataset(dataset, index))
        return results

    def _validate_single_dataset(self, dataset: InputDatasetDef, index: int) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        if dataset.column is not None and not dataset.column.strip():
            results.append(
                ValidationResult(
                    method='dataset_column_check',
                    is_valid=False,
                    level='error',
                    message=f'Dataset {index} is missing column: {dataset.column}',
                )
            )
        year = _forecast_from(dataset_pipeline(dataset))
        if year is not None and (year < 1900 or year > 2100):
            results.append(
                ValidationResult(
                    method='dataset_forecast_year_check',
                    is_valid=False,
                    level='warning',
                    message=f'Dataset {index} has questionable forecast year: {year}',
                )
            )
        return results


class EdgeRule(ValidationRule):
    def explain(self, node: ExplainedNode, context: Context) -> NodeExplanation:
        # Terms are handled by BasketRule to keep inputs in one place.
        return NodeExplanation()

    def get_explanation_for_tag(self, tags: Sequence[str], skip_tags: set[str] | None = None) -> list[str]:
        html: list[str] = []
        for tag in tags:
            if skip_tags and tag in skip_tags:
                continue
            if tag in TAG_TO_BASKET.keys():  # These show up in basket explanations
                continue
            description = TAG_DESCRIPTIONS.get(tag, _('Has tag <i>%s</i>.') % tag)
            html.append(f'<li>{description}</li>')
        return html

    def get_explanation_for_edge_from(self, edge: ExplainedEdge, context: Context) -> list[str]:
        edge_html: list[str] = []
        for op in edge.transformations:
            if not isinstance(op, FilterDimensionOp):
                continue
            dimlabel = str(context.dimensions[op.dimension].label)
            cats = list(op.categories)

            if cats:
                category_dict = {cat.id: cat for cat in context.dimensions[op.dimension].categories}
                cats_str = ', '.join([str(category_dict[c].label) for c in cats])
                if op.exclude:
                    text = _('From dimension <i>%(dimension)s</i>, exclude categories: <i>%(categories)s</i>') % {
                        'dimension': dimlabel,
                        'categories': cats_str,
                    }
                else:
                    text = _('In dimension <i>%(dimension)s</i>, include categories: <i>%(categories)s</i>') % {
                        'dimension': dimlabel,
                        'categories': cats_str,
                    }
                edge_html.append(f'<li>{text}</li>')

            if op.flatten:
                edge_html.append(_('<li>Sum over dimension <i>%(dim)s</i></li>') % {'dim': dimlabel})
        return edge_html

    def get_explanation_for_edge_to(self, edge: ExplainedEdge, context: Context) -> list[str]:
        edge_html: list[str] = []
        for op in edge.transformations:
            if not isinstance(op, AssignDimensionOp):
                continue
            dimension = context.dimensions[op.dimension]
            category_dict = {cat.id: cat for cat in dimension.categories}
            text = _('Categorize the values to <i>%(categories)s</i> in a new dimension <i>%(dimension)s</i>.') % {
                'categories': str(category_dict[op.category].label),
                'dimension': str(dimension.label),
            }
            edge_html.append(f'<li>{text}</li>')
        return edge_html

    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        return [
            ValidationResult(method='edge_rule', is_valid=True, level='info', message='There is no validation rule for edges.')
        ]


class BasketRule(ValidationRule):
    def explain(self, node: ExplainedNode, context: Context) -> NodeExplanation:
        node_id = node.id

        nes = context.node_explanation_system
        assert nes is not None
        baskets = nes.baskets[node_id]
        operation_list = self.get_param(node, 'operations')
        if not operation_list:
            operation_list = context.nodes[node_id].DEFAULT_OPERATIONS
        operations = [o.strip() for o in operation_list.split(',')]
        raw_terms = self._collect_terms(node, context)
        formula = self._build_formula_from_config(node, operations, baskets, raw_terms)
        if not formula and not raw_terms:
            return NodeExplanation()

        term_infos = [
            TermInfo(
                label=str(t['label']),
                kind=t['kind'],
                name=str(t['name']),
                unit=t.get('unit'),
                value=t.get('value'),
                output_dimensions=t.get('output_dimensions'),
                details=t.get('details', []),
            )
            for t in raw_terms
        ]

        has_dataset_terms = any(t['kind'] == 'dataset' for t in raw_terms)
        has_impute_inputs = bool(baskets.get('impute'))
        filtered_ops = [
            op
            for op in operations
            if not (op == 'get_single_dataset' and not has_dataset_terms) and not (op == 'impute' and not has_impute_inputs)
        ]
        functions = self._collect_functions(filtered_ops, raw_terms, node)

        remaining_baskets = [b for b in baskets if b not in operations and b != 'skip']
        leftover: list[str] = []
        if remaining_baskets:
            leftover.append(_('These groups are left over without an operation:') + '<ol>')
            for basket in remaining_baskets:
                input_nodes = baskets.get(basket, [])
                basket_display = BASKET_DISPLAY_NAMES.get(basket, basket)
                nodes_str = '</li><li>'.join(input_nodes)
                leftover.append(f'<li>{_("Group %(basket)s with nodes:") % {"basket": basket_display}}<ul>')
                leftover.append(f'<li>{nodes_str}</li></ul></li>')
            leftover.append('</ol>')

        return NodeExplanation(
            formula=formula or None,
            terms=term_infos,
            functions=functions,
            leftover_html=leftover,
        )

    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        return [
            ValidationResult(
                method='basket_rule', is_valid=True, level='info', message='There is no validation rule for baskets.'
            )
        ]
        # Each input node must belong to some basket
        # If an input node belongs to the unknown basket, give a warning

    def _collect_terms(  # noqa: C901, PLR0912
        self,  # FIXME Somehow does not show edge function list correctly
        node: ExplainedNode,
        context: Context,
    ) -> list[dict[str, Any]]:
        terms: list[dict[str, Any]] = []
        nes = context.node_explanation_system
        assert nes is not None

        for edge in node.inputs:
            if 'ignore_content' in edge.tags:
                continue
            input_id = edge.source_id
            input_node = nes.graph.nodes.get(input_id)
            func_tags = [tag for tag in edge.tags if tag in TAG_DESCRIPTIONS and tag not in TAG_TO_BASKET]
            label_tag = next((tag for tag in edge.tags if tag not in TAG_TO_BASKET and tag not in TAG_DESCRIPTIONS), None)
            if not func_tags or label_tag is None:
                node_tags = input_node.tags if input_node is not None else ()
                if not func_tags:
                    func_tags = [tag for tag in node_tags if tag in TAG_DESCRIPTIONS and tag not in TAG_TO_BASKET]
                if label_tag is None:
                    label_tag = next(
                        (tag for tag in node_tags if tag not in TAG_TO_BASKET and tag not in TAG_DESCRIPTIONS),
                        None,
                    )
            output_dimensions: list[str] | None = None
            if input_node is not None and input_node.output_dimensions is not None:
                output_dimensions = list(input_node.output_dimensions)
            # The delivered shape: the source's dimensions, minus what the edge
            # sums over, plus what it filters on, declares or assigns.
            filter_ops = [op for op in edge.transformations if isinstance(op, FilterDimensionOp)]
            assigned = [op.dimension for op in edge.transformations if isinstance(op, AssignDimensionOp)]
            adjusted_dims: list[str] | None = None
            if output_dimensions is not None or filter_ops or edge.declared_dimensions or assigned:
                dim_ids = list(output_dimensions or [])
                for op in filter_ops:
                    if op.flatten:
                        if op.dimension in dim_ids:
                            dim_ids.remove(op.dimension)
                        continue
                    if op.dimension not in dim_ids:
                        dim_ids.append(op.dimension)
                for dim_id in [*edge.declared_dimensions, *assigned]:
                    if dim_id not in dim_ids:
                        dim_ids.append(dim_id)
                adjusted_dims = dim_ids

            term = {
                'kind': 'node',
                'key': input_id,
                'label': label_tag,
                'name': context.nodes[input_id].name,
                'var_names': self._term_var_names(edge),
                'unit': input_node.unit if input_node is not None else None,
                'output_dimensions': adjusted_dims if adjusted_dims is not None else output_dimensions,
                'functions': func_tags,
                'details': self._node_term_details(edge, context, label_tag),
            }
            terms.append(term)

        for dataset in node.datasets:
            # Binding-level output_dimensions was retired; the node's declared
            # dimensions are the only source.
            ds_output_dimensions = list(node.output_dimensions) if node.output_dimensions is not None else None
            ds_unit = str(dataset.unit) if dataset.unit is not None else node.unit
            tags = [tag for tag in dataset.tags if tag != 'cleaned']
            label_tag = tags[0] if tags else None
            term = {
                'kind': 'dataset',
                'key': dataset.id,
                'label': label_tag,
                'name': dataset.id,
                'var_names': self._dataset_var_names(dataset),
                'unit': ds_unit,
                'output_dimensions': ds_output_dimensions,
                'functions': [],
                'details': self._dataset_term_details(dataset, context),
            }
            terms.append(term)

        for param in node.params:
            if param.id in ['formula', 'operations']:
                continue
            if param.value is not None:
                term = {
                    'kind': 'constant',
                    'key': param.id,
                    'label': param.id,
                    'name': _('Constant'),
                    'var_names': [param.id],
                    'unit': param.unit or None,
                    'value': param.value,
                    'functions': [],
                    'details': [],
                }
                terms.append(term)

        counter = 1
        for term in terms:
            if not term['label']:
                term['label'] = f't{counter}'
                counter += 1

        return terms

    def _collect_functions(  # noqa: C901
        self,
        operations: list[str],
        terms: list[dict[str, Any]],
        node: ExplainedNode,
    ) -> list[str]:
        functions: list[str] = []
        ops_seen: set[str] = set()
        formula_param = self.get_param(node, 'formula')
        if formula_param:
            for func in self._extract_formula_functions(formula_param):
                if func in ops_seen:
                    continue
                description = TAG_DESCRIPTIONS.get(func)
                if description:
                    functions.append(f'<li><b>{func}</b>: {description}</li>')
                    ops_seen.add(func)
        for op in operations:
            if op in ops_seen:
                continue
            description = TAG_DESCRIPTIONS.get(op)
            if description:
                functions.append(f'<li><b>{op}</b>: {description}</li>')
                ops_seen.add(op)
        for term in terms:
            for func in term.get('functions', []):
                if func in ops_seen:
                    continue
                description = TAG_DESCRIPTIONS.get(func)
                if description:
                    functions.append(f'<li><b>{func}</b>: {description}</li>')
                    ops_seen.add(func)
        return functions

    def _extract_formula_identifiers(self, formula: str) -> set[str]:
        import ast

        try:
            tree = ast.parse(formula, '<string>', mode='eval')
        except SyntaxError:
            return set()
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _extract_formula_functions(self, formula: str) -> set[str]:
        import ast

        try:
            tree = ast.parse(formula, '<string>', mode='eval')
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
                # Nest from the inside out so the last-applied function ends up outermost,
                # matching the actual execution order in Node._process_edge_output.
                wrapped = var
                for func in funcs:
                    wrapped = f'{func}({wrapped})'
                if wrapped in updated:
                    continue
                updated = re.sub(rf'\b{re.escape(var)}\b', wrapped, updated)
        return updated

    def _term_var_names(self, edge: ExplainedEdge) -> list[str]:
        tags = [tag for tag in edge.tags if tag not in TAG_TO_BASKET and tag not in TAG_DESCRIPTIONS]
        if tags:
            return tags
        return [edge.source_id]

    def _dataset_var_names(self, dataset: InputDatasetDef) -> list[str]:
        tags = [tag for tag in dataset.tags if tag != 'cleaned']
        if tags:
            return tags
        return [dataset.id]

    def _build_formula_from_config(  # noqa: C901
        self,
        node: ExplainedNode,
        operations: list[str],
        baskets: dict[str, list[str]],
        terms: list[dict[str, Any]],
    ) -> str:
        formula_param = self.get_param(node, 'formula')
        label_by_id: dict[str, str] = {}
        for term in terms:
            label = str(term['label'])
            for func in term.get('functions', []):
                label = f'{func}({label})'
            label_by_id[term['key']] = label
        if formula_param:
            import re as _re

            formula_param = self._apply_term_functions(formula_param, terms)
            # Replace each term's identifiers with its display label (t1, t2, or explicit tag)
            for term in terms:
                label = str(term['label'])
                vars_to_sub: set[str] = set()
                for var in term.get('var_names') or []:
                    vars_to_sub.add(var)
                    vars_to_sub.add(make_identifier(var))
                for var in sorted(vars_to_sub, key=len, reverse=True):
                    formula_param = _re.sub(rf'\b{_re.escape(var)}\b', label, formula_param)
            used_names = self._extract_formula_identifiers(formula_param)
            unused_labels = [
                str(term['label']) for term in terms if term['kind'] == 'node' and str(term['label']) not in used_names
            ]
            if unused_labels:
                return f'({formula_param} + {" + ".join(unused_labels)})'
            return formula_param
        if node.class_name == 'AdditiveNode':
            add_terms = [term['label'] for term in terms if term['kind'] != 'constant']
            if not add_terms:
                return ''
            return f'({BASKET_OPERATION_LABEL["add"].join(add_terms)})'
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

    def _render_operation(self, operation: str, terms: list[str]) -> str:
        op_label = BASKET_OPERATION_LABEL.get(operation, ' + ')
        no_name = ['add', 'multiply']
        op_name = '' if operation in no_name else BASKET_DISPLAY_NAMES.get(operation, operation)
        joined = op_label.join(terms)
        if op_name:
            return f'{op_name}({joined})'
        return f'({joined})'

    def _node_term_details(
        self,
        edge: ExplainedEdge,
        context: Context,
        label_tag: str | None,
    ) -> list[str]:
        details: list[str] = []
        if edge.metrics:
            metrics_str = ', '.join(edge.metrics)
            details.append(f'<li>{_("Metrics: %(metrics)s") % {"metrics": metrics_str}}</li>')
        func_tags = [tag for tag in edge.tags if tag in TAG_DESCRIPTIONS and tag not in TAG_TO_BASKET]
        details.extend(
            EdgeRule().get_explanation_for_tag(
                edge.tags,
                skip_tags=set(filter(None, [label_tag, *func_tags])) or None,
            )
        )
        details.extend(EdgeRule().get_explanation_for_edge_from(edge, context))
        details.extend(EdgeRule().get_explanation_for_edge_to(edge, context))
        return details

    def _dataset_term_details(self, dataset: InputDatasetDef, context: Context) -> list[str]:
        html: list[str] = []
        if dataset.column is not None:
            text = _('Metric: %(name)s') % {'name': dataset.column}
            html.append(f'<li>{text}</li>')
        pipeline = dataset_pipeline(dataset)
        year = _forecast_from(pipeline)
        if year is not None:
            text = _('Has forecast values from: %(year)s') % {'year': year}
            html.append(f'<li>{text}</li>')
        if _drops_nulls(pipeline):
            html.append(f'<li>{_("Rows with missing values are dropped.")}</li>')
        html.extend(DatasetRule().explain_pipeline(pipeline, context))
        return html


class FormulaValidationMixin(ValidationRule, ABC):
    def explain(self, _node: ExplainedNode, _context: Context) -> NodeExplanation:
        return NodeExplanation()

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

    def _build_formula_spec(  # noqa: C901
        self,
        node: ExplainedNode,
        context: Context,
        operations: list[str],
    ) -> FormulaSpec:
        node_id = node.id
        nes = context.node_explanation_system
        assert nes is not None

        basket_rule = BasketRule()
        terms = basket_rule._collect_terms(node, context)
        baskets = nes.baskets.get(node_id, {})

        display_expression = basket_rule._build_formula_from_config(
            node,
            operations,
            baskets,
            terms,
        )

        formula_param = self.get_param(node, 'formula')
        label_by_id: dict[str, str] = {}
        for term in terms:
            label = term['key']
            for func in term.get('functions', []):
                label = f'{func}({label})'
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
                expression = f'({formula_param} + {" + ".join(unused_labels)})'
            else:
                expression = formula_param
        elif node.class_name == 'AdditiveNode':
            add_terms = [label_by_id[term['key']] for term in terms if term['kind'] != 'constant' and term['key'] in label_by_id]
            expression = f'({BASKET_OPERATION_LABEL["add"].join(add_terms)})' if add_terms else ''
        elif not operations:
            expression = ''
        else:
            expression = basket_rule._build_formula_from_operations(
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


class FormulaDimensionRule(FormulaValidationMixin):
    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        nes = context.node_explanation_system
        if nes is None:
            return results
        graph = self._ensure_baskets(context)
        operations = self._get_operations(node.id, context, graph)
        spec = self._build_formula_spec(node, context, operations)
        if not spec.expression:
            return results

        results.extend(self._validate_dataset_dims(node, spec))
        results.extend(self._validate_formula_dims(spec))
        results.extend(self._validate_output_dims(node, spec))
        return results

    def _validate_dataset_dims(
        self,
        node: ExplainedNode,
        spec: FormulaSpec,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        used_names = BasketRule()._extract_formula_identifiers(spec.expression)
        node_output_dimensions = node.output_dimensions
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
                label_str = f" '{label}'" if label else ''
                results.append(
                    ValidationResult(
                        method='formula_dimension_rule',
                        is_valid=False,
                        level='info',
                        message=(
                            f'Dataset term{label_str} is missing output_dimensions; dimension validation may be incomplete.'
                        ),
                    )
                )
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
        node: ExplainedNode,
        spec: FormulaSpec,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        expected_dims = set(node.output_dimensions or ())
        analysis = analyze_formula_dimensions(
            spec.expression,
            build_name_dimension_map(spec.terms)[0],
            passthrough_functions=set(TAG_DESCRIPTIONS.keys()),
        )
        if analysis.dims is not None and expected_dims and analysis.dims != expected_dims:
            results.append(
                ValidationResult(
                    method='formula_dimension_rule',
                    is_valid=False,
                    level='error',
                    message=(
                        'Formula output dimensions do not match node.output_dimensions: '
                        f'{sorted(analysis.dims)} vs {sorted(expected_dims)}. '
                        f'Expression: {spec.expression}'
                    ),
                )
            )
        return results


class FormulaUnitRule(FormulaValidationMixin):
    def validate(self, node: ExplainedNode, context: Context) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        node_unit = node.unit
        if not node_unit:
            return results
        nes = context.node_explanation_system
        if nes is None:
            return results
        graph = self._ensure_baskets(context)
        operations = self._get_operations(node.id, context, graph)
        spec = self._build_formula_spec(node, context, operations)
        if not spec.expression:
            return results

        name_units = build_name_unit_map(spec.terms)
        results.extend(self._apply_multiplier_unit_inference(node, operations, spec, name_units))

        unit_analysis = analyze_formula_units(
            spec.expression,
            name_units,
            passthrough_functions=set(TAG_DESCRIPTIONS.keys()),
            unit_overrides=cast('dict[str, UnitOverride]', FORMULA_FUNCTION_UNIT_OVERRIDES),
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
                results.append(
                    ValidationResult(
                        method='formula_unit_rule',
                        is_valid=False,
                        level='error',
                        message=(f'Formula output unit does not match node.unit: {unit_analysis.unit} vs {expected_unit}'),
                    )
                )
            elif unit_analysis.unit != expected_unit:
                results.append(
                    ValidationResult(
                        method='formula_unit_rule',
                        is_valid=True,
                        level='info',
                        message=(
                            'Formula output unit differs from node.unit but is compatible: '
                            f'{unit_analysis.unit} vs {expected_unit}'
                        ),
                    )
                )
        return results

    def _apply_multiplier_unit_inference(
        self,
        node: ExplainedNode,
        operations: list[str],
        spec: FormulaSpec,
        name_units: dict[str, Unit | None],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        multiplier = node.param('multiplier')
        multiplier_unit = multiplier.unit if multiplier is not None else None
        if not multiplier_unit or 'apply_multiplier' not in operations or node.unit is None:
            return results

        explicit_ds_units = {ds.id for ds in node.datasets if ds.unit is not None}
        inferred_unit = cast('Unit', unit_registry.parse_units(node.unit) / unit_registry.parse_units(multiplier_unit))
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
            results.append(
                ValidationResult(
                    method='formula_unit_rule',
                    is_valid=True,
                    level='warning',
                    message=('Dataset unit inferred from node.unit and multiplier; consider setting dataset.unit explicitly.'),
                )
            )
        return results
