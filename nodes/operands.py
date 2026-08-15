"""
Deciding what each of a node's inputs is *for*.

Several node classes answer the same question before they can compute anything: of
everything wired into this node, which values get **added**, which are **factors** in a
product, which are **imputed** over the result afterwards, and which are none of those
because some other operation has claimed them?

They answer it with the same rule, written out four times —
``GenericNode._get_add_multiply_nodes``, ``MultiplicativeNode._compute``, and the
``infer_legacy_port_roles`` classmethods on both simple classes. This module is that rule,
written once:

1. an explicit tag decides — ``additive``, ``non_additive``, ``impute``;
2. a tag belonging to some other operation (``use_as_totals``, ``split_dims``, …) means the
   input is not ours to combine;
3. otherwise the unit decides — compatible with the node's own unit means additive,
   incompatible means a factor.

The rule is deliberately indifferent to whether the input arrived as a node or as a
dataset: by the time arithmetic happens both are a ``PathsDataFrame``. See
``docs/plans/additive-multiplicative-modernization.md``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN

from .exceptions import NodeError
from .explanations import TAG_TO_BASKET

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Container

    from common.polars import PathsDataFrame
    from nodes.datasets import Dataset
    from nodes.node import Node
    from nodes.units import Unit

type OperandRole = Literal['additive', 'factor', 'impute']

ADDITIVE_TAG = 'additive'
FACTOR_TAG = 'non_additive'
IMPUTE_TAG = 'impute'
IGNORE_TAG = 'ignore_content'

#: Tags that name a role outright, in the order they are checked. The arithmetic tags come
#: first because that is the precedence ``GenericNode`` has always applied; an input tagged
#: both ``additive`` and ``impute`` is therefore added, not imputed. No config does that
#: today (all 18 uses of ``impute`` are bare), so the ordering is a tie-break nobody hits —
#: but it is a tie-break the v2 classes should refuse outright rather than silently resolve.
ROLE_TAGS: dict[str, OperandRole] = {
    ADDITIVE_TAG: 'additive',
    FACTOR_TAG: 'factor',
    IMPUTE_TAG: 'impute',
}


def role_from_tags(tags: Container[str]) -> OperandRole | None:
    """Return the role named outright by a tag, or None when the tags say nothing about it."""
    for tag, role in ROLE_TAGS.items():
        if tag in tags:
            return role
    return None


def claimed_by_other_operation(tags: Collection[str]) -> bool:
    """
    Return True when a tag hands this input to an operation other than add/multiply/impute.

    ``TAG_TO_BASKET`` is the full tag vocabulary; the roles this module assigns are the
    part of it that add/multiply nodes handle themselves. Anything else — ``use_as_totals``,
    ``split_dims``, ``skip_dim_test`` and friends — belongs to an operation that resolves
    its own inputs, so the input must not also be swept into a sum or a product.
    """
    return any(tag in TAG_TO_BASKET and tag not in ROLE_TAGS for tag in tags)


@dataclass
class NodeOperands:
    """A node's input nodes, split by what they are for. Input order is preserved."""

    additive: list[Node] = field(default_factory=list)
    factors: list[Node] = field(default_factory=list)
    impute: list[Node] = field(default_factory=list)
    claimed_elsewhere: list[Node] = field(default_factory=list)
    """Inputs a tag handed to another operation. Listed so a caller can refuse them
    rather than drop them silently."""


def output_unit_of(target: Node, source: Node) -> Unit | None:
    """Return the unit a source node actually delivers to ``target`` (what ``GenericNode`` uses)."""
    return source.get_output_pl(target_node=target).get_unit(VALUE_COLUMN)


def declared_unit_of(_target: Node, source: Node) -> Unit | None:
    """Return the unit a source node declares, uncomputed (what ``MultiplicativeNode`` uses)."""
    return source.unit


def resolve_input_nodes(
    node: Node,
    *,
    exclude_ids: Container[str] = frozenset(),
    unit_of: Callable[[Node, Node], Unit | None] = output_unit_of,
    default_role: OperandRole | None = None,
) -> NodeOperands:
    """
    Split ``node``'s input nodes into additive inputs, factors and imputed overlays.

    ``exclude_ids`` drops inputs a caller has already consumed by other means (weighted
    sums do this). ``unit_of`` chooses whether the unit test reads the source's computed
    output or its declared unit — computing the output is authoritative but costs a
    compute; the declared unit is free but can disagree with reality.

    ``default_role`` decides untagged inputs whose unit cannot be read at all. Left as
    None, such an input raises, because guessing here silently mis-sorts data into the
    wrong arithmetic.
    """
    operands = NodeOperands()
    for edge in node.edges:
        if edge.output_node is not node:
            continue
        source = edge.input_node
        tags = set(edge.tags) | set(source.tags)
        if IGNORE_TAG in tags or source.id in exclude_ids:
            continue

        role = role_from_tags(tags)
        if role is None and claimed_by_other_operation(tags):
            operands.claimed_elsewhere.append(source)
            continue
        if role is None:
            role = _role_from_unit(node, source, unit_of=unit_of, default_role=default_role)

        bucket = {'additive': operands.additive, 'factor': operands.factors, 'impute': operands.impute}[role]
        bucket.append(source)
    return operands


def _role_from_unit(
    node: Node,
    source: Node,
    *,
    unit_of: Callable[[Node, Node], Unit | None],
    default_role: OperandRole | None,
) -> OperandRole:
    source_unit = unit_of(node, source)
    if source_unit is None or node.unit is None:
        if default_role is not None:
            return default_role
        raise NodeError(
            node,
            "Cannot tell whether input '%s' is additive or a factor: %s has no unit. "
            "Tag the input 'additive' or 'non_additive' to say which it is."
            % (source.id, 'it' if source_unit is None else 'this node'),
        )
    return 'additive' if node.is_compatible_unit(node.unit, source_unit) else 'factor'


# =================================================================================
# Materialised operands: a node input and a dataset input, reduced to the same thing.
# =================================================================================


@dataclass(frozen=True)
class Operand:
    """One input to a node, as a frame plus enough provenance to name it in an error."""

    df: PathsDataFrame
    role: OperandRole
    source_id: str
    kind: Literal['node', 'dataset']

    def __str__(self) -> str:
        return f'{self.kind} {self.source_id}'


@dataclass
class OperandSet:
    """Everything wired into a node, materialised and split by role."""

    additive: list[Operand] = field(default_factory=list)
    factors: list[Operand] = field(default_factory=list)
    impute: list[Operand] = field(default_factory=list)
    claimed_elsewhere: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    """Input nodes that failed or are INCOMPLETE, skipped because the context tolerates
    node failures. See ``docs/architecture/fault-tolerance.md``."""

    @property
    def is_empty(self) -> bool:
        return not (self.additive or self.factors)


def dataset_metric_column(node: Node, df: PathsDataFrame, metric: str | None) -> str:
    """
    Pick the column of a dataset that carries this node's values.

    The single metric column if there is only one; else the one named by the ``metric``
    parameter; else the only column whose unit is compatible with the node's own — which
    is a guess that happens to be right for an additive input and cannot be made at all
    for a factor, so an ambiguous factor dataset has to name its metric.
    """
    if VALUE_COLUMN in df.metric_cols:
        return VALUE_COLUMN
    if len(df.metric_cols) == 1:
        return df.metric_cols[0]
    if metric is not None:
        if metric not in df.columns:
            raise NodeError(node, "Metric '%s' is not a column of the input dataset" % metric)
        return metric
    assert node.unit is not None
    compatible = [col for col in df.metric_cols if node.is_compatible_unit(df.get_unit(col), node.unit)]
    if len(compatible) == 1:
        return compatible[0]
    raise NodeError(
        node,
        "Input dataset has %d metric columns (%s) and no 'metric' parameter to choose between them"
        % (len(df.metric_cols), ', '.join(df.metric_cols)),
    )


def dataset_operand_frame(node: Node, dataset: Dataset, metric: str | None) -> PathsDataFrame:
    """
    Reduce a dataset to the shape a node operand has: one ``Value`` column and a forecast flag.

    Deliberately *not* ``get_cleaned_dataset``: nothing is dropped, back-filled or extended
    here. Whatever shaping the data needs is the dataset binding's business (``interpolate``,
    ``extend``, and an explicit ``filters: - column: <dim>`` to drop a dimension the selected
    metric does not use), so that a dataset means the same thing wherever it is bound.
    """
    df = dataset.get_copy()
    col = dataset_metric_column(node, df, metric)
    if col != VALUE_COLUMN:
        df = df.rename({col: VALUE_COLUMN})
    keep = [YEAR_COLUMN, *df.dim_ids, VALUE_COLUMN]
    if FORECAST_COLUMN in df.columns:
        keep.append(FORECAST_COLUMN)
    else:
        df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        keep.append(FORECAST_COLUMN)
    return df.select(keep)


def resolve_operands(
    node: Node,
    *,
    metric: str | None = None,
    exclude_ids: Container[str] = frozenset(),
    unit_of: Callable[[Node, Node], Unit | None] = output_unit_of,
    default_role: OperandRole | None = None,
) -> OperandSet:
    """
    Materialise every input of ``node`` — nodes and datasets alike — and split by role.

    Datasets are classified by exactly the rule that classifies nodes: a tag if there is
    one, otherwise unit compatibility with the node's own unit.

    When the context tolerates node failures, an input node that raised or is INCOMPLETE
    is recorded in ``unavailable`` instead of propagating.
    """
    operands = OperandSet()
    node_roles = resolve_input_nodes(node, exclude_ids=exclude_ids, unit_of=unit_of, default_role=default_role)
    operands.claimed_elsewhere = [n.id for n in node_roles.claimed_elsewhere]

    roles: tuple[tuple[OperandRole, list[Node]], ...] = (
        ('additive', node_roles.additive),
        ('factor', node_roles.factors),
        ('impute', node_roles.impute),
    )
    for role, sources in roles:
        for source in sources:
            _add_node_operand(node, operands, source, role)

    for dataset in node.input_dataset_instances:
        _add_dataset_operand(node, operands, dataset, metric=metric, default_role=default_role)

    return operands


def _add_node_operand(node: Node, operands: OperandSet, source: Node, role: OperandRole) -> None:
    from nodes.node import NodeStatus

    tolerant = node.context.tolerate_node_failures
    try:
        df = source.get_output_pl(target_node=node)
    except NodeError:
        if not tolerant:
            raise
        operands.unavailable.append(source.id)
        return
    if tolerant and source.status is NodeStatus.INCOMPLETE:
        # An INCOMPLETE upstream produces an empty self-report only; treat it as absent.
        operands.unavailable.append(source.id)
        return
    _bucket(operands, role).append(Operand(df=df, role=role, source_id=source.id, kind='node'))


def _add_dataset_operand(
    node: Node,
    operands: OperandSet,
    dataset: Dataset,
    *,
    metric: str | None,
    default_role: OperandRole | None,
) -> None:
    tagged_role = role_from_tags(dataset.tags)
    if tagged_role is None and claimed_by_other_operation(dataset.tags):
        operands.claimed_elsewhere.append(dataset.id)
        return
    df = dataset_operand_frame(node, dataset, metric)
    role: OperandRole
    if tagged_role is not None:
        role = tagged_role
    elif node.unit is not None:
        role = 'additive' if node.is_compatible_unit(node.unit, df.get_unit(VALUE_COLUMN)) else 'factor'
    elif default_role is not None:
        role = default_role
    else:
        raise NodeError(node, "Cannot classify dataset '%s': this node has no unit" % dataset.id)
    _bucket(operands, role).append(Operand(df=df, role=role, source_id=dataset.id, kind='dataset'))


def _bucket(operands: OperandSet, role: OperandRole) -> list[Operand]:
    return {'additive': operands.additive, 'factor': operands.factors, 'impute': operands.impute}[role]


# =================================================================================
# Combining operands. These are the modern joins — the same ones FormulaNode uses.
# =================================================================================


def sum_operands(node: Node, operands: list[Operand], unit: Unit) -> PathsDataFrame:
    """
    Add operands together. Every one must carry the same dimensions; missing values are zero.

    ``add_with_dims(how='outer')`` fills both an absent row and a null value with zero, so a
    series that starts late contributes nothing to the years it does not cover rather than
    deleting them.
    """
    result: PathsDataFrame | None = None
    for operand in operands:
        df = operand.df.ensure_unit(VALUE_COLUMN, unit)
        if result is None:
            result = df
            continue
        if set(df.dim_ids) != set(result.dim_ids):
            raise NodeError(
                node,
                'Dimensions do not match with %s: %s vs. %s' % (operand, sorted(df.dim_ids), sorted(result.dim_ids)),
            )
        result = result.paths.add_with_dims(df, how='outer')
    assert result is not None
    return result


def multiply_operands(node: Node, operands: list[Operand], unit: Unit) -> PathsDataFrame:
    """
    Multiply operands together: inner join, dimensions are the union, units multiply.

    A row absent from any factor cannot yield a product and is absent from the result. A
    row whose factor value is *null* keeps its place and stays null — the value is unknown,
    which is not the same as the row not existing.
    """
    if len(operands) < 2:
        raise NodeError(
            node,
            'Multiplication needs at least two inputs, got %s' % ([str(operand) for operand in operands] or 'none'),
        )
    result = operands[0].df
    for operand in operands[1:]:
        result = result.paths.multiply_with_dims(operand.df, how='inner')
    return result.ensure_unit(VALUE_COLUMN, unit)


def impute_operands(node: Node, df: PathsDataFrame, operands: list[Operand]) -> PathsDataFrame:
    """
    Overlay imputed operands onto ``df``, in order, each taking priority over what came before.

    Each operand's value wins wherever it has one; ``df``'s survives only where it does not.
    """
    for operand in operands:
        if set(df.dim_ids) != set(operand.df.dim_ids):
            raise NodeError(
                node,
                'Dimensions must match for imputing: %s vs %s (%s)' % (sorted(df.dim_ids), sorted(operand.df.dim_ids), operand),
            )
        odf = operand.df.ensure_unit(VALUE_COLUMN, df.get_unit(VALUE_COLUMN))
        df = odf.paths.coalesce_df(df, how='outer')
    return df


def empty_output_frame(node: Node) -> PathsDataFrame:
    """
    Build an empty but schema-valid output for a node with no usable inputs.

    Used when a node is not wired up yet, so that one INCOMPLETE node does not take the
    whole model down with it. The frame is dimensionless: a transparent node with no inputs
    has no categorical dimensions. See ``docs/architecture/fault-tolerance.md``.
    """
    from common import polars as ppl

    schema: dict[str, Any] = {YEAR_COLUMN: pl.Int64}
    units = {}
    for metric in node.output_metrics.values():
        schema[metric.column_id] = pl.Float64
        units[metric.column_id] = metric.unit
    schema[FORECAST_COLUMN] = pl.Boolean
    meta = ppl.DataFrameMeta(units=units, primary_keys=[YEAR_COLUMN])
    return ppl.to_ppdf(pl.DataFrame(schema=schema), meta=meta)
