"""
Value identities, facts, origins, and conflicts for the constraint solver.

Every value the solver reasons about — a port aggregate, one delivered
binding value, a dataset metric before its binding transformations, a
compiled pipeline intermediate — gets a ``ValueKey`` and a mutable
``ValueFacts`` record inside a ``FactStore``. Facts are multi-facet from the
start (dimensions, categories, unit, quantity) and monotone: unknown may
become known, requirement sets only grow, and a contradicting fact never
overwrites an established one — it records a ``ConstraintConflict`` carrying
both origins instead.

Nothing here reads dataframes, the ORM, or runtime nodes. The solver
(``nodes/constraints/solver.py``) owns constraint semantics; this module owns
the fact lattice and its merge operations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from nodes.units import Unit

type ValueDirection = Literal['input', 'output']


@dataclass(frozen=True, slots=True)
class PortValue:
    """The aggregate value at one node port."""

    node_id: UUID
    port_id: UUID
    direction: ValueDirection


@dataclass(frozen=True, slots=True)
class BindingValue:
    """The value one binding delivers to its target port, after its transformations."""

    binding_id: UUID


@dataclass(frozen=True, slots=True)
class DatasetSourceValue:
    """The bound dataset metric's value before the binding's transformations."""

    binding_id: UUID


@dataclass(frozen=True, slots=True)
class IntermediateValue:
    """A compiled pipeline intermediate, produced by exactly one rule on the node."""

    node_id: UUID
    value_id: UUID


type ValueKey = PortValue | BindingValue | DatasetSourceValue | IntermediateValue

type ConstraintOriginKind = Literal[
    'declaration',
    'node_rule',
    'binding',
    'transformation',
    'dataset_schema',
    'dataset_profile',
]


@dataclass(frozen=True, slots=True)
class ConstraintOrigin:
    """Where a fact or requirement entered the program, for conflict provenance."""

    kind: ConstraintOriginKind
    node_id: UUID | None = None
    port_id: UUID | None = None
    binding_id: UUID | None = None
    transformation_index: int | None = None
    rule_index: int | None = None
    """Position of the rule among the node's compiled rules; distinguishes two rules constraining one output."""


@dataclass(frozen=True, slots=True)
class ConstraintConflict:
    """One structural contradiction, with every origin that participates in it."""

    code: str
    message: str
    value: ValueKey | None
    origins: tuple[ConstraintOrigin, ...]


@dataclass(slots=True)
class ValueFacts:
    """Mutable multi-facet solve state for one value."""

    dims_exact: frozenset[UUID] | None = None
    dims_exact_origin: ConstraintOrigin | None = None
    dims_required: dict[UUID, ConstraintOrigin] = field(default_factory=dict)
    dims_forbidden: dict[UUID, ConstraintOrigin] = field(default_factory=dict)
    categories: dict[UUID, frozenset[UUID]] = field(default_factory=dict)
    categories_origin: dict[UUID, ConstraintOrigin] = field(default_factory=dict)
    unit: Unit | None = None
    unit_origin: ConstraintOrigin | None = None
    quantity: str | None = None
    quantity_origin: ConstraintOrigin | None = None


@dataclass(frozen=True, slots=True)
class EffectiveValueShape:
    """Immutable snapshot of one value's facts in a solve result."""

    dimensions: frozenset[UUID] | None
    required_dimensions: frozenset[UUID]
    forbidden_dimensions: frozenset[UUID]
    categories: dict[UUID, frozenset[UUID]]
    unit: Unit | None
    quantity: str | None


def _units_compatible(a: Unit, b: Unit) -> bool:
    if a == b:
        return True
    return a.is_compatible_with(b)


class FactStore:
    """
    The fact map plus its monotone merge operations.

    Merge operations return whether anything changed (driving the fixpoint)
    and record conflicts instead of raising: the solver must deliver the
    complete conflict set, not the first contradiction. Conflicts deduplicate
    on their full identity because constraints are re-applied every sweep.
    """

    def __init__(self, *, describe: Callable[[UUID], str] | None = None) -> None:
        self.facts: dict[ValueKey, ValueFacts] = {}
        self._conflicts: dict[ConstraintConflict, None] = {}
        self.describe = describe or str

    @property
    def conflicts(self) -> tuple[ConstraintConflict, ...]:
        return tuple(self._conflicts)

    def get(self, key: ValueKey) -> ValueFacts:
        facts = self.facts.get(key)
        if facts is None:
            facts = ValueFacts()
            self.facts[key] = facts
        return facts

    def add_conflict(
        self,
        code: str,
        message: str,
        value: ValueKey | None,
        origins: tuple[ConstraintOrigin, ...],
    ) -> None:
        self._conflicts.setdefault(ConstraintConflict(code=code, message=message, value=value, origins=origins), None)

    def _describe_dims(self, dims: frozenset[UUID]) -> str:
        return '{' + ', '.join(sorted(self.describe(dim) for dim in dims)) + '}'

    def set_dims_exact(self, key: ValueKey, dims: frozenset[UUID], origin: ConstraintOrigin) -> bool:
        facts = self.get(key)
        if facts.dims_exact is not None:
            if facts.dims_exact != dims:
                assert facts.dims_exact_origin is not None
                self.add_conflict(
                    'dimension_mismatch',
                    f'Dimensions {self._describe_dims(facts.dims_exact)} contradict {self._describe_dims(dims)}',
                    key,
                    (facts.dims_exact_origin, origin),
                )
            return False
        facts.dims_exact = dims
        facts.dims_exact_origin = origin
        for dim, required_origin in facts.dims_required.items():
            if dim not in dims:
                self.add_conflict(
                    'missing_required_dimension',
                    f'Required dimension {self.describe(dim)} is not among {self._describe_dims(dims)}',
                    key,
                    (required_origin, origin),
                )
        for dim, forbidden_origin in facts.dims_forbidden.items():
            if dim in dims:
                self.add_conflict(
                    'forbidden_dimension_present',
                    f'Dimension {self.describe(dim)} must be absent but is among {self._describe_dims(dims)}',
                    key,
                    (forbidden_origin, origin),
                )
        return True

    def require_dimension(self, key: ValueKey, dim: UUID, origin: ConstraintOrigin) -> bool:
        facts = self.get(key)
        if dim in facts.dims_required:
            return False
        facts.dims_required[dim] = origin
        if facts.dims_exact is not None and dim not in facts.dims_exact:
            assert facts.dims_exact_origin is not None
            self.add_conflict(
                'missing_required_dimension',
                f'Required dimension {self.describe(dim)} is not among {self._describe_dims(facts.dims_exact)}',
                key,
                (origin, facts.dims_exact_origin),
            )
        forbidden_origin = facts.dims_forbidden.get(dim)
        if forbidden_origin is not None:
            self.add_conflict(
                'dimension_required_and_forbidden',
                f'Dimension {self.describe(dim)} is both required and forbidden',
                key,
                (origin, forbidden_origin),
            )
        return True

    def forbid_dimension(self, key: ValueKey, dim: UUID, origin: ConstraintOrigin) -> bool:
        facts = self.get(key)
        if dim in facts.dims_forbidden:
            return False
        facts.dims_forbidden[dim] = origin
        if facts.dims_exact is not None and dim in facts.dims_exact:
            assert facts.dims_exact_origin is not None
            self.add_conflict(
                'forbidden_dimension_present',
                f'Dimension {self.describe(dim)} must be absent but is among {self._describe_dims(facts.dims_exact)}',
                key,
                (origin, facts.dims_exact_origin),
            )
        required_origin = facts.dims_required.get(dim)
        if required_origin is not None:
            self.add_conflict(
                'dimension_required_and_forbidden',
                f'Dimension {self.describe(dim)} is both required and forbidden',
                key,
                (required_origin, origin),
            )
        return True

    def set_unit(self, key: ValueKey, unit: Unit, origin: ConstraintOrigin) -> bool:
        facts = self.get(key)
        if facts.unit is not None:
            if not _units_compatible(facts.unit, unit):
                assert facts.unit_origin is not None
                self.add_conflict(
                    'unit_incompatible',
                    f'Unit {facts.unit} is not convertible to {unit}',
                    key,
                    (facts.unit_origin, origin),
                )
            return False
        facts.unit = unit
        facts.unit_origin = origin
        return True

    def set_quantity(self, key: ValueKey, quantity: str, origin: ConstraintOrigin) -> bool:
        facts = self.get(key)
        if facts.quantity is not None:
            if facts.quantity != quantity:
                assert facts.quantity_origin is not None
                self.add_conflict(
                    'quantity_mismatch',
                    f'Quantity {facts.quantity!r} contradicts {quantity!r}',
                    key,
                    (facts.quantity_origin, origin),
                )
            return False
        facts.quantity = quantity
        facts.quantity_origin = origin
        return True

    def set_categories(self, key: ValueKey, dim: UUID, categories: frozenset[UUID], origin: ConstraintOrigin) -> bool:
        """
        Record the observed categories of one dimension of a value.

        Unlike the other facets this is recomputed derived data (unions and
        intersections over other values), so the *same* writer may replace its
        set as its inputs become known — but a second writer never overwrites
        the first (first writer wins, keeping the fixpoint monotone). Category
        contradictions are detected at the constraints that consume these sets
        (e.g. a disjoint filter), not at the merge.
        """
        facts = self.get(key)
        existing_origin = facts.categories_origin.get(dim)
        if existing_origin is not None and existing_origin != origin:
            return False
        if facts.categories.get(dim) == categories:
            return False
        facts.categories[dim] = categories
        facts.categories_origin[dim] = origin
        return True

    def snapshot(self) -> dict[ValueKey, EffectiveValueShape]:
        return {
            key: EffectiveValueShape(
                dimensions=facts.dims_exact,
                required_dimensions=frozenset(facts.dims_required),
                forbidden_dimensions=frozenset(facts.dims_forbidden),
                categories=dict(facts.categories),
                unit=facts.unit,
                quantity=facts.quantity,
            )
            for key, facts in self.facts.items()
        }
