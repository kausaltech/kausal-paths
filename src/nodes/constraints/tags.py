"""
How a binding's tag operations bear on the shapes a constraint can state.

Shared vocabulary rather than solver-internal detail, because node classes
need it too: ``Node.shape_rules()`` is compiled from *ports*, while a tag
operation is authored on a *binding*, so a class that wants to state the
algebra of its inputs has to read the tags back off the bindings and decide
what it can still honestly claim.

Three cases, in the order a caller should test them:

* **neutral** — the operation provably preserves dimensions, categories,
  unit and quantity, so it changes nothing a rule says;
* **inverting** — the operation's unit effect is exactly reciprocal, which
  ``ProductShapeRule.inverse_inputs`` states precisely;
* **opaque** — anything else registered: the value is re-united or reshaped
  in a way no rule kind can express, so the constraint must be dropped
  rather than guessed at.

A tag that is not a registered dataframe operation at all selects behavior
(``non_additive``, a formula variable name, a city-specific selector) rather
than transforming the frame, and stays neutral.
"""

NEUTRAL_TAG_OPERATIONS = frozenset({
    'abs',
    'absolute',
    'add_missing_years',
    'arithmetic_inverse',
    'cumulative',
    'drop_infs',
    'drop_nans',
    'drop_zeros',
    'empty_to_zero',
    'extend_all',
    'extend_both_ways',
    'extend_forecast_values',
    'extend_to_history',
    'extend_values',
    'extrapolate',
    'fill_metrics_nan_null_zero',
    'forecast_only',
    'ignore_content',
    'inventory_only',
    'linear_interpolate',
    'make_nonnegative',
    'make_nonpositive',
    'minus',
    'observed_only_extend_all',
    'round_to_five',
    'truncate_before_start',
    'truncate_beyond_end',
    'use_observations',
})
"""
Registered tag operations that provably preserve dimensions, categories,
unit, and quantity. Any *other* registered operation (``complement``,
``ratio_to_last_historical_value``, …) reshapes or re-units the value, so
the binding carrying it goes opaque unless it is listed as inverting below.
"""

INVERTING_TAG_OPERATIONS = frozenset({
    'geometric_inverse',
})
"""
Registered tag operations whose effect on a unit is exactly reciprocal.

These are opaque to anything that can only pass a unit through, but a
product *can* state them: the tagged operand divides the product instead of
multiplying it. Dimensions and categories are untouched — division does not
remove a dimension — so a product rule remains fully expressible.
"""


def tag_operation_is_registered(tag: str) -> bool:
    """Whether the tag names a dataframe operation at all, as opposed to selecting behavior."""
    from common.polars_ext import PathsExt

    return tag in PathsExt._OPERATION_METHODS


def tag_operation_is_opaque(tag: str) -> bool:
    """Whether the tag transforms a value in a way no shape rule can state."""
    return tag_operation_is_registered(tag) and tag not in NEUTRAL_TAG_OPERATIONS


def tag_operation_inverts(tag: str) -> bool:
    """Whether the tag's unit effect is reciprocal, i.e. expressible as a product's inverse operand."""
    return tag in INVERTING_TAG_OPERATIONS and tag_operation_is_registered(tag)
