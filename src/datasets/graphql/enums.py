"""GraphQL enum definitions shared by dataset types and findings."""

import strawberry as sb

from datasets.models import PlausibilityAggregation, PlausibilityDenominator, PlausibilityReference

# Register before any Strawberry type uses these choices, regardless of import order.
sb.enum(
    PlausibilityAggregation,
    name='PlausibilityAggregation',
    description='Whether each selected cell is checked, or their sum per year.',
)
sb.enum(PlausibilityDenominator, name='PlausibilityDenominator')
sb.enum(
    PlausibilityReference,
    name='PlausibilityReference',
    description='Whether the bounds apply to the value, or to its ratio to an earlier year.',
)
