from __future__ import annotations

from strawberry.tools import merge_types

from kausal_common.deployment import test_mode_enabled
from kausal_common.testing.schema import TestModeMutations

from paths.schema import SBQuery, generate_schema

SB_MUTATION_TYPES: list[type] = []
if test_mode_enabled():
    SB_MUTATION_TYPES.append(TestModeMutations)

SBMutation: type | None = None
if SB_MUTATION_TYPES:
    SBMutation = merge_types('Mutation', tuple(SB_MUTATION_TYPES))

sb_schema, schema = generate_schema(SBQuery, SBMutation)
