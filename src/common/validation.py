"""Operation gates shared by model and dataset validation."""

from enum import StrEnum
from typing import Annotated, Literal, assert_never

import strawberry as sb
from pydantic import AfterValidator


@sb.enum(
    name='ValidationRuleEnforcement',
    description=(
        'Which operation a validation problem blocks. BLOCK_EDIT rejects edits that introduce it; '
        'BLOCK_PUBLISH allows edits but blocks publication; BLOCK_SUBMISSION allows edits and '
        'publication but blocks finalising a submission. Each tier also blocks the ones after it.'
    ),
)
class ValidationRuleEnforcement(StrEnum):
    """
    Which operation a validation problem blocks.

    The tiers are ordered: each blocks the operations of the tiers after it too, so a
    `BLOCK_EDIT` problem also blocks publication and submission. Choose by what the problem
    means for the result, not by how urgent it feels.

    One type for every validation scope: dataset rules, node input contracts and instance-wide
    constraints. A scope may allow only some tiers (see `PublishOrSubmitEnforcement`).
    """

    BLOCK_EDIT = 'block_edit'
    """
    The value itself is invalid, such as negative energy. A write that introduces a new
    violation is refused; violations that already exist never block an unrelated edit.
    """

    BLOCK_PUBLISH = 'block_publish'
    """
    The result would be wrong or broken if shown: a value that could not be computed, two rows
    where one is allowed, activity without the factor it needs, a data-entry layout that cannot
    be rendered. Edits are allowed and the problem is shown, but the instance cannot be
    published until it is resolved.
    """

    BLOCK_SUBMISSION = 'block_submission'
    """
    The result is sound to show, but not complete or conformant enough to be finalised: a
    required combination not yet entered, a value without its quality grade, no inventory years
    declared. Edits and publication are allowed; finalising a submission is refused.
    """


def _not_edit(enforcement: ValidationRuleEnforcement) -> ValidationRuleEnforcement:
    if enforcement is ValidationRuleEnforcement.BLOCK_EDIT:
        raise ValueError('block_edit is not available here; use block_publish or block_submission')
    return enforcement


type PublishOrSubmitEnforcement = Annotated[ValidationRuleEnforcement, AfterValidator(_not_edit)]
"""
A tier for checks over computed values or a whole dataset, which cannot gate individual edits: a
dataset being entered incrementally is always incomplete mid-entry. Node input contracts and the
gap rule use it.
"""

type ValidationOperation = Literal['edit', 'publish', 'submit']


def blocks_operation(enforcement: ValidationRuleEnforcement, operation: ValidationOperation) -> bool:
    match operation:
        case 'edit':
            return enforcement == ValidationRuleEnforcement.BLOCK_EDIT
        case 'publish':
            return enforcement in (ValidationRuleEnforcement.BLOCK_EDIT, ValidationRuleEnforcement.BLOCK_PUBLISH)
        case 'submit':
            return True
        case _:
            assert_never(operation)
