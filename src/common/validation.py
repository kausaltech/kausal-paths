"""Operation gates shared by model and dataset validation."""

from typing import Literal, assert_never

type Enforcement = Literal['block_edit', 'block_publish', 'block_submission']
type ValidationOperation = Literal['edit', 'publish', 'submit']


def blocks_operation(enforcement: Enforcement, operation: ValidationOperation) -> bool:
    match operation:
        case 'edit':
            return enforcement == 'block_edit'
        case 'publish':
            return enforcement in ('block_edit', 'block_publish')
        case 'submit':
            return True
        case _:
            assert_never(operation)
