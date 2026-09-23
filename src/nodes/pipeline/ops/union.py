from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .arithmetic import (
    AddOperationSpec,
    ClipOperationSpec,
    DivideOperationSpec,
    IdentityOperationSpec,
    MultiplyOperationSpec,
    SubtractOperationSpec,
)
from .temporal import BackfillOperationSpec, ExtendOperationSpec, InterpolateOperationSpec

AnyOperationSpec = Annotated[
    IdentityOperationSpec
    | AddOperationSpec
    | SubtractOperationSpec
    | MultiplyOperationSpec
    | DivideOperationSpec
    | ClipOperationSpec
    | InterpolateOperationSpec
    | ExtendOperationSpec
    | BackfillOperationSpec,
    Field(discriminator='kind'),
]
