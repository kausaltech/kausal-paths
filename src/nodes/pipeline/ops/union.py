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
from .dimensional import SelectCategoryOperationSpec
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
    | BackfillOperationSpec
    | SelectCategoryOperationSpec,
    Field(discriminator='kind'),
]
