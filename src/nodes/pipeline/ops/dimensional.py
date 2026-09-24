"""Operations over a node's dimensions."""

from typing import Literal

from paths.identifiers import DimensionCategoryIdentifier, DimensionIdentifier  # noqa: TC002  (pydantic fields)

from .base import InputOperationSpec, ParameterInputRef


class SelectCategoryOperationSpec(InputOperationSpec):
    """
    Keep one category of a dimension and drop the dimension.

    The category is either fixed or the value of a parameter, typically a
    `DimensionCategoryParameter`: that is how a scenario picks, say, the
    implementation variant a measure follows. The selection belongs to the
    computation rather than to a binding, so datasets stay scenario-independent.
    """

    kind: Literal['select_category'] = 'select_category'
    dimension: DimensionIdentifier
    category: DimensionCategoryIdentifier | ParameterInputRef
