from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

from params import ParameterWithUnit
from params.base import parameter
from params.param import ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from nodes.node import Node


class ShiftTarget(BaseModel):
    node: str | int | None = None
    # dimension_id -> category_id
    categories: dict[str, str] = Field(default_factory=dict)


class ShiftAmount(BaseModel):
    year: int
    source_amount: float
    dest_amounts: list[float]


class ShiftEntry(BaseModel):
    source: ShiftTarget
    dests: list[ShiftTarget]
    amounts: list[ShiftAmount]

    # @validator('amounts')
    # def enough_years(cls, v):
    #     if len(v) < 2:
    #         raise ValueError("Must supply values for at least two years")
    #     return v

    @model_validator(mode='after')
    def dimensions_must_match(self):
        existing_dims: set[str] = set()

        def validate_target(target: ShiftTarget) -> None:
            dims = set(target.categories.keys())
            if not existing_dims:
                existing_dims.update(dims)
                return
            if dims != existing_dims:
                raise ValueError('Dimensions for yearly values for each target be equal')

        validate_target(self.source)
        for dest in self.dests:
            validate_target(dest)
        return self

    def make_index(
        self,
        output_nodes: list[Node],
        extra_level: str | None = None,
        extra_level_values: Iterable[str] | None = None,
    ) -> pd.MultiIndex:
        import pandas as pd

        dims: dict[str, set[str]] = {dim: set() for dim in list(self.source.categories.keys())}
        nodes: set[str] = set()

        def get_node_id(node: str | int | None) -> str:
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return output_nodes[nr].id

        nodes.add(get_node_id(self.source.node))
        for dim, cat in self.source.categories.items():
            dims[dim].add(cat)
        for dest in self.dests:
            nodes.add(get_node_id(dest.node))
            for dim, cat in dest.categories.items():
                dims[dim].add(cat)

        level_list = list(dims.keys())
        cat_list: list[set[str]] = [dims[dim] for dim in level_list]
        level_list.insert(0, 'node')
        cat_list.insert(0, nodes)
        if extra_level:
            level_list.append(extra_level)
            assert extra_level_values is not None
            cat_list.append(set(extra_level_values))
        index = pd.MultiIndex(cat_list, names=level_list)  # type: ignore[arg-type]
        return index


class ShiftParameterValue(RootModel[list[ShiftEntry]]):
    root: list[ShiftEntry]


@parameter
class ShiftParameter(ParameterWithUnit[ShiftParameterValue]):
    type: Literal['shift'] = 'shift'
    value: ShiftParameterValue | None = None

    def serialize_value(self) -> Any:
        return super().serialize_value()

    def clean(self, value: Any) -> ShiftParameterValue:
        if not isinstance(value, list):
            raise ValidationError(self, 'Input must be a list')

        try:
            return ShiftParameterValue.model_validate(value)
        except:
            from rich import print

            print(value)
            raise


class ReduceAmount(BaseModel):
    year: int
    amount: float


class ReduceTarget(BaseModel):
    node: str | int | None = None
    # dimension_id -> category_id
    categories: dict[str, str] = Field(default_factory=dict)


class ReduceFlow(BaseModel):
    target: ReduceTarget
    amounts: list[ReduceAmount]

    @field_validator('amounts')
    def enough_years(cls, v):  # noqa: N805
        if len(v) < 2:
            raise ValueError('Must supply values for at least two years')
        return v

    def make_index(
        self,
        output_nodes: list[Node],
        extra_level: str | None = None,
        extra_level_values: Iterable[str] | None = None,
    ) -> pd.MultiIndex:
        import pandas as pd

        dims: dict[str, set[str]] = {dim: set() for dim in list(self.target.categories.keys())}
        nodes: set[str] = set()

        def get_node_id(node: str | int | None) -> str:
            if isinstance(node, str):
                return node
            if node is None:
                nr = 0
            else:
                nr = node
            return output_nodes[nr].id

        nodes.add(get_node_id(self.target.node))
        for dim, cat in self.target.categories.items():
            dims[dim].add(cat)

        level_list = list(dims.keys())
        cat_list: list[set[str]] = [dims[dim] for dim in level_list]
        level_list.insert(0, 'node')
        cat_list.insert(0, nodes)
        if extra_level:
            level_list.append(extra_level)
            assert extra_level_values is not None
            cat_list.append(set(extra_level_values))
        index = pd.MultiIndex(cat_list, names=level_list)  # type: ignore[arg-type]
        return index


class ReduceParameterValue(RootModel[list[ReduceFlow]]):
    root: list[ReduceFlow]


@parameter
class ReduceParameter(ParameterWithUnit[ReduceParameterValue]):
    type: Literal['reduce'] = 'reduce'
    value: ReduceParameterValue | None = None

    def clean(self, value: Any) -> ReduceParameterValue:
        if not isinstance(value, list):
            raise ValidationError(self, 'Input must be a list')

        return ReduceParameterValue.model_validate(value)
