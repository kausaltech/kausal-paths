from typing import TYPE_CHECKING, Annotated, cast

import strawberry as sb

from kausal_common.strawberry.pydantic import pydantic_type

from paths.graphql_helpers import pass_context

from nodes.scenario import Scenario

from .metric import MetricDimensionCategoryType, MetricDimensionType

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.graphql.types.node import ActionNodeType


@sb.type
class ScenarioParameterOverrideType:
    parameter_id: str
    value: sb.scalars.JSON


@pydantic_type(model=Scenario)
class ScenarioType:
    id: sb.ID
    name: sb.auto
    kind: sb.auto
    all_actions_enabled: bool
    is_selectable: sb.auto
    actual_historical_years: sb.auto

    @sb.field
    @staticmethod
    def identifier(root: Scenario) -> str:
        return root.id

    @sb.field
    @staticmethod
    def description(root: Scenario) -> str | None:
        return str(root.description) if root.description is not None else None

    @sb.field
    @staticmethod
    def parameter_overrides(root: Scenario) -> list[ScenarioParameterOverrideType]:
        return [
            ScenarioParameterOverrideType(parameter_id=param_id, value=cast('sb.scalars.JSON', value))
            for param_id, value in root.param_values.items()
        ]

    @sb.field
    @pass_context
    @staticmethod
    def is_active(root: Scenario, context: 'Context') -> bool:
        return context.active_scenario == root

    @sb.field
    @staticmethod
    def is_default(root: Scenario) -> bool:
        return root.default


@sb.type
class ScenarioValue:
    scenario: ScenarioType
    value: float | None
    year: int


@sb.type
class MetricDimensionCategoryValue:
    dimension: MetricDimensionType
    category: MetricDimensionCategoryType
    value: float | None
    year: int


@sb.type
class ActionImpactType:
    action: Annotated['ActionNodeType', sb.lazy('nodes.schema')]
    value: float
    year: int


@sb.type
class ScenarioActionImpacts:
    scenario: ScenarioType
    impacts: list[ActionImpactType]
