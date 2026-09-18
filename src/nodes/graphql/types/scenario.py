from typing import TYPE_CHECKING, Annotated, cast

import strawberry as sb

from kausal_common.strawberry.pydantic import pydantic_type

from paths.graphql_helpers import pass_context

from nodes.scenario import CustomScenario, Scenario

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
    kind: sb.auto
    all_actions_enabled: bool
    is_selectable: sb.auto

    @sb.field
    @staticmethod
    def actual_historical_years(root: Scenario) -> list[int] | None:
        return root.get_actual_historical_years()

    @sb.field
    @staticmethod
    def name(root: Scenario) -> str:
        return str(root.name)

    @sb.field
    @staticmethod
    def identifier(root: Scenario) -> str:
        return root.id

    @sb.field
    @staticmethod
    def description(root: Scenario) -> str | None:
        return str(root.description) if root.description is not None else None

    @sb.field(
        description=(
            'For the custom scenario, the scenario its overrides are applied on top of; null for '
            'every other scenario. The custom scenario is a diff, and this is what it is a diff '
            'against -- it changes when the visitor edits a parameter while a different scenario '
            'is active.'
        ),
    )
    @staticmethod
    def base_scenario(root: Scenario) -> Annotated['ScenarioType', sb.lazy('nodes.graphql.types.scenario')] | None:
        if not isinstance(root, CustomScenario):
            return None
        return cast('ScenarioType', root.resolve_base())

    @sb.field(
        description=(
            'The parameters this scenario sets. Empty for the custom scenario: a visitor'
            "'s own overrides live in the session rather than in the model, so use "
            '`customizedParameters` to learn which parameters differ from the base.'
        ),
    )
    @staticmethod
    def parameter_overrides(root: Scenario) -> list[ScenarioParameterOverrideType]:
        return [
            ScenarioParameterOverrideType(parameter_id=param_id, value=cast('sb.scalars.JSON', value))
            for param_id, value in root.param_values.items()
        ]

    @sb.field(
        description=(
            'Ids of the parameters the custom scenario overrides on top of its base; empty for '
            'every other scenario. Together with `baseScenario` this is enough to show a visitor '
            'how their own scenario differs from the one they branched from.'
        ),
    )
    @staticmethod
    def customized_parameters(root: Scenario) -> list[str]:
        if not isinstance(root, CustomScenario):
            return []
        return list(root.get_customized_param_ids())

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
