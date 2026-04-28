from __future__ import annotations

from enum import StrEnum

from pydantic import model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nStringInstance

from paths.refs import DimensionRef

from nodes.units import Unit


class ImpactGraphType(StrEnum):
    COST_BENEFIT = 'cost_benefit'
    COST_EFFICIENCY = 'cost_efficiency'
    RETURN_ON_INVESTMENT = 'return_on_investment'
    RETURN_ON_INVESTMENT_GROSS = 'return_on_investment_gross'
    BENEFIT_COST_RATIO = 'benefit_cost_ratio'
    VALUE_OF_INFORMATION = 'value_of_information'
    SIMPLE_EFFECT = 'simple_effect'
    STACKED_RAW_IMPACT = 'stacked_raw_impact'
    WEDGE_DIAGRAM = 'wedge_diagram'


class ImpactOverviewSpec(I18nBaseModel):
    graph_type: ImpactGraphType
    effect_node_id: str
    indicator_unit: Unit
    cost_node_id: str | None = None
    cost_unit: Unit | None = None
    effect_unit: Unit | None = None
    plot_limit_for_indicator: float | None = None
    invert_cost: bool = False
    invert_effect: bool = False
    indicator_cutpoint: float | None = None
    cost_cutpoint: float | None = None
    stakeholder_dimension_id: DimensionRef | None = None
    outcome_dimension_id: DimensionRef | None = None
    label: I18nStringInstance | None = None
    cost_category_label: I18nStringInstance | None = None
    effect_category_label: I18nStringInstance | None = None
    cost_label: I18nStringInstance | None = None
    effect_label: I18nStringInstance | None = None
    indicator_label: I18nStringInstance | None = None
    description: I18nStringInstance | None = None

    @model_validator(mode='after')
    def validate_graph_type_fields(self) -> ImpactOverviewSpec:
        rf = ['effect_node_id', 'cost_node_id', 'indicator_unit']
        ff = ['outcome_dimension_id', 'stakeholder_dimension_id', 'cost_unit', 'effect_unit']
        field_lists = {
            ImpactGraphType.COST_BENEFIT: {
                'required': ['effect_node_id', 'indicator_unit'],
                'forbidden': ['cost_unit', 'effect_unit'],
            },
            ImpactGraphType.COST_EFFICIENCY: {'required': rf, 'forbidden': ['outcome_dimension_id', 'stakeholder_dimension_id']},
            ImpactGraphType.RETURN_ON_INVESTMENT: {'required': rf, 'forbidden': ff},
            ImpactGraphType.RETURN_ON_INVESTMENT_GROSS: {'required': rf, 'forbidden': ff},
            ImpactGraphType.BENEFIT_COST_RATIO: {'required': rf, 'forbidden': ff},
            ImpactGraphType.VALUE_OF_INFORMATION: {
                'required': ['effect_node_id', 'indicator_unit'],
                'forbidden': [*ff, 'cost_node_id'],
            },
            ImpactGraphType.SIMPLE_EFFECT: {
                'required': ['effect_node_id', 'indicator_unit'],
                'forbidden': [*ff, 'cost_node_id'],
            },
            ImpactGraphType.STACKED_RAW_IMPACT: {
                'required': ['effect_node_id', 'indicator_unit'],
                'forbidden': [*ff, 'cost_node_id'],
            },
            ImpactGraphType.WEDGE_DIAGRAM: {
                'required': ['effect_node_id', 'indicator_unit'],
                'forbidden': [*ff, 'cost_node_id'],
            },
        }
        required_fields = field_lists[self.graph_type]['required']
        forbidden_fields = field_lists[self.graph_type]['forbidden']

        for field_name in required_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"Field '{field_name}' must be given for graph type '{self.graph_type}'")

        for field_name in forbidden_fields:
            if getattr(self, field_name) is not None:
                raise ValueError(f"Field '{field_name}' must not be used for graph type '{self.graph_type}'")

        return self
