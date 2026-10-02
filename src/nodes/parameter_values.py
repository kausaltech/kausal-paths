"""Persist administrator-selected scenario values without changing inherited declarations."""

from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError
from django.db import transaction

from nodes.instance_serialization import build_instance_snapshot
from nodes.models import InstanceConfig
from nodes.scenario import Scenario, ScenarioKind
from nodes.template_graph import build_local_nodes, template_snapshot
from nodes.template_spec import parameters_by_id
from params.base import ParameterOwner
from params.param import ReferenceParameter, ValidationError as ParameterValidationError

if TYPE_CHECKING:
    from pydantic import JsonValue

    from users.models import User


@transaction.atomic
def set_scenario_parameter(  # noqa: C901, PLR0912, PLR0915
    instance: InstanceConfig,
    identifier: str,
    value: JsonValue,
    *,
    scenario_id: str | None = None,
    reset: bool = False,
    user: User,
) -> None:
    instance = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
    if not instance.permission_policy().user_has_perm(user, 'change', instance):
        raise ValidationError('You do not have permission to change this instance')
    if instance.is_locked:
        raise ValidationError('Instance is locked')
    effective = build_instance_snapshot(instance)
    scenario = (
        next((s for s in effective.spec.scenarios if s.id == scenario_id), None)
        if scenario_id
        else next(
            (s for s in effective.spec.scenarios if s.default),
            None,
        )
    )
    if scenario is None:
        raise ValidationError('Unknown scenario')
    parameter = parameters_by_id(effective.spec, effective.nodes).get(identifier)
    if parameter is None:
        raise ValidationError(f'Unknown parameter {identifier}')
    if instance.template_revision_id is not None and parameter.owner == ParameterOwner.FRAMEWORK:
        raise ValidationError('Parameter is controlled by the framework')
    if isinstance(parameter, ReferenceParameter):
        identifier = parameter.target_id
        parameter = parameters_by_id(effective.spec, effective.nodes).get(identifier)
        if parameter is None:
            raise ValidationError('Unknown referenced parameter')
        if instance.template_revision_id is not None and parameter.owner == ParameterOwner.FRAMEWORK:
            raise ValidationError('Parameter is controlled by the framework')
    try:
        cleaned = parameter.clean(value) if not reset else None
    except ParameterValidationError as exc:
        raise ValidationError(str(exc)) from exc
    spec = instance.ensure_spec()
    if instance.template_revision_id is not None:
        base = template_snapshot(instance)
        inherited_scenario = next((s for s in base.spec.scenarios if s.id == scenario.id), None)
        if inherited_scenario is None and not base.spec.scenarios and scenario.default:
            inherited_scenario = Scenario(id=scenario.id, name='Default', kind=ScenarioKind.DEFAULT)
    else:
        inherited_scenario = None
    if inherited_scenario is not None:
        declared = parameters_by_id(base.spec, base.nodes).get(identifier)
        if declared is None:
            declared = parameters_by_id(spec, build_local_nodes(instance)).get(identifier)
        fallback = declared.model_dump(mode='json').get('value') if declared is not None else parameter.serialize_value()
        inherited_value = inherited_scenario.param_values.get(identifier, fallback)
        # Named scenarios inherit the municipal default for values they do not name.
        if not scenario.default and identifier not in inherited_scenario.param_values:
            inherited_value = parameter.serialize_value()
        override = spec.local_scenario(scenario.id)
        if reset or cleaned == inherited_value:
            override.param_values.pop(identifier, None)
            override.parameter_types.pop(identifier, None)
        else:
            override.param_values[identifier] = cleaned
            override.parameter_types[identifier] = parameter.type
        if not override.param_values:
            spec.scenarios.remove(override)
    else:
        local_scenario = next(s for s in spec.scenarios if s.id == scenario.id)
        if reset or cleaned == parameter.serialize_value():
            local_scenario.param_values.pop(identifier, None)
        else:
            local_scenario.param_values[identifier] = cleaned
    instance.spec = spec
    instance.save(update_fields=['spec'])
    instance.invalidate_cache()
