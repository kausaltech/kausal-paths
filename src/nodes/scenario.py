from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any

import strawberry as sb
from pydantic import ConfigDict, Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import ParameterGlobalId, ScenarioIdentifier

from params.storage import SettingStorage

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from params import Parameter

    from .context import Context


@sb.enum
class ScenarioKind(Enum):
    DEFAULT = 'default'
    BASELINE = 'baseline'
    CUSTOM = 'custom'
    PROGRESS_TRACKING = 'progress_tracking'


class Scenario(I18nBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: ScenarioIdentifier
    name: I18nString
    description: I18nString | None = None
    kind: ScenarioKind | None = None
    all_actions_enabled: bool = False
    is_selectable: bool = True
    param_values: dict[ParameterGlobalId, Any] = Field(default_factory=dict)
    actual_historical_years: list[int] | None = None

    _context: 'Context | None' = PrivateAttr(default=None)

    @property
    def default(self) -> bool:
        return self.kind == ScenarioKind.DEFAULT

    @property
    def context(self) -> Context:
        if self._context is None:
            raise RuntimeError('Context is not set')
        return self._context

    def get_param_values(self) -> Iterable[tuple[Parameter, Any]]:
        for param_id, val in self.param_values.items():
            param = self.context.get_parameter(param_id)
            yield param, val

    def get_actual_historical_years(self) -> list[int] | None:
        """
        Years for which actual (observed) data exists.

        Authored values win; a progress-tracking scenario without one derives
        the years lazily from the framework measure datapoints, so the value
        tracks live data instead of freezing at model build time.
        """
        if self.actual_historical_years is not None:
            return self.actual_historical_years
        if self.kind == ScenarioKind.PROGRESS_TRACKING:
            return self.context.measure_datapoint_years
        return None

    @contextmanager
    def override(self, set_active: bool = False) -> Generator[None]:
        old_vals: dict[str, Any] = {}

        old_scenario = self.context.active_scenario

        for param, _ in self.get_param_values():
            old_vals[param.global_id] = param.value

        self.activate()
        if set_active:
            self.context.active_scenario = self

        yield

        if set_active:
            self.context.active_scenario = old_scenario

        for param_id, val in old_vals.items():
            param = self.context.get_parameter(param_id)
            param.set(val)

    def activate(self):
        """Reset each parameter in the context to its setting for this scenario if it has one."""

        for param, val in self.get_param_values():
            param.reset_to_scenario_setting(self, val)

    def add_parameter(self, param: Parameter, value: Any):
        assert param.global_id not in self.param_values
        self.param_values[param.global_id] = value

    def has_parameter(self, param: Parameter):
        return param.global_id in self.param_values

    def get_parameter_value(self, param: Parameter):
        return self.param_values[param.global_id]

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        instance = self.context.instance if self._context is not None else None
        return "Scenario(id=%s, name='%s', instance=%s)" % (
            self.id,
            str(self.name),
            instance.id if instance is not None else None,
        )


class CustomScenario(Scenario):
    """
    The user's own scenario: a stored set of parameter overrides on top of another one.

    **The base is whichever scenario the user branched from**, not a fixed one. It was
    fixed to the default scenario until 18 Sep 2026, which made the custom scenario a
    diff against something the user might not be looking at, with two visible
    consequences: overrides made in one scenario survived a visit to another and
    reappeared on top of the default, and touching any action while a non-default
    scenario was active silently re-based everything else onto the default's values.
    On Mainz that meant turning a measure *off* could lower emissions, because the
    hidden variant switch outweighed the measure.

    `base_scenario` remains as the fallback for a session that has not branched yet, and
    for a stored base id that no longer resolves.
    """

    base_scenario: Scenario
    kind: ScenarioKind | None = ScenarioKind.CUSTOM
    _storage: SettingStorage = PrivateAttr(init=False)

    def set_storage(self, storage: SettingStorage):
        self._storage = storage

    def has_storage(self) -> bool:
        """
        Return whether a session has been attached yet.

        The storage arrives per request (`paths/schema_context.py`), so a caller outside
        a request -- a management command, a test, a resolver reached by an unusual path
        -- can legitimately meet this scenario without one.
        """
        # An unset pydantic `PrivateAttr(init=False)` raises on access, so `hasattr` is
        # the check. `'_storage' in self.__private_attributes__` is *not* -- that names
        # the declared attribute and is true whether or not a value was ever assigned.
        return hasattr(self, '_storage')

    def resolve_base(self) -> Scenario:
        """Return the scenario this custom scenario is currently a deviation from."""
        if not self.has_storage():
            return self.base_scenario
        base_id = self._storage.get_custom_base()
        if base_id is None:
            return self.base_scenario
        scenario = self.context.scenarios.get(base_id)
        if scenario is None or scenario is self:
            # A base the config no longer has, or a session that stored this scenario's
            # own id. Falling back is right: the stored overrides are still meaningful
            # against the default, and refusing to activate would strand the session.
            self.context.log.warning('custom scenario base %s does not resolve; using %s' % (base_id, self.base_scenario.id))
            self._storage.set_custom_base(None)
            return self.base_scenario
        return scenario

    def reset(self):
        self._storage.reset()
        self.base_scenario.activate()

    def get_param_values(self) -> Iterable[tuple[Parameter, Any]]:
        params = list(self._storage.get_customized_param_values().items())
        for param_id, val in params:
            param = self.context.get_parameter(param_id, required=False)
            is_valid = True
            cleaned_val = None
            if param is None:
                # The parameter might be stale (e.g. set with an older version of the backend)
                self.context.log.error('parameter %s not found in context' % param_id)
                is_valid = False
            else:
                try:
                    cleaned_val = param.clean(val)
                except Exception:
                    self.context.log.error('parameter %s has invalid value: %s', param_id, val)
                    is_valid = False
            if not is_valid:
                self._storage.reset_param(param_id)
                continue
            assert param is not None
            yield param, cleaned_val

    def get_customized_param_ids(self) -> list[ParameterGlobalId]:
        """
        Return the parameters this scenario overrides on top of its base.

        Reported separately from `param_values`, which stays empty here: a custom
        scenario's overrides live in the session rather than in the model, so the base
        class's view of "what this scenario sets" does not see them.
        """
        if not self.has_storage():
            return []
        # `ParameterGlobalId` is an annotated `str` alias, so the keys are already of
        # that type; it is not a constructor.
        return list(self._storage.get_customized_param_values())

    def activate(self):
        self.resolve_base().activate()
        for param, val in self.get_param_values():
            if not param.is_value_equal(val):
                param.set(val)
                param.is_customized = True
            else:
                self.context.log.warning('parameter %s was set to default value (%s)' % (param.global_id, val))
                self._storage.reset_param(param.global_id)
