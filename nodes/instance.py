from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, overload

from pydantic.dataclasses import dataclass as pydantic_dataclass

from loguru import logger

from common import base32_crockford
from common.i18n import I18nBaseModel, TranslatedString

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from loguru import Logger

    from nodes.actions.action import ActionGroup
    from nodes.context import Context
    from nodes.goals import NodeGoalsEntry
    from pages.config import OutcomePage

    from .excel_results import InstanceResultExcel
    from .models import InstanceConfig


class InstanceTerms(I18nBaseModel):
    """
    The specific terms to be used for different concepts in the UI (e.g. "action").

    If a term is not set, the default term for the concept will be used.
    """

    action: TranslatedString | None = None
    enabled_label: TranslatedString | None = None


@pydantic_dataclass
class InstanceFeatures:
    """
    Features available for the instance.

    Used mostly by the UI to customize the display of the results.
    """

    baseline_visible_in_graphs: bool = True
    show_accumulated_effects: bool = True
    show_significant_digits: int | None = 3
    maximum_fraction_digits: int | None = None
    hide_node_details: bool = False
    show_refresh_prompt: bool = False
    requires_authentication: bool = False


@dataclass
class Instance:
    """
    A calculation model instance with its configuration and settings.

    An Instance manages the configuration and execution context for a specific calculation
    setup, including language configuration, available UI and other features, the specific terms
    to be used in the UI, and result handling.
    """

    id: str
    "The globally unique identifier for the instance."

    name: TranslatedString
    "The human-readable name of the instance."

    owner: TranslatedString
    """The owner of the instance (e.g. "City of Sunnydale")."""

    default_language: str
    """The default language for the instance (e.g. "de-CH")."""

    _: dataclasses.KW_ONLY
    yaml_file_path: Path | None = None
    site_url: str | None = None
    """The URL for the main deployment of thepublic UI for this instance."""

    reference_year: int | None = None
    """Reference (or baseline) year for the instance."""

    minimum_historical_year: int
    """The earliest year for which historical data is available.

    This is commonly also the minimum year for which outputs will be generated.
    """

    maximum_historical_year: int | None = None
    """The latest year for which historical data is available.

    The years after this year are generally forecasted years.
    """

    supported_languages: list[str] = field(default_factory=list)
    """All the supported language codes for the instance.

    Includes the default language.
    """

    lead_title: TranslatedString | None = None
    """The lead title shown in the UI."""

    lead_paragraph: TranslatedString | None = None
    """The lead paragraph shown in the UI."""

    theme_identifier: str | None = None
    """The identifier for the UI theme to be used for this instance."""

    features: InstanceFeatures = field(default_factory=InstanceFeatures)
    """Features available for the instance."""

    terms: InstanceTerms = field(default_factory=InstanceTerms)
    """The specific terms to be used for different concepts in the UI (e.g. "action")."""

    result_excels: list[InstanceResultExcel] = field(default_factory=list)
    """Configuration for the Excel file exports that can be generated for the instance."""

    action_groups: list[ActionGroup] = field(default_factory=list)
    """List of all the action groups.

    Actions can be grouped into thematic groups for the UI. Can be an empty list
    if no grouping is desired.
    """

    pages: list[OutcomePage] = field(default_factory=list)
    """The default configuration for all the main outcome pages in the UI."""

    context: Context = field(init=False)
    """Computation execution context."""

    obj_id: str = field(init=False)
    """Randomly generated identifier for the instance. Used for logging."""

    lock: threading.Lock = field(init=False)
    """Lock for thread-safe operations."""

    @property
    def target_year(self) -> int:
        return self.context.target_year

    @property
    def model_end_year(self) -> int:
        return self.context.model_end_year

    @cached_property
    def config(self) -> InstanceConfig:
        """Returns the Django model object that corresponds to this instance."""
        from .models import InstanceConfig

        return InstanceConfig.objects.get(identifier=self.id)

    def __post_init__(self):
        self.modified_at: datetime | None = None
        self.lock = threading.Lock()
        self.obj_id = base32_crockford.gen_obj_id(self)
        self.log: Logger = logger.bind(instance=self.id, instance_obj_id=self.obj_id, markup=True)
        if isinstance(self.features, dict):
            self.features = InstanceFeatures(**self.features)
        if isinstance(self.terms, dict):
            self.terms = InstanceTerms(**self.terms)
        if not self.supported_languages:
            self.supported_languages = [self.default_language]
        elif self.default_language not in self.supported_languages:
            self.supported_languages.append(self.default_language)

    def set_context(self, context: Context):
        self.context = context

    def update_dataset_repo_commit(self, commit_id: str):
        from ruamel.yaml import YAML as RuamelYAML  # noqa: N811
        yaml = RuamelYAML()

        assert self.yaml_file_path is not None
        with self.yaml_file_path.open('r', encoding='utf8') as f:
            data = yaml.load(f)
        instance_data = data.get('instance', data)
        instance_data['dataset_repo']['commit'] = commit_id
        with self.yaml_file_path.open('w', encoding='utf8') as f:
            yaml.dump(data, f)

    def warning(self, msg: str, *args):
        self.log.opt(depth=1).warning(msg, *args)

    @overload
    def get_goals(self, goal_id: str) -> NodeGoalsEntry: ...

    @overload
    def get_goals(self) -> list[NodeGoalsEntry]: ...

    def get_goals(self, goal_id: str | None = None) -> NodeGoalsEntry | list[NodeGoalsEntry]:
        """
        Return a named goal or all the goals configured for the instance.

        Args:
            goal_id: The id of the goal to return. If not provided, all the goals are returned.

        Returns:
            A single goal or a list of all the goals.

        Raises:
            Exception: If goal_id is provided and not found.

        """

        ctx = self.context
        outcome_nodes = ctx.get_outcome_nodes()
        goals: list[NodeGoalsEntry] = []
        for node in outcome_nodes:
            if not node.goals:
                continue
            for ge in node.goals.root:
                if not ge.is_main_goal:
                    continue
                if goal_id is not None and ge.get_id() != goal_id:
                    continue
                goals.append(ge)

        if goal_id:
            if len(goals) != 1:
                raise Exception('Goal with id %s not found', goal_id)
            return goals[0]

        return goals

    def clean(self):
        """
        Clear references to other objects.

        This is invoked right before the instance is deleted to avoid memory leaks
        from circular references.
        """
        # Workaround for pytests; if we have a globally set instance, do not
        # clean it.
        from nodes.models import _pytest_instances  # pyright: ignore

        if _pytest_instances.get(self.id) == self:
            return

        self.log.debug('Cleaning instance')
        self.context.clean()  # type: ignore
        self.context.instance = None  # type: ignore
        self.config = None  # type: ignore
        self.fw_config = None
        self.context = None  # type: ignore
