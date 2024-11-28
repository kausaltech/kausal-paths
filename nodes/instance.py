from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
import pickle
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self, cast, overload

from django.db.models.aggregates import Max, Min
from pydantic.dataclasses import dataclass as pydantic_dataclass

import orjson
import platformdirs
from loguru import logger
from rich import print
from ruamel.yaml import YAML as RuamelYAML, CommentedMap  # noqa: N811
from sentry_sdk import start_span
from yaml import safe_load as yaml_load

from common import base32_crockford
from common.i18n import I18nBaseModel, TranslatedString, gettext_lazy as _, set_default_language
from nodes.constants import DecisionLevel
from nodes.exceptions import NodeError
from nodes.normalization import Normalization
from pages.config import OutcomePage, pages_from_config

from .excel_results import InstanceResultExcel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from loguru import Logger
    from ruamel.yaml.comments import LineCol

    from frameworks.models import FrameworkConfig
    from nodes.actions.action import ActionGroup
    from nodes.context import Context
    from nodes.datasets import Dataset
    from nodes.goals import NodeGoalsEntry
    from nodes.node import Node
    from nodes.units import Unit
    from params import Parameter

    from .models import InstanceConfig


yaml = RuamelYAML()


class InstanceTerms(I18nBaseModel):
    action: TranslatedString | None = None
    enabled_label: TranslatedString | None = None


@pydantic_dataclass
class InstanceFeatures:
    baseline_visible_in_graphs: bool = True
    show_accumulated_effects: bool = True
    show_significant_digits: int | None = 3
    maximum_fraction_digits: int | None = None
    hide_node_details: bool = False
    show_refresh_prompt: bool = False


@dataclass
class InstanceYAMLConfig:
    entrypoint: Path
    dependencies: list[Path] = field(default_factory=list)
    mtime_hash: str | None = None
    data: dict | None = None

    def _merge_framework_config(self, confs: list[dict], fw_confs: list[dict], entity_type: str) -> None:
        self._merge_config(confs, fw_confs, allow_override=True, entity_type=entity_type)

    def _merge_include_config(self, existing: list[dict], newconf: list[dict], entity_type: str, apply_group: str | None) -> None:
        self._merge_config(existing, newconf, allow_override=False, entity_type=entity_type, apply_group=apply_group)

    def _merge_config(
        self,
        existing: list[dict],
        newconf: list[dict],
        allow_override: bool,
        entity_type: str,
        apply_group: str | None = None,
    ) -> None:
        by_id = {d['id']: d for d in existing}
        for nc in newconf:
            c = by_id.get(nc['id'])
            if c is not None:
                if not allow_override:
                    msg = f"{entity_type} '{nc["id"]}' was already defined"
                    raise Exception(msg)
                continue
            assert 'node_group' not in nc
            nc['node_group'] = apply_group
            existing.append(nc)

    def load(self) -> dict:
        with self.entrypoint.open('r', encoding='utf8') as f:
            data: dict = yaml_load(f)
        if 'instance' in data:
            data = data['instance']

        self.dependencies = []
        frameworks = data.get('frameworks')
        if frameworks:
            for framework in frameworks:
                framework_fn = self.entrypoint.parent.joinpath('frameworks', framework).with_suffix('.yaml')
                if not framework_fn.exists():
                    raise Exception('Config expects framework but %s does not exist' % framework_fn)
                with framework_fn.open('r') as fw_f:
                    fw_data = yaml.load(fw_f)
                self.dependencies.append(framework_fn)
                self._merge_framework_config(data['nodes'], fw_data.get('nodes', []), 'Node')
                self._merge_framework_config(data['emission_sectors'], fw_data.get('emission_sectors', []), 'Emission sector')
                self._merge_framework_config(data['actions'], fw_data.get('actions', []), 'Action')
                # Some nodes, emission sectors and actions must exist in main yaml.

        includes = data.get('include', [])
        for iconf in includes:
            apply_group = iconf.get('node_group', None)
            ifn = Path(self.entrypoint).parent / Path(iconf['file'])
            if not ifn.exists():
                raise Exception('Include file "%s" not found' % str(ifn))
            with ifn.open('r') as f:
                idata = yaml.load(ifn.open('r'))
            self.dependencies.append(ifn)
            self._merge_include_config(data['nodes'], idata.get('nodes', []), 'Node', apply_group=apply_group)
            self._merge_include_config(data['dimensions'], idata.get('dimensions', []), 'Dimension', apply_group=apply_group)
        self.data = data
        return data

    def to_dict(self):
        return dict(
            entrypoint=str(self.entrypoint),
            dependencies=[str(p) for p in self.dependencies],
            mtime_hash=self._get_config_mtime_hash(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> Self | None:
        conf = cls(
            entrypoint=Path(data['entrypoint']),
            dependencies=[Path(s) for s in data['dependencies']],
        )
        if conf._get_config_mtime_hash() != data['mtime_hash']:
            logger.info('Stale YAML cache for %s' % data['entrypoint'])
            return None
        return conf

    @classmethod
    def _get_cache_fn(cls, entrypoint: Path) -> Path:
        cache_dir = platformdirs.user_cache_dir(appname='paths', appauthor='kausaltech', ensure_exists=True)
        cache_fn = Path(str(entrypoint.absolute()).replace('/', '-').lstrip('-').replace(' ', '_')).with_suffix('.pickle')
        cache_path = Path(cache_dir) / cache_fn
        return cache_path

    @classmethod
    def load_from_cache(cls, entrypoint: Path) -> dict | None:
        cache_path = cls._get_cache_fn(entrypoint)
        cache_meta_path = cache_path.with_suffix('.json')
        if not cache_path.exists() or not cache_meta_path.exists():
            return None
        try:
            with cache_meta_path.open('rb') as f:
                meta = orjson.loads(f.read())
        except Exception:
            logger.exception("Unable to load cache metadata for '%s' from '%s'" % (entrypoint, cache_meta_path))
            return None

        conf = cls.from_dict(meta)
        if conf is None:
            return None

        try:
            with cache_path.open('rb') as f:
                data = pickle.load(f)  # noqa: S301
        except Exception:
            logger.exception("Unable to load cache metadata for '%s' from '%s'" % (entrypoint, cache_meta_path))
            return None
        assert isinstance(data, dict)
        return data

    def save_to_cache(self):
        cache_path = self._get_cache_fn(self.entrypoint)
        cache_meta_path = cache_path.with_suffix('.json')

        with cache_path.open('wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = self.to_dict()
        with cache_meta_path.open('wb') as f:
            f.write(orjson.dumps(meta))

    def _get_config_mtime_hash(self) -> str:
        h = hashlib.md5(usedforsecurity=False)
        for p in (self.entrypoint, *self.dependencies):
            h.update(bytes(str(p.stat().st_mtime_ns), encoding='ascii'))
        return h.hexdigest()


@dataclass
class Instance:
    id: str
    name: TranslatedString
    owner: TranslatedString
    default_language: str
    _: dataclasses.KW_ONLY
    yaml_file_path: Path | None = None
    site_url: str | None = None
    reference_year: int | None = None
    minimum_historical_year: int
    maximum_historical_year: int | None = None
    supported_languages: list[str] = field(default_factory=list)
    lead_title: TranslatedString | None = None
    lead_paragraph: TranslatedString | None = None
    theme_identifier: str | None = None
    features: InstanceFeatures = field(default_factory=InstanceFeatures)
    terms: InstanceTerms = field(default_factory=InstanceTerms)
    result_excels: list[InstanceResultExcel] = field(default_factory=list)
    action_groups: list[ActionGroup] = field(default_factory=list)
    pages: list[OutcomePage] = field(default_factory=list)

    context: Context = field(init=False)
    obj_id: str = field(init=False)
    lock: threading.Lock = field(init=False)

    @property
    def target_year(self) -> int:
        return self.context.target_year

    @property
    def model_end_year(self) -> int:
        return self.context.model_end_year

    @cached_property
    def config(self) -> InstanceConfig:
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

    def get_goals(self, goal_id: str | None = None):
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


type InstanceLoaderFuncT[**P, R, SC: InstanceLoader] = Callable[Concatenate[SC, P], R]


class InstanceLoader:
    instance: Instance
    context: Context
    default_language: str
    yaml_file_path: Path | None = None
    config: CommentedMap | dict
    fw_config: FrameworkConfig | None = None
    _node_classes: dict[str, type[Node]]
    _input_nodes: dict[str, list[dict | str]]
    _output_nodes: dict[str, list[dict | str]]
    _subactions: dict[str, list[str]]
    _scenario_values: dict[str, list[tuple[Parameter, Any]]]

    @staticmethod
    def wrap_with_span[**P, R, SC: InstanceLoader](
        name: str,
        op: str,
    ) -> Callable[[InstanceLoaderFuncT[P, R, SC]], InstanceLoaderFuncT[P, R, SC]]:
        def wrap_with_span_outer(fn: InstanceLoaderFuncT[P, R, SC]) -> InstanceLoaderFuncT[P, R, SC]:
            @wraps(fn)
            def wrapper(self: SC, *args, **kwargs) -> R:
                _rich_traceback_omit = True
                with self.context.start_span(name, op=op):
                    return fn(self, *args, **kwargs)

            return cast(InstanceLoaderFuncT[P, R, SC], wrapper)

        return wrap_with_span_outer

    @overload
    def make_trans_string(
        self,
        config: dict,
        attr: str,
        pop: bool = False,
        required: Literal[True] = True,
        default_language=None,
    ) -> TranslatedString: ...

    @overload
    def make_trans_string(
        self,
        config: dict,
        attr: str,
        pop: bool = False,
        required: Literal[False] = False,
        default_language=None,
    ) -> TranslatedString | None: ...

    def make_trans_string(  # noqa: C901
        self,
        config: dict,
        attr: str,
        pop: bool = False,
        required: bool = False,
        default_language=None,
    ) -> None | TranslatedString:
        default_language = default_language or self.config['default_language']
        all_langs = {self.config['default_language']}
        all_langs.update(set(self.config.get('supported_languages', [])))

        default = config.get(attr)
        if pop and default is not None:
            del config[attr]
        langs = {}
        if default is not None:
            langs[self.config['default_language']] = default
        for key in list(config.keys()):
            m = re.match(r'%s_(([a-z]{2})(-[A-Z]{2})?)$' % attr, key)
            if m is None:
                continue
            full, lang, _region = m.groups()
            if full not in all_langs:
                matches = [x for x in all_langs if x.startswith('%s-' % lang)]
                if len(matches) > 1:
                    raise Exception('Too many languages match %s' % full)
                if len(matches) == 1:
                    full = matches[0]
                else:
                    # FIXME: Re-enable later when configs have been cleaned up
                    # self.logger.warning("Ignoring '%s' due to unsupported language" % key)
                    continue

            langs[full] = config[key]
            if pop:
                del config[key]
        if not langs:
            if required:
                raise Exception('Value for field %s missing' % attr)
            return None
        return TranslatedString(**langs, default_language=default_language or self.default_language)

    def simple_trans_string(self, s: str) -> TranslatedString:
        langs = {
            self.default_language: s,
        }
        return TranslatedString(**langs, default_language=self.default_language)

    def _make_node_datasets(self, config: dict, node_class: type[Node], unit: Unit | None) -> list[Dataset]:
        from nodes.datasets import DVCDataset, FixedDataset
        from nodes.units import Unit

        ds_config = config.get('input_datasets')
        datasets: list[Dataset] = []

        # If the graph doesn't specify input datasets, the node
        # might.
        if ds_config is None:
            ds_config = getattr(node_class, 'input_datasets', [])

        ds_interpolate = False
        idp_confs = config.get('input_dataset_processors', [])
        if idp_confs:
            if len(idp_confs) != 1:
                raise Exception('Only one dataset processor supported')
            proc = idp_confs[0]
            if proc != 'LinearInterpolation':
                raise Exception('Only LinearInterpolation dataset processor supported')
            ds_interpolate = True
        for ds in ds_config:
            if isinstance(ds, str):
                ds_id = ds
                dc = {}
            else:
                ds_id = ds.pop('id')
                dc = ds
            ds_unit_conf = dc.pop('unit', None)
            if isinstance(ds_unit_conf, Unit):
                ds_unit = ds_unit_conf
            elif ds_unit_conf is not None:
                ds_unit = self.context.unit_registry.parse_units(ds_unit_conf)
            else:
                ds_unit = None
            tags = dc.pop('tags', [])

            ds_obj: DVCDataset | None = None
            if self.fw_config is not None:
                from nodes.gpc import DatasetNode

                if issubclass(node_class, DatasetNode):
                    from frameworks.datasets import FrameworkMeasureDVCDataset

                    ds_obj = FrameworkMeasureDVCDataset(id=ds_id, unit=ds_unit, tags=tags, **dc)
            if ds_obj is None:
                ds_obj = DVCDataset(id=ds_id, unit=ds_unit, tags=tags, **dc)
            ds_obj.interpolate = ds_interpolate
            datasets.append(ds_obj)

        if 'historical_values' in config or 'forecast_values' in config:
            fds = FixedDataset(
                id=config['id'],
                unit=unit,  # type: ignore
                tags=config.get('tags', []),
                historical=config.get('historical_values'),
                forecast=config.get('forecast_values'),
                use_interpolation=ds_interpolate,
            )
            datasets.append(fds)
        return datasets

    def _make_node_params(self, config: dict, node: Node) -> None:
        from params.param import Parameter, ReferenceParameter

        params = config.get('params', [])
        if not params:
            return
        if isinstance(params, dict):
            params = [dict(id=param_id, value=value) for param_id, value in params.items()]
        # Ensure that the node class allows these parameters
        node_class = type(node)
        class_allowed_params = {p.local_id: p for p in getattr(node_class, 'allowed_parameters', [])}
        for pc in params:
            param_id = pc.pop('id')

            param_obj = class_allowed_params.get(param_id)
            if param_obj is None:
                raise NodeError(node, 'Parameter %s not allowed by node class' % param_id)
            param_class = type(param_obj)

            label = self.make_trans_string(pc, 'label', pop=True) or param_obj.label
            ref = pc.pop('ref', None)
            description = self.make_trans_string(pc, 'description', pop=True) or param_obj.description

            scenario_values = pc.pop('values', {})

            if ref is not None:
                target = self.context.global_parameters.get(ref)
                if target is None:
                    raise NodeError(node, 'Parameter %s refers to an unknown global parameter: %s' % (param_id, ref))

                if not isinstance(target, param_class):
                    raise NodeError(
                        node,
                        'Node requires parameter of type %s, but referenced parameter %s is %s'
                        % (
                            param_class,
                            ref,
                            type(target),
                        ),
                    )
                param = ReferenceParameter(
                    local_id=param_obj.local_id,
                    label=param_obj.label,
                    target=target,
                    context=self.context,
                )
                node.add_parameter(param)
                continue

            # Merge parameter values
            fields = asdict(param_obj)
            fields.update(pc)
            if description is not None:
                fields['description'] = description
            if label is not None:
                fields['label'] = label
            fields['context'] = self.context

            unit = fields.get('unit', None)
            if unit is not None and isinstance(unit, str):
                fields['unit'] = self.context.unit_registry.parse_units(unit)

            value = fields.pop('value', None)
            param = param_class(**fields)
            assert isinstance(param, Parameter)
            node.add_parameter(param)

            try:
                if value is not None:
                    param.set(value)
            except:
                self.instance.log.error('Error setting parameter %s for node %s' % (param.local_id, node.id))
                raise

            for scenario_id, value in scenario_values.items():
                sv = self._scenario_values.setdefault(scenario_id, list())
                sv.append((param, param.clean(value)))

    def make_node(self, node_class: type[Node], config: dict, yaml_lc: LineCol | None = None) -> Node:  # noqa: C901, PLR0912
        from nodes.node import NodeMetric
        from nodes.units import Unit

        metrics_conf = config.get('output_metrics')
        metrics: dict[str, NodeMetric] | None
        if metrics_conf is not None:
            metrics = {m['id']: NodeMetric.from_config(m) for m in metrics_conf}
            class_metrics = None
        else:
            metrics = None
            class_metrics = getattr(node_class, 'output_metrics', None)
        unit = config.get('unit')
        if unit is None:
            unit = getattr(node_class, 'default_unit', None)
            if unit is None:
                unit = getattr(node_class, 'unit', None)
            if not unit and not metrics and not class_metrics:
                raise Exception('Node %s (%s) has no unit set' % (config['id'], node_class.__name__))

        if unit and not isinstance(unit, Unit):
            unit = self.context.unit_registry.parse_units(unit)

        quantity = config.get('quantity')
        if quantity is None:
            quantity = getattr(node_class, 'quantity', None)
            if not quantity and not metrics and not class_metrics:
                raise Exception('Node %s (%s) has no quantity set' % (config['id'], node_class.__name__))

        datasets = self._make_node_datasets(config, node_class, unit)

        yaml_lct: tuple[int, int] | None = (yaml_lc.line + 1, yaml_lc.col) if yaml_lc else None  # type: ignore
        node: Node = node_class(
            id=config['id'],
            context=self.context,
            name=self.make_trans_string(config, 'name'),
            short_name=self.make_trans_string(config, 'short_name'),
            quantity=quantity,
            unit=unit,
            node_group=config.get('node_group'),
            description=self.make_trans_string(config, 'description'),
            color=config.get('color'),
            order=config.get('order'),
            is_visible=config.get('is_visible', True),
            is_outcome=config.get('is_outcome', False),
            minimum_year=config.get('minimum_year'),
            target_year_goal=config.get('target_year_goal'),
            goals=config.get('goals'),
            allow_nulls=config.get('allow_nulls', False),
            input_datasets=datasets,
            output_dimension_ids=config.get('output_dimensions'),
            input_dimension_ids=config.get('input_dimensions'),
            output_metrics=metrics,
            yaml_lc=yaml_lct,
            yaml_fn=self.yaml_file_path,
        )
        if node.id in self._input_nodes or node.id in self._output_nodes:
            raise Exception('Node %s is already configured' % node.id)
        assert node.id not in self._input_nodes
        assert node.id not in self._output_nodes
        self._input_nodes[node.id] = config.get('input_nodes', [])
        self._output_nodes[node.id] = config.get('output_nodes', [])

        self._make_node_params(config, node)

        tags = config.get('tags')
        if isinstance(tags, str):
            tags = [tags]
        if tags:
            for tag in tags:
                if not isinstance(tag, str):
                    raise NodeError(node, "'tags' must be a list of strings")
            node.tags.update(tags)

        return node

    def import_class(
        self,
        path: str,
        path_prefix: str | None = None,
        allowed_classes: Iterable[type] | None = None,
        disallowed_classes: Iterable[type] | None = None,
        node_id: str | None = None,
    ) -> type:
        if not path:
            raise Exception('Node %s: no class path given' % node_id)
        parts = path.split('.')
        class_name = parts.pop(-1)
        if path_prefix:
            prefix_parts = path_prefix.split('.')
            parts = prefix_parts + parts

        mod_path = '.'.join(parts)
        parts.append(class_name)
        full_path = '.'.join(parts)
        if full_path in self._node_classes:
            return self._node_classes[full_path]

        mod = importlib.import_module(mod_path)
        klass = getattr(mod, class_name)
        if allowed_classes and not issubclass(klass, tuple(allowed_classes)):
            raise Exception('%s is not a subclass of %s' % (klass, allowed_classes))
        if disallowed_classes:
            for k in disallowed_classes:
                if issubclass(klass, k):
                    raise TypeError('%s is a subclass of disallowed %s' % (klass, disallowed_classes))
        self._node_classes[full_path] = klass
        return klass

    def setup_dimensions(self):
        from .dimensions import Dimension

        for dc in self.config.get('dimensions', []):
            try:
                dim = Dimension.model_validate(dc)
            except Exception:
                print(dc)
                raise
            assert dim.id not in self.context.dimensions
            self.context.dimensions[dim.id] = dim

    @wrap_with_span('setup-nodes', 'function')
    def setup_nodes(self):
        from nodes.actions.action import ActionNode
        from nodes.node import Node

        for nc in self.config.get('nodes', []):
            try:
                node_class = self.import_class(
                    nc['type'],
                    'nodes',
                    allowed_classes=[Node],
                    disallowed_classes=[ActionNode],
                    node_id=nc['id'],
                )
            except ImportError:
                self.logger.error('Unable to import node class for %s' % nc.get('id'))
                raise
            node = self.make_node(node_class, nc, yaml_lc=getattr(nc, 'lc', None))
            self.context.add_node(node)

    def generate_nodes_from_emission_sectors(self):
        from nodes.simple import SectorEmissions

        node_class = self.import_class(
            'SectorEmissions',
            'nodes.simple',
            allowed_classes=[SectorEmissions],
        )
        dataset_id = self.config.get('emission_dataset')
        emission_unit = self.config.get('emission_unit')
        assert emission_unit is not None
        emission_unit = self.context.unit_registry.parse_units(emission_unit)

        for ec in self.config.get('emission_sectors', []):
            parent_id = ec.pop('part_of', None)
            data_col = ec.pop('column', None)
            data_category = ec.pop('category', None)
            if 'name_en' in ec and 'emissions' not in ec['name_en']:
                ec['name_en'] += ' emissions'
            nc = dict(
                output_nodes=[parent_id] if parent_id else [],
                input_dimensions=self.config.get('emission_dimensions', []),
                output_dimensions=self.config.get('emission_dimensions', []),
                input_datasets=[
                    dict(
                        id=dataset_id,
                        column=data_col,
                        forecast_from=self.config.get('emission_forecast_from'),
                        unit=emission_unit,
                    ),
                ]
                if data_col or data_category
                else [],
                unit=emission_unit,
                params=dict(category=data_category) if data_category else [],
                **ec,
            )
            node = self.make_node(node_class, nc, yaml_lc=getattr(ec, 'lc', None))
            self.context.add_node(node)

    @wrap_with_span('setup-actions', 'function')
    def setup_actions(self):
        from nodes.actions.action import ActionNode

        for nc in self.config.get('actions', []):
            node_class = self.import_class(
                nc['type'],
                'nodes.actions',
                allowed_classes=[ActionNode],
                node_id=nc['id'],
            )
            node = self.make_node(node_class, nc)
            assert isinstance(node, ActionNode)

            decision_level = nc.get('decision_level')
            if decision_level is not None:
                for name, val in DecisionLevel.__members__.items():
                    if decision_level == name.lower():
                        node.decision_level = val
                        break
                else:
                    raise Exception('Invalid decision level for action %s: %s' % (nc['id'], decision_level))

            ag_id = nc.get('group', None)
            if ag_id is not None:
                assert isinstance(ag_id, str)
                for ag in self.instance.action_groups:
                    if ag.id == ag_id:
                        break
                else:
                    raise Exception("Action group '%s' not found for action %s" % (ag_id, nc['id']))
                node.group = ag

            parent_id = nc.get('parent', None)
            if parent_id is not None:
                subs = self._subactions.setdefault(parent_id, [])
                subs.append(node.id)

            self.context.add_node(node)

    def _setup_edges(self) -> None:
        from nodes.edges import Edge

        ctx = self.context
        for node in ctx.nodes.values():
            try:
                for ec in self._output_nodes.get(node.id, []):
                    edge = Edge.from_config(ec, node=node, is_output=True, context=ctx)
                    node.add_edge(edge)
                    edge.output_node.add_edge(edge)

                for ec in self._input_nodes.get(node.id, []):
                    edge = Edge.from_config(ec, node=node, is_output=False, context=ctx)
                    node.add_edge(edge)
                    edge.input_node.add_edge(edge)
            except Exception:
                self.logger.error('Error setting up edges for node %s' % node)
                raise

    def _setup_subactions(self) -> None:
        from nodes.actions.action import ActionNode
        from nodes.actions.parent import ParentActionNode

        ctx = self.context
        for parent_id, subs in self._subactions.items():
            parent = ctx.nodes.get(parent_id)
            if parent is None:
                raise Exception("Action parent '%s' not found" % parent_id)
            if not isinstance(parent, ParentActionNode):
                raise TypeError("Action '%s' is marked as a parent but is not a ParentActionNode" % parent_id)
            for sub_id in subs:
                node = ctx.get_node(sub_id)
                assert isinstance(node, ActionNode)
                parent.add_subaction(node)
                node.parent_action = parent

    @wrap_with_span('setup-edges', 'function')
    def setup_edges(self) -> None:
        # Setup edges
        with start_span(name='setup-edges', op='function'):
            self._setup_edges()

        with start_span(name='setup-subactions', op='function'):
            self._setup_subactions()

        with start_span(name='finalize-nodes', op='function'):
            self.context.finalize_nodes()

    def setup_progress_tracking_scenario(self):
        from frameworks.models import MeasureDataPoint

        pt_scenario = self.context.scenarios.get('progress_tracking')
        if pt_scenario is None:
            return
        fwc = self.fw_config
        if fwc is None:
            return
        years = (
            MeasureDataPoint.objects.filter(measure__framework_config=fwc)
            .order_by()
            .values_list('year', flat=True)
            .distinct('year')
        )
        pt_scenario.actual_historical_years = list(years)

    def setup_scenarios(self):
        from nodes.scenario import CustomScenario, Scenario, ScenarioKind

        default_scenario = None

        for sc in self.config['scenarios']:
            name = self.make_trans_string(sc, 'name', pop=True)
            params_config = sc.pop('params', [])
            actual_historical_years = sc.pop('actual_historical_years', None)
            default = sc.pop('default', False)
            scenario_id: str = sc.pop('id')
            kind: ScenarioKind | None = None
            if default:
                kind = ScenarioKind.DEFAULT
            elif scenario_id == 'progress_tracking':
                kind = ScenarioKind.PROGRESS_TRACKING
            elif scenario_id == 'baseline':
                kind = ScenarioKind.BASELINE
            scenario = Scenario(
                context=self.context, id=scenario_id, name=name, actual_historical_years=actual_historical_years, kind=kind, **sc
            )

            for pc in params_config:
                param = self.context.get_parameter(pc['id'])
                scenario.add_parameter(param, param.clean(pc['value']))

            for param, value in self._scenario_values.get(scenario.id, []):
                scenario.add_parameter(param, value)

            if scenario.default:
                assert default_scenario is None
                default_scenario = scenario
            self.context.add_scenario(scenario)

        if default_scenario is None:
            raise Exception('Default scenario not defined')

        for param in self.context.get_all_parameters():
            if not param.is_customizable:
                continue
            if default_scenario.has_parameter(param):
                continue
            default_scenario.add_parameter(param, param.value)

        self.context.set_custom_scenario(
            CustomScenario(
                context=self.context,
                id='custom',
                name=_('Custom'),
                base_scenario=default_scenario,
            ),
        )

        if self.fw_config is not None:
            self.setup_progress_tracking_scenario()

    def setup_global_parameters(self):
        context = self.context
        for pc in self.config.get('params', []):
            param_id = pc.pop('id')
            pc['local_id'] = param_id
            unit_str = pc.get('unit', None)
            if unit_str is not None:
                unit = context.unit_registry.parse_units(unit_str)
                pc['unit'] = unit
            param_type = context.get_parameter_type(param_id)
            param_val = pc.pop('value', None)
            if 'is_customizable' not in pc:
                pc['is_customizable'] = False
            pc['label'] = self.make_trans_string(pc, 'label', pop=True)
            pc['description'] = self.make_trans_string(pc, 'description', pop=True)
            param = param_type(**pc)
            sub_node_ids = pc.get('subscription_nodes', None)
            if sub_node_ids is not None:
                sub_nodes = []
                for node_id in sub_node_ids:
                    sub_nodes += [context.get_node(node_id)]
                param.subscription_nodes = sub_nodes
            param.set(param_val)
            context.add_global_parameter(param)

    def setup_action_efficiency_pairs(self):
        from nodes.actions.action import ActionEfficiencyPair

        # TODO add an ID so that there can be several impact overviews for different decision makers.
        conf = self.config.get('action_efficiency_pairs', [])
        for aepc in conf:
            label = self.make_trans_string(aepc, 'label', pop=False)
            aep = ActionEfficiencyPair.from_config(
                context=self.context,
                graph_type=aepc['graph_type'],
                cost_node_id=aepc['cost_node'],
                impact_node_id=aepc['impact_node'],
                cost_unit=aepc['cost_unit'],
                impact_unit=aepc['impact_unit'],
                indicator_unit=aepc.get('indicator_unit', None),
                plot_limit_for_indicator=aepc.get('plot_limit_for_indicator', None),
                invert_cost=aepc.get('invert_cost', False),
                invert_impact=aepc.get('invert_impact', False),
                indicator_cutpoint=aepc.get('indicator_cutpoint', None),
                cost_cutpoint=aepc.get('cost_cutpoint', None),  # TODO Make these parameters.
                stakeholder_dimension=aepc.get('stakeholder_dimension', None),
                outcome_dimension=aepc.get('outcome_dimension', None),
                label=label,
            )
            self.context.action_efficiency_pairs.append(aep)

    def setup_normalizations(self):
        ncs = self.config.get('normalizations', [])
        for nc in ncs:
            n = Normalization.from_config(self.context, nc)
            n_id = n.normalizer_node.id
            self.context.add_normalization(n_id, n)

    @classmethod
    def from_dict_config(cls, config: dict, fw_config: FrameworkConfig | None = None) -> Self:
        yaml_path = config.get('yaml_file_path')
        return cls(
            config=config,
            yaml_file_path=Path(yaml_path) if yaml_path else None,
            fw_config=fw_config,
        )

    @classmethod
    def from_yaml(cls, filename: Path, fw_config: FrameworkConfig | None = None) -> Self:
        yaml_fn = Path(filename).absolute().relative_to(Path(__file__).parent.parent.absolute())

        with start_span(name='load-from-cache: %s' % yaml_fn, op='function') as span:
            config = InstanceYAMLConfig.load_from_cache(yaml_fn)
            span.set_data('cache_hit', config is not None)
        if config is None:
            yaml_conf = InstanceYAMLConfig(yaml_fn)
            with start_span(name='load-from-yaml: %s' % yaml_fn, op='function') as span:
                yaml_conf.load()
            try:
                yaml_conf.save_to_cache()
            except Exception:
                logger.exception('Unable to save instance configuration to cache')
            config = yaml_conf.data
            assert config is not None
        return cls(config=config, yaml_file_path=yaml_fn, fw_config=fw_config)

    def __init__(
        self,
        config: dict,
        yaml_file_path: Path | None = None,
        fw_config: FrameworkConfig | None = None,
    ):
        self.yaml_file_path = yaml_file_path.absolute() if yaml_file_path else None
        self.config = config
        self.fw_config = fw_config
        self.default_language = config['default_language']
        self.logger = logger.bind(instance=config['id'])
        self._node_classes = {}
        with set_default_language(self.default_language):
            self._init_instance()

    def _init_instance(self) -> None:  # noqa: PLR0915
        import dvc_pandas

        from nodes.actions.action import ActionGroup
        from nodes.context import Context

        config = self.config
        instance_id: str = config['id']
        fwc = self.fw_config
        if fwc is not None:
            instance_id = fwc.instance_config.identifier
        dataset_repo_default_path = None

        dataset_repo_config = self.config['dataset_repo']
        repo_url = dataset_repo_config['url']
        commit = dataset_repo_config.get('commit')
        creds = dvc_pandas.RepositoryCredentials(
            git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
            git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
            git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
            git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
        )
        dataset_repo = dvc_pandas.Repository(
            repo_url=repo_url,
            dvc_remote=dataset_repo_config.get('dvc_remote'),
            repo_credentials=creds,
            # cache_prefix=instance_id
        )
        dataset_repo.set_target_commit(commit)
        dataset_repo_default_path = dataset_repo_config.get('default_path')

        agc_all = self.config.get('action_groups', [])
        agcs = []
        for agc in agc_all:
            assert 'name' in agc
            ag = ActionGroup(agc['id'], self.make_trans_string(agc, 'name'), agc.get('color'))
            agcs.append(ag)

        instance_attrs = [
            'supported_languages',
            'theme_identifier',
        ]

        if fwc is None:
            owner = self.make_trans_string(self.config, 'owner', required=True)
            name = self.make_trans_string(self.config, 'name', required=True)
            max_hist_year: int | None = self.config.get('maximum_historical_year')
            min_hist_year: int = self.config['minimum_historical_year']
            site_url = self.config.get('site_url')
            reference_year = self.config.get('reference_year')
        else:
            from frameworks.models import MeasureDataPoint
            owner = self.simple_trans_string(fwc.organization_name or '')
            name = self.simple_trans_string(fwc.instance_config.get_name())
            mdp_data = MeasureDataPoint.objects.filter(measure__framework_config=fwc).aggregate(
                min_year=Min('year'),
                max_year=Max('year'),
            )
            max_hist_year = mdp_data['max_year'] or fwc.baseline_year
            min_hist_year = mdp_data['min_year'] or fwc.baseline_year
            site_url = fwc.get_view_url()
            reference_year = fwc.baseline_year
        self.instance = Instance(
            id=instance_id,
            name=name,
            owner=owner,
            default_language=self.config['default_language'],
            action_groups=agcs,
            features=self.config.get('features', {}),
            terms=self.config.get('terms', {}),
            result_excels=[InstanceResultExcel.model_validate(r) for r in self.config.get('result_excels', [])],
            yaml_file_path=self.yaml_file_path,
            pages=pages_from_config(self.config.get('pages', [])),
            maximum_historical_year=max_hist_year,
            minimum_historical_year=min_hist_year,
            site_url=site_url,
            reference_year=reference_year,
            **{attr: self.config.get(attr) for attr in instance_attrs},  # type: ignore
            # FIXME: The YAML file seems to specify what's supposed to be in InstanceConfig.lead_title (and other
            # attributes), but not under `instance` but under `pages` for a "page" whose `id' is `home`. It's a mess.
            **self._build_instance_args_from_home_page(),  # type: ignore[arg-type]
        )

        target_year = self.config['target_year']
        model_end_year = self.config.get('model_end_year', target_year)
        with start_span(name='create-context', op='function'):
            self.context = Context(
                instance=self.instance,
                dataset_repo=dataset_repo,
                target_year=target_year,
                model_end_year=model_end_year,
                dataset_repo_default_path=dataset_repo_default_path,
            )
        self.instance.set_context(self.context)

        # Store input and output node configs for each created node, to be used in setup_edges().
        self._input_nodes = {}
        self._output_nodes = {}
        self._subactions = {}
        self._scenario_values = {}
        self.setup_dimensions()
        self.generate_nodes_from_emission_sectors()
        self.setup_global_parameters()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_action_efficiency_pairs()
        self.setup_scenarios()
        self.setup_normalizations()

        for scenario in self.context.scenarios.values():
            if scenario.default:
                break
        else:
            raise Exception('No default scenario defined')
        self.context.activate_scenario(scenario)

    def _build_instance_args_from_home_page(self) -> dict[str, TranslatedString]:
        # FIXME: This is an ugly hack
        pages = self.config.get('pages', [])
        for page in pages:
            if page['id'] == 'home':
                break
        else:
            return {}
        default_language = self.config['default_language']
        return {
            'lead_title': self.make_trans_string(page, 'lead_title', default_language=default_language),
            'lead_paragraph': self.make_trans_string(page, 'lead_paragraph', default_language=default_language),
        }
