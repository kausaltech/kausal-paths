from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Literal, Self, TypedDict, cast, overload

from django.db.models.aggregates import Max, Min
from pydantic import BaseModel, Field, field_validator

import platformdirs
from loguru import logger
from rich import print
from ruamel.yaml import YAML as RuamelYAML  # noqa: N811
from sentry_sdk import start_span

from kausal_common.i18n.pydantic import TranslatedString, get_i18n_context, gettext_lazy as _, set_i18n_context

from nodes.actions.action import ActionNode
from nodes.constants import DecisionLevel
from nodes.defs.instance_defs import DatasetRepoSpec
from nodes.exceptions import NodeError
from nodes.explanations import NodeExplanationSystem
from pages.config import pages_from_config
from params.discover import discover_global_parameters

from .excel_results import InstanceResultExcel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from uuid import UUID

    from ruamel.yaml import CommentedMap
    from ruamel.yaml.comments import LineCol

    from kausal_common.datasets.models import Dataset as DBDatasetModel
    from kausal_common.i18n.pydantic import I18nString

    from frameworks.models import FrameworkConfig
    from nodes.context import Context
    from nodes.datasets import Dataset
    from nodes.defs.node_defs import NodeSpec
    from nodes.edges import Edge
    from nodes.instance import Instance
    from nodes.instance_serialization import DatasetPortSnapshot, EdgeSnapshot, InstanceSnapshot, NodeSnapshot
    from nodes.node import Node, NodeMetric
    from nodes.scenario import Scenario
    from nodes.units import Unit
    from params import Parameter


class ConfigLocation(TypedDict):
    file_path: str
    line: int
    column: int


class FileDependency(BaseModel):
    path: Path
    """Canonical path to the file."""

    modified_at: int
    """Timestamp in nanoseconds since epoch."""


class InstanceYAMLMeta(BaseModel):
    CURRENT_METADATA_VERSION: ClassVar[int] = 1

    metadata_version: int
    entrypoint: FileDependency
    dependencies: list[FileDependency] = Field(default_factory=list)
    mtime_hash: str | None = None

    @field_validator('metadata_version')
    @classmethod
    def validate_metadata_version(cls, v: int) -> int:
        if v != cls.CURRENT_METADATA_VERSION:
            raise ValueError('Unsupported metadata version: %s' % v)
        return v

    @field_validator('entrypoint')
    @classmethod
    def validate_entrypoint(cls, v: FileDependency) -> FileDependency:
        if not v.path.exists():
            raise ValueError('Entrypoint does not exist: %s' % v.path)
        return v

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v: list[FileDependency]) -> list[FileDependency]:
        for dep in v:
            if not dep.path.exists():
                raise ValueError('Dependency does not exist: %s' % dep.path)
        return v

    def add_dependency(self, path: Path) -> None:
        self.dependencies.append(FileDependency(path=path, modified_at=path.stat().st_mtime_ns))

    def calculate_mtime_hash(self) -> str:
        h = hashlib.md5(usedforsecurity=False)
        for p in (self.entrypoint, *self.dependencies):
            h.update(bytes(str(p.path), encoding='utf8'))
            h.update(bytes(str(p.modified_at), encoding='ascii'))
        return h.hexdigest()

    def refresh(self) -> Self:
        new_meta = self.model_copy()
        for p in (new_meta.entrypoint, *new_meta.dependencies):
            p.modified_at = p.path.stat().st_mtime_ns
        new_meta.mtime_hash = new_meta.calculate_mtime_hash()
        return new_meta

    @cached_property
    def latest_modified_at(self) -> int:
        return max(p.modified_at for p in (self.entrypoint, *self.dependencies))


@dataclass
class InstanceYAMLConfig:
    meta: InstanceYAMLMeta
    data: dict[str, Any] | None = None

    def _merge_framework_config(
        self, confs: list[CommentedMap], fw_confs: list[CommentedMap], entity_type: str, config_path: Path | None
    ) -> None:
        self._merge_config(confs, fw_confs, allow_override=True, entity_type=entity_type, config_path=config_path)

    def _merge_include_config(
        self,
        existing: list[CommentedMap],
        newconf: list[CommentedMap],
        entity_type: str,
        apply_group: str | None,
        config_path: Path | None,
        allow_override: bool = False,
        dataset_replacements: list[dict[str, str]] | None = None,
    ) -> None:
        # Create a mapping of old dataset IDs to new ones
        if dataset_replacements is None:
            dataset_replacements = []
        dataset_map = {rep['from']: rep['to'] for rep in dataset_replacements}

        # Process each node in the new configuration
        for nc in newconf:
            # Replace dataset IDs in input_datasets if present
            if 'input_datasets' in nc:
                for i, ds in enumerate(nc['input_datasets']):
                    if isinstance(ds, str):
                        nc['input_datasets'][i] = dataset_map.get(ds, ds)
                        # print(f"Node {nc.id} datasets from {ds} to {dataset_map.get(ds)}")
                    else:
                        ds_id = ds['id']
                        nc['input_datasets'][i]['id'] = dataset_map.get(ds_id, ds_id)

        self._merge_config(
            existing,
            newconf,
            entity_type=entity_type,
            apply_group=apply_group,
            config_path=config_path,
            allow_override=allow_override,
        )

    def _merge_config(
        self,
        existing: list[CommentedMap],
        newconf: list[CommentedMap],
        entity_type: str,
        apply_group: str | None = None,
        config_path: Path | None = None,
        allow_override: bool = False,
    ) -> None:
        by_id = {d['id']: d for d in existing}
        for nc in newconf:
            c = by_id.get(nc['id'])
            if c is not None:
                if not allow_override:
                    msg = f"{entity_type} '{nc['id']}' was already defined"
                    raise Exception(msg)
                continue
            assert 'node_group' not in nc
            nc['node_group'] = apply_group
            if config_path is not None:
                nc['config_location'] = ConfigLocation(file_path=str(config_path), line=nc.lc.line + 1, column=nc.lc.col)
            existing.append(nc)

    def _init_group(self, objs: list[CommentedMap]) -> None:
        for d in objs:
            d['config_location'] = ConfigLocation(file_path=str(self.meta.entrypoint.path), line=d.lc.line + 1, column=d.lc.col)

    def load(self):
        meta = self.meta
        entrypoint = meta.entrypoint
        yaml = RuamelYAML()
        with entrypoint.path.open('r', encoding='utf8') as f:
            loaded = yaml.load(f)
        if not isinstance(loaded, dict):
            msg = 'Expected mapping at YAML root, got %s' % type(loaded).__name__
            raise TypeError(msg)
        data: dict[str, Any] = loaded
        if 'instance' in data:
            data = data['instance']

        meta.dependencies = []
        config_path = entrypoint.path.parent
        frameworks = data.get('frameworks', [])

        nodes = data.get('nodes', [])
        emission_sectors = data.get('emission_sectors', [])
        actions = data.get('actions', [])

        dimensions = data.get('dimensions', [])

        self._init_group(nodes)
        self._init_group(emission_sectors)
        self._init_group(actions)

        for framework in frameworks:
            framework_fn = config_path.joinpath('frameworks', framework).with_suffix('.yaml').resolve()
            if not framework_fn.exists():
                raise Exception('Config expects framework but %s does not exist' % framework_fn)
            with framework_fn.open('r') as fw_f:
                fw_data = yaml.load(fw_f)
            meta.add_dependency(framework_fn)
            self._merge_framework_config(nodes, fw_data.get('nodes', []), 'Node', config_path=framework_fn)
            self._merge_framework_config(
                emission_sectors, fw_data.get('emission_sectors', []), 'Emission sector', config_path=framework_fn
            )
            self._merge_framework_config(actions, fw_data.get('actions', []), 'Action', config_path=framework_fn)
            # Some nodes, emission sectors and actions must exist in main yaml.

        includes = data.get('include', [])
        for iconf in includes:
            allow_override = iconf.get('allow_override', False)
            apply_group = iconf.get('node_group', None)
            dataset_replacements = iconf.get('dataset_replacements', [])
            ifn = (config_path / Path(iconf['file'])).resolve()
            if not ifn.exists():
                raise Exception('Include file "%s" not found' % str(ifn))
            with ifn.open('r') as f:
                idata = yaml.load(f)
            meta.add_dependency(ifn)
            self._merge_include_config(
                nodes,
                idata.get('nodes', []),
                'Node',
                apply_group=apply_group,
                config_path=ifn,
                allow_override=allow_override,
                dataset_replacements=dataset_replacements,
            )
            self._merge_include_config(
                dimensions,
                idata.get('dimensions', []),
                'Dimension',
                apply_group=apply_group,
                config_path=None,
                allow_override=allow_override,
            )
            self._merge_include_config(
                actions,
                idata.get('actions', []),
                'Action',
                apply_group=apply_group,
                config_path=None,
                allow_override=allow_override,
                dataset_replacements=dataset_replacements,
            )

        # Make sure that assignment works even if they are originally empty.
        data['actions'] = actions
        data['nodes'] = nodes
        data['dimensions'] = dimensions

        # Serialize and deserialize to get rid of Ruamel extras
        ser_data = json.dumps(data)
        data = json.loads(ser_data)

        self.data = data

    @classmethod
    def from_meta(cls, meta: InstanceYAMLMeta) -> Self | None:
        conf = cls(meta=meta)
        disk_meta = meta.refresh()
        if conf.meta.mtime_hash != disk_meta.mtime_hash:
            logger.info('Stale YAML cache for %s' % str(meta.entrypoint.path))
            return None
        return conf

    @classmethod
    def _get_cache_fn(cls, entrypoint: Path) -> Path:
        cache_dir = platformdirs.user_cache_dir(appname='paths', appauthor='kausaltech', ensure_exists=True)
        cache_fn = Path(str(entrypoint.absolute()).replace('/', '-').lstrip('-').replace(' ', '_')).with_suffix('.pickle')
        cache_path = Path(cache_dir) / cache_fn
        return cache_path

    @classmethod
    def load_from_cache(cls, entrypoint_path: Path) -> Self | None:
        cache_path = cls._get_cache_fn(entrypoint_path)
        cache_meta_path = cache_path.with_suffix('.json')
        if not cache_path.exists() or not cache_meta_path.exists():
            return None
        try:
            with cache_meta_path.open('rb') as f:
                meta = InstanceYAMLMeta.model_validate_json(f.read())
        except Exception as e:
            logger.warning("Unable to load cache metadata for '%s' from '%s': %s" % (entrypoint_path, cache_meta_path, str(e)))
            return None

        conf = cls.from_meta(meta)
        if conf is None:
            return None

        try:
            with cache_path.open('rb') as f:
                data = pickle.load(f)  # noqa: S301
        except Exception:
            logger.exception("Unable to load cached instance for '%s' from '%s'" % (entrypoint_path, cache_path))
            return None
        assert isinstance(data, dict)
        conf.data = cast('dict[str, Any]', data)

        return conf

    @classmethod
    def load_for_entrypoint(cls, entrypoint: Path) -> Self:
        entrypoint = entrypoint.resolve()
        try:
            relative_fn = entrypoint.relative_to(Path(__file__).parent.parent.resolve())
        except ValueError:
            relative_fn = entrypoint

        with start_span(name='load-from-cache: %s' % relative_fn, op='function') as span:
            yaml_conf = cls.load_from_cache(entrypoint)
            span.set_data('cache_hit', yaml_conf is not None)

        if yaml_conf is not None:
            return yaml_conf

        logger.info('Cached instance not found or stale for %s, loading from YAML' % relative_fn)
        yaml_conf = cls.from_entrypoint(entrypoint)
        with start_span(name='load-from-yaml: %s' % relative_fn, op='function'):
            yaml_conf.load()
        yaml_conf.meta.mtime_hash = yaml_conf.meta.calculate_mtime_hash()
        try:
            yaml_conf.save_to_cache()
        except Exception:
            logger.exception('Unable to save instance configuration to cache')
        return yaml_conf

    @classmethod
    def from_entrypoint(cls, entrypoint: Path) -> Self:
        version = InstanceYAMLMeta.CURRENT_METADATA_VERSION
        meta = InstanceYAMLMeta(
            metadata_version=version, entrypoint=FileDependency(path=entrypoint, modified_at=entrypoint.stat().st_mtime_ns)
        )
        conf = cls(meta=meta)
        return conf

    def save_to_cache(self):
        meta = self.meta
        entrypoint = self.meta.entrypoint
        cache_path = self._get_cache_fn(entrypoint.path)
        cache_meta_path = cache_path.with_suffix('.json')

        with cache_path.open('wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with cache_meta_path.open('w', encoding='utf8') as f:
            meta.mtime_hash = meta.calculate_mtime_hash()
            f.write(meta.model_dump_json(indent=2))

    def _get_config_mtime_hash(self) -> str:
        return self.meta.calculate_mtime_hash()


type InstanceLoaderFuncT[**P, R, SC: InstanceLoader] = Callable[Concatenate[SC, P], R]


@overload
def make_trans_string(
    config: dict[str, Any],
    attr: str,
    pop: bool = False,
    required: Literal[True] = True,
    default_language=None,
) -> TranslatedString: ...


@overload
def make_trans_string(
    config: dict[str, Any],
    attr: str,
    pop: bool = False,
    required: Literal[False] = False,
    default_language=None,
) -> TranslatedString | None: ...


def make_trans_string(  # noqa: C901, PLR0912
    config: dict[str, Any],
    attr: str,
    pop: bool = False,
    required: bool = False,
    default_language=None,
) -> TranslatedString | None:
    ctx = get_i18n_context()
    assert ctx is not None
    default_language = default_language or ctx.default_language

    default = config.get(attr)
    if pop and default is not None:
        del config[attr]
    # If default is already a TranslatedString or a multi-language dict, use it directly
    if isinstance(default, TranslatedString):
        return default
    if isinstance(default, dict):
        return TranslatedString(default_language=default_language, **default)
    langs = {}
    if default is not None:
        langs[default_language] = default
    for key in list(config.keys()):
        m = re.match(r'%s_(([a-z]{2})(-[A-Z]{2})?)$' % attr, key)
        if m is None:
            continue
        full, lang, _region = m.groups()
        if full not in ctx.all_languages:
            matches = [x for x in ctx.all_languages if x.startswith('%s-' % lang)]
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
    return TranslatedString(**langs, default_language=default_language)


class InstanceLoader:
    instance: Instance
    context: Context
    default_language: str
    yaml_file_path: Path | None = None
    config: CommentedMap | dict[str, Any]
    fw_config: FrameworkConfig | None = None
    config_mtime_hash: str | None = None
    db_datasets: dict[str, DBDatasetModel] = {}
    db_dataset_refs: dict[str, Any] = {}
    dataset_payload_store: Any = None
    supplied_dataset_payload_refs: list[Any] | None = None

    _node_classes: dict[str, type[Node]]
    _input_nodes: dict[str, list[dict[str, Any] | str]]
    _output_nodes: dict[str, list[dict[str, Any] | str]]
    _subactions: dict[str, list[str]]
    _scenario_values: dict[str, list[tuple[Parameter, Any]]]
    _node_visualizations: dict[str, list[dict[str, Any]]]
    # Snapshot-path dataset-binding stash (see _stash_snapshot_bindings).
    _snapshot_dataset_ports: dict[UUID, list[DatasetPortSnapshot]]

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

            return cast('InstanceLoaderFuncT[P, R, SC]', wrapper)

        return wrap_with_span_outer

    def simple_trans_string(self, s: str) -> TranslatedString:
        langs = {
            self.default_language: s,
        }
        return TranslatedString(**langs, default_language=self.default_language)

    def _make_node_datasets(self, config: dict[str, Any], node_class: type[Node], unit: Unit | None) -> list[Dataset]:  # noqa: C901, PLR0912, PLR0915
        from nodes.datasets import DBDataset, DVCDataset, FixedDataset, GenericDataset
        from nodes.defs.node_defs import InputDatasetDef
        from nodes.generic import GenericNode
        from nodes.simple import AdditiveNode

        ds_config = config.get('input_datasets')
        datasets: list[Dataset] = []

        # If the graph doesn't specify input datasets, the node
        # might.
        if ds_config is None:
            ds_config = getattr(node_class, 'input_datasets', [])
        elif isinstance(ds_config, list):
            import copy

            ds_config = copy.deepcopy(ds_config)

        # Two sources of interpolation, and they behave differently on purpose: the legacy
        # `input_dataset_processors` entry forces it on for every binding, while a class
        # default is only a default and yields to a binding that says `interpolate: false`.
        # See docs/plans/additive-multiplicative-modernization.md.
        class_interpolate = node_class.interpolates_input_datasets_by_default
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
                ds_def = InputDatasetDef(id=ds, interpolate=ds_interpolate or class_interpolate)
            else:
                ds_def = InputDatasetDef.model_validate(ds)
                if ds_interpolate:
                    ds_def.interpolate = True
                elif class_interpolate and 'interpolate' not in ds:
                    ds_def.interpolate = True

            ds_obj: Dataset | None = None
            if issubclass(node_class, GenericNode) and not issubclass(node_class, AdditiveNode):
                ds_obj = GenericDataset.from_def(ds_def, self.context)

            use_framework_ds = 'framework_measure_data' in ds_def.tags
            use_obs_ds = 'observation_dataset' in ds_def.tags
            use_city_ds = 'city_data' in ds_def.tags
            if use_obs_ds:
                from frameworks.datasets import ObservationDataset

                ds_obj = ObservationDataset.from_def(ds_def, self.context)
            elif use_city_ds:
                from frameworks.datasets import FrameworkMeasureDVCDataset2

                # Prefer a DB-stored dataset when one exists for this instance.
                # FrameworkMeasureDVCDataset2 handles both cases: when db_dataset_obj is
                # provided it loads from DB, otherwise falls through to DVC. Either way,
                # post_process() runs and handles the uuid dimension and any framework
                # measure datapoint overrides correctly.
                # Future: if both a DB dataset and framework measure datapoints exist,
                # the DB values should be loaded first and then overridden by the
                # framework measures where available. That case doesn't arise yet, so
                # for now the DB dataset simply wins outright.
                ds_db_obj = None
                payload_ref = None
                if self.instance.features.use_datasets_from_db:
                    ds_db_obj = self.db_datasets.get(ds_def.id)
                    payload_ref = self.db_dataset_refs.get(ds_def.id)
                ds_obj = FrameworkMeasureDVCDataset2.from_def(
                    ds_def,
                    self.context,
                    db_dataset_obj=ds_db_obj,
                    payload_ref=payload_ref,
                    payload_store=self.dataset_payload_store,
                )
            elif self.fw_config is not None:
                from nodes.gpc import DatasetNode

                if issubclass(node_class, DatasetNode) or use_framework_ds:
                    from frameworks.datasets import FrameworkMeasureDVCDataset

                    ds_obj = FrameworkMeasureDVCDataset.from_def(ds_def, self.context)
            elif use_framework_ds:
                from frameworks.datasets import FrameworkMeasureDVCDataset

                ds_obj = FrameworkMeasureDVCDataset.from_def(ds_def, self.context)
            elif self.instance.features.use_datasets_from_db:
                ds_db_obj = self.db_datasets.get(ds_def.id)
                payload_ref = self.db_dataset_refs.get(ds_def.id)
                if payload_ref is not None:
                    from nodes.datasets import SerializedDBDataset

                    assert self.dataset_payload_store is not None
                    ds_obj = SerializedDBDataset.from_def(
                        ds_def,
                        self.context,
                        payload_ref=payload_ref,
                        payload_store=self.dataset_payload_store,
                    )
                elif ds_db_obj is not None:
                    ds_obj = DBDataset.from_def(ds_def, self.context, db_dataset_obj=ds_db_obj)

            if ds_obj is None:
                ds_obj = DVCDataset.from_def(ds_def, self.context)
            ds_obj.interpolate = ds_interpolate or ds_def.interpolate
            datasets.append(ds_obj)

        if 'historical_values' in config or 'forecast_values' in config:
            fds = FixedDataset(
                config['id'],
                self.context,
                unit=unit,  # type: ignore
                tags=config.get('tags', []),
                historical=config.get('historical_values'),
                forecast=config.get('forecast_values'),
                use_interpolation=ds_interpolate,
            )
            datasets.append(fds)
        return datasets

    def _init_failure(self, node: Node, msg: str, *, cause: Exception | None = None) -> None:
        """
        Handle a node init-phase (construction) failure.

        In tolerant (draft) mode, record the failure on the node and return so the caller can
        skip the offending piece and keep loading the rest of the graph. Otherwise raise a
        structured ``NodeError``. See ``docs/architecture/fault-tolerance.md``.
        """
        from nodes.node import NodeErrorPhase, NodeStatus, NodeStatusError

        if self.context.tolerate_node_failures:
            node.mark_status(NodeStatus.FAILED, NodeStatusError(phase=NodeErrorPhase.INITIALIZATION, message=msg))
            self.logger.warning('Node %s failed to initialize: %s' % (node.id, msg))
            return
        raise NodeError(node, msg) from cause

    def _make_node_params(self, config: dict[str, Any], node: Node) -> None:  # noqa: C901, PLR0912, PLR0915
        from params.base import Parameter
        from params.param import ReferenceParameter

        params = config.get('params', [])
        if not params:
            return
        if isinstance(params, dict):
            params = [dict(id=param_id, value=value) for param_id, value in params.items()]
        # Ensure that the node class allows these parameters
        node_class = type(node)
        class_allowed_params: dict[str, Parameter] = {p.local_id: p for p in getattr(node_class, 'allowed_parameters', [])}
        for pc in params:
            param_id = pc.pop('id')

            param_obj = class_allowed_params.get(param_id)
            if param_obj is None:
                self._init_failure(node, 'Parameter %s not allowed by node class' % param_id)
                continue
            param_class = type(param_obj)

            label = make_trans_string(pc, 'label', pop=True, required=False) or param_obj.label
            ref = pc.pop('ref', None)
            description = make_trans_string(pc, 'description', pop=True, required=False) or param_obj.description
            is_customizable = pc.pop('is_customizable', None)

            scenario_values = pc.pop('values', {})

            if ref is not None:
                target = self.context.global_parameters.get(ref)
                if target is None:
                    self._init_failure(node, 'Parameter %s refers to an unknown global parameter: %s' % (param_id, ref))
                    continue

                if not isinstance(target, param_class):
                    self._init_failure(
                        node,
                        'Node requires parameter of type %s, but referenced parameter %s is %s'
                        % (
                            param_class,
                            ref,
                            type(target),
                        ),
                    )
                    continue
                ref_param = ReferenceParameter(
                    local_id=param_obj.local_id,
                    label=param_obj.label,
                    target_id=target.global_id,
                )
                node.add_parameter(ref_param)
                continue

            # Merge parameter values
            fields = param_obj.model_dump()
            fields.update(pc)
            if description is not None:
                fields['description'] = description
            if label is not None:
                fields['label'] = label
            if is_customizable is not None:
                fields['is_customizable'] = is_customizable

            unit = fields.get('unit', None)
            if unit is not None and isinstance(unit, str):
                fields['unit'] = self.context.unit_registry.parse_units(unit)

            value = fields.pop('value', None)
            param = param_class(**fields)
            assert isinstance(param, Parameter)
            node.add_parameter(param)

            try:
                if value is not None:
                    param.value = param.clean(value)
            except Exception as e:
                self._init_failure(node, 'Error setting parameter %s: %s' % (param.local_id, e), cause=e)
                continue

            for scenario_id, value in scenario_values.items():
                try:
                    cleaned = param.clean(value)
                except Exception as e:
                    self._init_failure(node, 'Invalid scenario value for parameter %s: %s' % (param.local_id, e), cause=e)
                    continue
                sv = self._scenario_values.setdefault(scenario_id, list())
                sv.append((param, cleaned))

    def _make_node_visualizations(self, node: Node, config: list[dict[str, Any]]) -> None:
        from nodes.visualizations import NodeVisualizations

        ctx = NodeVisualizations.ValidationContext(context=self.context, node=None, root_node=node)
        try:
            node.visualizations = NodeVisualizations.model_validate(config, context=ctx)
        except Exception as e:
            # Tolerant mode leaves visualizations at their default (a node without viz config is fine).
            self._init_failure(node, 'Error validating visualizations: %s' % e, cause=e)

    def make_node(  # noqa: C901, PLR0912, PLR0915
        self, node_class: type[Node], config: dict[str, Any], _yaml_lc: LineCol | None = None
    ) -> Node:
        from nodes.node import NodeMetric
        from nodes.units import Unit

        metrics_conf = config.get('output_metrics')
        metrics: dict[str, NodeMetric] | None
        if metrics_conf is not None:
            metrics = {m['id']: NodeMetric.from_config(m) for m in metrics_conf}
            # If the class defines output_metrics, remap config keys to match
            # class keys (e.g. class uses 'default' but config has 'Value').
            # The config key is the port's column_id; match it against the
            # class metric's column_id to find the canonical dict key.
            class_metrics_def: dict[str, NodeMetric] | None = getattr(node_class, 'output_metrics', None)
            if class_metrics_def:
                col_to_class_key = {m.column_id: k for k, m in class_metrics_def.items()}
                remapped: dict[str, NodeMetric] = {}
                for config_key, metric in metrics.items():
                    class_key = col_to_class_key.get(config_key, config_key)
                    remapped[class_key] = metric
                metrics = remapped
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

        loc_conf: ConfigLocation | None = config.get('config_location')
        config_location = ConfigLocation(**loc_conf) if loc_conf else None

        description = make_trans_string(config, 'description')
        node: Node = node_class(
            id=config['id'],
            context=self.context,
            name=make_trans_string(config, 'name'),
            short_name=make_trans_string(config, 'short_name'),
            quantity=quantity,
            unit=unit,
            node_group=config.get('node_group'),
            description=description,
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
            config_location=config_location,
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
            if all(isinstance(tag, str) for tag in tags):
                node.tags.update(tags)
            else:
                self._init_failure(node, "'tags' must be a list of strings")

        viz_config = config.get('visualizations')
        if viz_config:
            self._node_visualizations[node.id] = viz_config

        no_effect_value = config.get('no_effect_value')
        if no_effect_value is not None:
            assert isinstance(node, ActionNode)
            node.no_effect_value = no_effect_value

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
        klass: type = getattr(mod, class_name)
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

        if self.snapshot is not None:
            # Same dict shape as the YAML path, but sourced from the typed
            # spec; copied so the shared snapshot is never mutated.
            dim_configs: list[dict[str, Any]] = [dict(dc) for dc in self.snapshot.spec.dimensions]
        else:
            dim_configs = self.config.get('dimensions', [])
        for dc in dim_configs:
            try:
                dc['mtime_hash'] = self.config_mtime_hash
                dim = Dimension.from_yaml_config(dc)
            except Exception:
                print(dc)
                raise
            assert dim.id not in self.context.dimensions
            self.context.dimensions[dim.id] = dim

    def _import_node_class_for_spec(self, spec: NodeSpec, identifier: str, *, action: bool) -> type[Node]:
        from nodes.actions.action import ActionNode
        from nodes.defs import ActionConfig, FormulaConfig, SimpleConfig
        from nodes.node import Node

        tc = spec.type_config
        if isinstance(tc, FormulaConfig):
            type_path = 'formula.FormulaNode'
        elif isinstance(tc, (ActionConfig, SimpleConfig)):
            type_path = tc.node_class
        else:
            raise TypeError(f'Unknown node type config: {type(tc)}')
        prefix = None if type_path.startswith('nodes.') else ('nodes.actions' if action else 'nodes')
        return self.import_class(
            type_path,
            prefix,
            allowed_classes=[ActionNode] if action else [Node],
            disallowed_classes=None if action else [ActionNode],
            node_id=identifier,
        )

    def _setup_nodes_from_snapshot(self, *, actions: bool) -> None:
        from nodes.actions.action import ActionNode
        from nodes.defs import ActionConfig
        from nodes.defs.node_defs import NodeKind

        assert self.snapshot is not None
        for n in self.snapshot.nodes:
            spec = n.spec
            assert spec is not None
            if (spec.kind == NodeKind.ACTION) != actions:
                continue
            assert n.identifier is not None
            node_class = self._import_node_class_for_spec(spec, n.identifier, action=actions)
            node = self.make_node_from_snapshot(node_class, n)
            if actions:
                assert isinstance(node, ActionNode)
                tc = spec.type_config
                assert isinstance(tc, ActionConfig)
                if tc.decision_level is not None:
                    node.decision_level = tc.decision_level
                if tc.group is not None:
                    ag = next((ag for ag in self.instance.action_groups if ag.id == tc.group), None)
                    if ag is None:
                        self._init_failure(node, "Action group '%s' not found" % tc.group)
                    else:
                        node.group = ag
                if tc.parent is not None:
                    self._subactions.setdefault(tc.parent, []).append(node.id)
            self.context.add_node(node)

    def _resolve_output_metrics(  # noqa: C901
        self, node_class: type[Node], spec: NodeSpec, identifier: str
    ) -> tuple[dict[str, NodeMetric] | None, Any, str | None]:
        """Output metrics / unit / quantity from typed output ports, with the dict path's class fallbacks."""
        from nodes.node import NodeMetric
        from nodes.units import Unit

        metrics: dict[str, NodeMetric] | None = None
        unit: Any = None
        quantity: str | None = None
        if len(spec.output_ports) == 1:
            port = spec.output_ports[0]
            unit = port.unit
            quantity = port.quantity
        elif len(spec.output_ports) > 1:
            metrics = {}
            for port in spec.output_ports:
                column = str(port.column_id) if port.column_id is not None else None
                if column is None:
                    raise Exception('Node %s: multi-metric output port without column_id' % identifier)
                if port.quantity is None:
                    raise Exception('Node %s: output metric %s has no quantity' % (identifier, column))
                assert port.unit is not None
                metrics[column] = NodeMetric(unit=port.unit, quantity=port.quantity, id=column, column_id=column)
            # If the class defines output_metrics, remap keys to the class's
            # canonical keys by column_id (same as the dict path).
            class_metrics_def: dict[str, NodeMetric] | None = getattr(node_class, 'output_metrics', None)
            if class_metrics_def:
                col_to_class_key = {m.column_id: k for k, m in class_metrics_def.items()}
                metrics = {col_to_class_key.get(key, key): metric for key, metric in metrics.items()}

        class_metrics = None if metrics is not None else getattr(node_class, 'output_metrics', None)
        if unit is None:
            unit = getattr(node_class, 'default_unit', None) or getattr(node_class, 'unit', None)
            if not unit and not metrics and not class_metrics:
                raise Exception('Node %s (%s) has no unit set' % (identifier, node_class.__name__))
        if unit and not isinstance(unit, Unit):
            unit = self.context.unit_registry.parse_units(unit)
        if quantity is None:
            quantity = getattr(node_class, 'quantity', None)
            if not quantity and not metrics and not class_metrics:
                raise Exception('Node %s (%s) has no quantity set' % (identifier, node_class.__name__))
        return metrics, unit, quantity

    def make_node_from_snapshot(self, node_class: type[Node], n: NodeSnapshot) -> Node:  # noqa: C901
        """Native ``make_node`` twin: construct a runtime node from typed snapshot state."""
        spec = n.spec
        assert spec is not None
        identifier = n.identifier
        assert identifier is not None

        def ts(val: I18nString | None) -> TranslatedString | None:
            if val is None or isinstance(val, TranslatedString):
                return val
            return self.simple_trans_string(str(val))

        metrics, unit, quantity = self._resolve_output_metrics(node_class, spec, identifier)

        extra = spec.extra
        ds_fragment: dict[str, Any] = {'id': identifier}
        dataset_ports = self._snapshot_dataset_ports.get(n.uuid)
        if dataset_ports:
            from nodes.instance_from_db import _serialize_dataset_ports

            ds_fragment['input_datasets'] = _serialize_dataset_ports(dataset_ports)
        if extra.input_dataset_processors:
            ds_fragment['input_dataset_processors'] = list(extra.input_dataset_processors)
        if extra.historical_values:
            ds_fragment['historical_values'] = extra.historical_values
        if extra.forecast_values:
            ds_fragment['forecast_values'] = extra.forecast_values
        if extra.tags:
            ds_fragment['tags'] = list(extra.tags)
        datasets = self._make_node_datasets(ds_fragment, node_class, unit)

        node: Node = node_class(
            id=identifier,
            context=self.context,
            # A missing name fails inside the Node constructor, like the dict path.
            name=cast('TranslatedString', ts(n.name)),
            short_name=ts(n.short_name),
            quantity=quantity,
            unit=unit,
            node_group=spec.node_group,
            # The dict path maps the snapshot's short_description onto the
            # runtime description; keep that projection.
            description=ts(n.short_description),
            color=n.color or None,
            order=n.order,
            is_visible=n.is_visible,
            is_outcome=spec.is_outcome,
            minimum_year=spec.minimum_year,
            target_year_goal=extra.other.get('target_year_goal'),
            goals=spec.goals.model_dump(exclude_none=True) if spec.goals.root else None,
            allow_nulls=spec.allow_nulls,
            input_datasets=datasets,
            output_dimension_ids=spec.output_dimensions or None,
            input_dimension_ids=spec.input_dimensions or None,
            output_metrics=metrics,
            config_location=None,
        )
        if node.id in self._input_nodes or node.id in self._output_nodes:
            raise Exception('Node %s is already configured' % node.id)
        # Edges are constructed natively from snapshot bindings in
        # _setup_edges_from_snapshot; nothing is stashed for Edge.from_config.
        self._input_nodes[node.id] = []
        self._output_nodes[node.id] = []

        if spec.params:
            from nodes.instance_from_db import _param_to_dict

            self._make_node_params({'params': [_param_to_dict(p) for p in spec.params]}, node)

        if extra.tags:
            node.tags.update(extra.tags)

        if spec.visualizations.root:
            # Dump rather than pass the instance: model_validate short-circuits
            # on same-type instances and the context-dependent refs would not
            # be resolved against this context.
            self._node_visualizations[node.id] = spec.visualizations.model_dump(exclude_none=True)

        from nodes.defs import ActionConfig

        tc = spec.type_config
        if isinstance(tc, ActionConfig) and tc.no_effect_value is not None:
            assert isinstance(node, ActionNode)
            node.no_effect_value = tc.no_effect_value

        return node

    @wrap_with_span('setup-nodes', 'function')
    def setup_nodes(self):
        from nodes.actions.action import ActionNode
        from nodes.node import Node

        if self.snapshot is not None:
            self._setup_nodes_from_snapshot(actions=False)
            return
        for nc in self.config.get('nodes', []):
            if nc['type'].startswith('nodes.'):
                prefix = None
            else:
                prefix = 'nodes'
            try:
                node_class = self.import_class(
                    nc['type'],
                    prefix,
                    allowed_classes=[Node],
                    disallowed_classes=[ActionNode],
                    node_id=nc['id'],
                )
            except ImportError:
                self.logger.error('Unable to import node class for %s' % nc.get('id'))
                raise
            node = self.make_node(node_class, nc, _yaml_lc=getattr(nc, 'lc', None))
            self.context.add_node(node)

    def generate_nodes_from_emission_sectors(self):
        from nodes.simple import SectorEmissions

        if not self.config.get('emission_sectors'):
            return

        node_class = self.import_class(
            'SectorEmissions',
            'nodes.simple',
            allowed_classes=[SectorEmissions],
        )
        dataset_id = self.config.get('emission_dataset')
        emission_unit = self.config.get('emission_unit')
        assert emission_unit is not None
        emission_unit = self.context.unit_registry.parse_units(emission_unit)
        dims = self.config.get('emission_dimensions', [])

        for ec in self.config.get('emission_sectors', []):
            assert isinstance(ec, dict)
            parent_id = ec.pop('part_of', None)
            data_col = ec.pop('column', None)
            data_category = ec.pop('category', None)
            unit = ec.pop('unit', emission_unit)
            dim_i = ec.pop('input_dimensions', dims)
            dim_o = ec.pop('output_dimensions', dims)
            if 'name_en' in ec and 'emissions' not in ec['name_en']:
                ec['name_en'] += ' emissions'
            nc = dict(
                output_nodes=[parent_id] if parent_id else [],
                input_dimensions=dim_i,
                output_dimensions=dim_o,
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
                unit=unit,
                params=dict(category=data_category) if data_category else [],
                **ec,
            )
            node = self.make_node(node_class, nc, _yaml_lc=getattr(ec, 'lc', None))
            self.context.add_node(node)

    @wrap_with_span('setup-actions', 'function')
    def setup_actions(self):
        from nodes.actions.action import ActionNode

        if self.snapshot is not None:
            self._setup_nodes_from_snapshot(actions=True)
            return
        for nc in self.config.get('actions', []):
            if nc['type'].startswith('nodes.'):
                prefix = None
            else:
                prefix = 'nodes.actions'
            node_class = self.import_class(
                nc['type'],
                prefix,
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
                    self._init_failure(node, 'Invalid decision level: %s' % decision_level)

            ag_id = nc.get('group', None)
            if ag_id is not None:
                assert isinstance(ag_id, str)
                ag = next((ag for ag in self.instance.action_groups if ag.id == ag_id), None)
                if ag is None:
                    self._init_failure(node, "Action group '%s' not found" % ag_id)
                else:
                    node.group = ag

            parent_id = nc.get('parent', None)
            if parent_id is not None:
                subs = self._subactions.setdefault(parent_id, [])
                subs.append(node.id)

            self.context.add_node(node)

    def _require_dimension(self, dim_id: str, node: Node) -> Any:
        dim = self.context.dimensions.get(dim_id)
        if dim is None:
            raise NodeError(node, 'dimension %s not found' % dim_id)
        return dim

    def _edge_dimensions_from_transforms(
        self,
        transforms: list[Any],
        required_dimensions: Sequence[str],
        node: Node,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Typed transforms into runtime ``EdgeDimension`` maps (from_dims, to_dims).

        Mirrors ``_transforms_to_config`` + ``EdgeDimension.from_config``: the
        target port's declared dimensions (plus legacy ``FlattenTransformation``
        rows preserved in pre-step-2 snapshots) become bare exclude+flatten
        declarations, filters resolve categories and groups, and assigns pin
        one category.
        """
        from nodes.defs.transform_def import (
            AssignDimensionOp,
            FilterDimensionOp,
            FlattenTransformation,
            modernized_transformations,
        )
        from nodes.edges import EdgeDimension

        legacy_declared = [t.dimension for t in transforms if isinstance(t, FlattenTransformation)]
        declared = list(dict.fromkeys([*required_dimensions, *legacy_declared]))
        to_dims: dict[str, EdgeDimension] = {}
        for dim_id in declared:
            self._require_dimension(dim_id, node)
            to_dims[dim_id] = EdgeDimension(categories=[], exclude=True, flatten=True)

        from_dims: dict[str, EdgeDimension] = {}
        for t in modernized_transformations(transforms):
            match t:
                case FilterDimensionOp():
                    dim = self._require_dimension(t.dimension, node)
                    cat_ids = list(t.categories)
                    for gid in t.groups or ():
                        cat_ids += [cat.id for cat in dim.get_cats_for_group(gid)]
                    if not cat_ids:
                        # A filter with no category selection is a bare
                        # flatten declaration, exactly as the dict round-trip
                        # produced regardless of the stored flatten flag.
                        from_dims[t.dimension] = EdgeDimension(categories=[], exclude=True, flatten=True)
                    else:
                        from_dims[t.dimension] = EdgeDimension(
                            categories=[dim.get(cat_id) for cat_id in cat_ids],
                            exclude=bool(t.exclude),
                            flatten=bool(t.flatten),
                        )
                case AssignDimensionOp():
                    dim = self._require_dimension(t.dimension, node)
                    to_dims[t.dimension] = EdgeDimension(categories=[dim.get(t.category)], exclude=False, flatten=False)
                case _:
                    raise ValueError(f'Edge transformation "{t.kind}" is not executable by the legacy edge runtime')
        return from_dims, to_dims

    def _setup_edges_from_snapshot(self) -> None:
        """Construct runtime edges from snapshot bindings, grouped per node pair like the dict path."""
        from collections import defaultdict

        from nodes.constants import VALUE_COLUMN

        snapshot = self.snapshot
        assert snapshot is not None
        specs_by_uuid: dict[UUID, NodeSpec] = {}
        identifiers_by_uuid: dict[UUID, str] = {}
        for n in snapshot.nodes:
            assert n.spec is not None
            assert n.identifier is not None
            specs_by_uuid[n.uuid] = n.spec
            identifiers_by_uuid[n.uuid] = n.identifier

        # One runtime Edge per (from, to) pair; parallel bindings deliver the
        # source's metrics. Grouping preserves binding order.
        groups: dict[tuple[UUID, UUID], list[tuple[str, EdgeSnapshot]]] = {}
        for e in snapshot.edge_bindings:
            from_port = specs_by_uuid[e.from_node].output_port_by_id[e.from_port]
            groups.setdefault((e.from_node, e.to_node), []).append((from_port.column_id or VALUE_COLUMN, e))

        # Per target, edge groups apply in the target's input-port declaration
        # order (stable within a port: binding order).
        per_target: defaultdict[UUID, list[tuple[int, tuple[UUID, UUID]]]] = defaultdict(list)
        for (from_id, to_id), tuples in groups.items():
            to_spec = specs_by_uuid[to_id]
            port_order = {port.id: idx for idx, port in enumerate(to_spec.input_ports)}
            first = tuples[0][1]
            per_target[to_id].append((port_order.get(first.to_port, len(port_order)), (from_id, to_id)))

        ctx = self.context
        uuid_by_identifier = {identifier: node_id for node_id, identifier in identifiers_by_uuid.items()}
        for node in ctx.nodes.values():
            to_id = uuid_by_identifier.get(node.id)
            if to_id is None:
                continue
            for _port_idx, (from_id, group_to_id) in sorted(per_target.get(to_id, []), key=lambda item: item[0]):
                tuples = groups[(from_id, group_to_id)]
                try:
                    edge = self._make_edge_from_group(from_id, group_to_id, tuples, specs_by_uuid, identifiers_by_uuid)
                    node.add_edge(edge)
                    edge.input_node.add_edge(edge)
                except Exception as e:
                    self._init_failure(node, 'Invalid input edge: %s' % e, cause=e)

    def _make_edge_from_group(
        self,
        from_id: UUID,
        to_id: UUID,
        tuples: list[tuple[str, EdgeSnapshot]],
        specs_by_uuid: dict[UUID, NodeSpec],
        identifiers_by_uuid: dict[UUID, str],
    ) -> Edge:
        from nodes.edges import Edge

        first = tuples[0][1]
        transforms = first.transformations
        tags = first.tags
        metrics: list[str] = []
        for column_id, e in tuples:
            if transforms:
                assert e.transformations == transforms
            if tags:
                assert tuple(e.tags) == tuple(tags)
            metrics.append(column_id)

        input_node = self.context.get_node(identifiers_by_uuid[from_id])
        output_node = self.context.get_node(identifiers_by_uuid[to_id])
        to_port = specs_by_uuid[to_id].input_port_by_id[first.to_port]
        from_dims, to_dims = self._edge_dimensions_from_transforms(
            list(transforms),
            to_port.required_dimensions,
            output_node,
        )
        # Only deliver explicit `metrics` when the source is multi-metric; for
        # single-output nodes an empty list keeps the pass-through code path.
        from_is_multi_metric = len(specs_by_uuid[from_id].output_ports) > 1
        return Edge(
            input_node=input_node,
            output_node=output_node,
            tags=list(tags),
            from_dimensions=from_dims,
            to_dimensions=to_dims or None,
            metrics=metrics if from_is_multi_metric else [],
        )

    def _setup_edges(self) -> None:
        from nodes.edges import Edge

        ctx = self.context
        if self.snapshot is not None:
            self._setup_edges_from_snapshot()
            return
        for node in ctx.nodes.values():
            for ec in self._output_nodes.get(node.id, []):
                try:
                    edge = Edge.from_config(ec, node=node, is_output=True, context=ctx)
                    node.add_edge(edge)
                    edge.output_node.add_edge(edge)
                except Exception as e:
                    self._init_failure(node, 'Invalid output edge: %s' % e, cause=e)

            for ec in self._input_nodes.get(node.id, []):
                try:
                    edge = Edge.from_config(ec, node=node, is_output=False, context=ctx)
                    node.add_edge(edge)
                    edge.input_node.add_edge(edge)
                except Exception as e:
                    self._init_failure(node, 'Invalid input edge: %s' % e, cause=e)

    def _setup_subactions(self) -> None:
        from nodes.actions.action import ActionNode
        from nodes.actions.parent import ParentActionNode

        ctx = self.context
        for parent_id, subs in self._subactions.items():
            parent = ctx.nodes.get(parent_id)
            if parent is None:
                # No parent node to attribute to; record on each subaction that references it.
                for sub_id in subs:
                    sub = ctx.nodes.get(sub_id)
                    if sub is not None:
                        self._init_failure(sub, "Parent action '%s' not found" % parent_id)
                continue
            if not isinstance(parent, ParentActionNode):
                self._init_failure(parent, "Action '%s' is marked as a parent but is not a ParentActionNode" % parent_id)
                continue
            for sub_id in subs:
                node = ctx.get_node(sub_id)
                assert isinstance(node, ActionNode)
                parent.add_subaction(node)
                node.parent_action = parent

    @wrap_with_span('setup-edges', 'function')
    def setup_edges(self) -> None:
        # Setup edges
        self._setup_edges()
        self._setup_subactions()
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
            MeasureDataPoint.objects
            .filter(measure__framework_config=fwc)
            .filter(value__isnull=False)
            .order_by()
            .values_list('year', flat=True)
            .distinct('year')
        )
        pt_scenario.actual_historical_years = list(years)

    def _snapshot_scenarios(self) -> list[Scenario]:
        """Runtime scenarios from the typed spec (param values re-cleaned like the dict path)."""
        from nodes.scenario import Scenario, ScenarioKind

        assert self.snapshot is not None
        if not self.snapshot.spec.scenarios:
            fallback = Scenario(id='default', name=TranslatedString(_('Default')), kind=ScenarioKind.DEFAULT)
            fallback._context = self.context
            return [fallback]
        scenarios: list[Scenario] = []
        for sc in self.snapshot.spec.scenarios:
            kind = sc.kind
            if kind is None:
                if sc.id == 'progress_tracking':
                    kind = ScenarioKind.PROGRESS_TRACKING
                elif sc.id == 'baseline':
                    kind = ScenarioKind.BASELINE
            scenario = Scenario(
                id=sc.id,
                name=sc.name,
                description=sc.description,
                kind=kind,
                all_actions_enabled=sc.all_actions_enabled,
                is_selectable=sc.is_selectable,
                actual_historical_years=list(sc.actual_historical_years) if sc.actual_historical_years is not None else None,
            )
            scenario._context = self.context
            for param_id, value in sc.param_values.items():
                param = self.context.get_parameter(param_id)
                scenario.add_parameter(param, param.clean(value))
            scenarios.append(scenario)
        return scenarios

    def _config_scenarios(self) -> list[Scenario]:
        """Runtime scenarios from YAML-shaped config dicts."""
        from nodes.scenario import Scenario, ScenarioKind

        scenario_confs: list[dict[str, Any]] = self.config.get('scenarios', [])
        if not scenario_confs:
            scenario_confs = [
                {
                    'id': 'default',
                    'name': TranslatedString(_('Default')),
                    'default': True,
                }
            ]

        scenarios: list[Scenario] = []
        for sc in scenario_confs:
            name = make_trans_string(sc, 'name', pop=True)
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
            scenario = Scenario(id=scenario_id, name=name, actual_historical_years=actual_historical_years, kind=kind, **sc)
            scenario._context = self.context

            for pc in params_config:
                param = self.context.get_parameter(pc['id'])
                scenario.add_parameter(param, param.clean(pc['value']))
            scenarios.append(scenario)
        return scenarios

    def setup_scenarios(self):
        from nodes.scenario import CustomScenario

        default_scenario = None

        scenarios = self._snapshot_scenarios() if self.snapshot is not None else self._config_scenarios()
        for scenario in scenarios:
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

        custom_scenario = CustomScenario(
            id='custom',
            name=_('Custom'),
            base_scenario=default_scenario,
        )

        self.context.set_custom_scenario(custom_scenario)

        if self.fw_config is not None:
            self.setup_progress_tracking_scenario()

    def setup_global_parameters(self):
        global_params = discover_global_parameters()

        context = self.context
        if self.snapshot is not None:
            for spec_param in self.snapshot.spec.params:
                if spec_param.local_id not in global_params:
                    raise Exception('Unknown global parameter: %s' % spec_param.local_id)
                param = spec_param.model_copy(deep=True)
                param.set_context(context)
                context.add_global_parameter(param)
            return
        for pc in self.config.get('params', []):
            param_id = pc.pop('id')
            pc['local_id'] = param_id
            unit_str = pc.get('unit', None)
            if unit_str is not None:
                unit = context.unit_registry.parse_units(unit_str)
                pc['unit'] = unit
            param = global_params.get(param_id)
            if param is None:
                raise Exception('Unknown global parameter: %s' % param_id)
            param_val = pc.pop('value', None)
            if 'is_customizable' not in pc:
                pc['is_customizable'] = False
            pc['label'] = make_trans_string(pc, 'label', pop=True)
            pc['description'] = make_trans_string(pc, 'description', pop=True)

            param_type = type(param)
            param = param_type(**pc)
            param.set_context(context)
            param.set(param_val)

            assert 'subscription_nodes' not in pc  # check for legacy

            context.add_global_parameter(param)

    def setup_impact_overviews(self):
        from nodes.actions.action import ImpactOverview
        from nodes.defs.action_def import ImpactOverviewSpec

        if self.snapshot is not None:
            seen: set[str] = set()
            for overview_spec in self.snapshot.spec.impact_overviews:
                spec = overview_spec.model_copy(deep=True)
                assert spec.id is not None
                if spec.id in seen:
                    raise ValueError(f"Duplicate impact overview id '{spec.id}'. Set an explicit 'id' field to disambiguate.")
                seen.add(spec.id)
                self.context.impact_overviews.append(ImpactOverview(spec, self.context))
            return

        conf = self.config.get('impact_overviews', [])
        seen_ids: set[str] = set()
        for aepc in conf:
            spec_config = dict(aepc)
            rename_map = {
                'effect_node': 'effect_node_id',
                'cost_node': 'cost_node_id',
                'stakeholder_dimension': 'stakeholder_dimension_id',
                'outcome_dimension': 'outcome_dimension_id',
            }
            for old_name, new_name in rename_map.items():
                if old_name in spec_config and new_name not in spec_config:
                    spec_config[new_name] = spec_config.pop(old_name)
            spec = ImpactOverviewSpec.from_yaml_config(spec_config)
            assert spec.id is not None
            if spec.id in seen_ids:
                raise ValueError(f"Duplicate impact overview id '{spec.id}'. Set an explicit 'id' field to disambiguate.")
            seen_ids.add(spec.id)
            aep = ImpactOverview(spec, self.context)
            self.context.impact_overviews.append(aep)

    def setup_normalizations(self):
        from paths.refs import ValidationContext

        from nodes.defs.instance_defs import NormalizationSpec
        from nodes.normalization import Normalization

        if self.snapshot is not None:
            # Re-validate against this context so node refs resolve here.
            spec_configs: list[dict[str, Any]] = [n.model_dump() for n in self.snapshot.spec.normalizations]
        else:
            spec_configs = []
            for nc in self.config.get('normalizations', []):
                spec_config = dict(nc)
                if 'normalizer_node' in spec_config and 'normalizer_node_id' not in spec_config:
                    spec_config['normalizer_node_id'] = spec_config.pop('normalizer_node')
                spec_configs.append(spec_config)
        for spec_config in spec_configs:
            normalization = Normalization(
                NormalizationSpec.model_validate(spec_config, context=ValidationContext(context=self.context)),
                self.context,
            )
            self.context.add_normalization(normalization.normalizer_node.id, normalization)

    def setup_validation_graph(self):
        config = self.config
        nodes = config.get('nodes')
        assert isinstance(nodes, list)

        all_nodes = []  # FIXME Or collect from context?
        all_nodes.extend(nodes)
        all_actions = config.get('actions')
        if all_actions is not None:
            all_nodes.extend(all_actions)

        emission_sectors = config.get('emission_sectors')
        if emission_sectors is not None:
            for es in emission_sectors:
                es['type'] = 'simple.SectorEmissions'
                es['unit'] = config.get('emission_unit')
                es['input_dimensions'] = config.get('emission_dimensions')
                es['output_dimensions'] = config.get('emission_dimensions')
                all_nodes.append(es)

        nes = NodeExplanationSystem(self.context, all_nodes)
        self.context.node_explanation_system = nes

    def setup_validations(self):
        nes = self.context.node_explanation_system
        assert nes is not None
        nes.generate_validations()
        nes.generate_input_baskets()
        nes.generate_explanations()

    @classmethod
    def from_dict_config(cls, config: dict[str, Any], fw_config: FrameworkConfig | None = None) -> Self:
        yaml_path = config.get('yaml_file_path')
        return cls(
            config=config,
            yaml_file_path=Path(yaml_path) if yaml_path else None,
            fw_config=fw_config,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: InstanceSnapshot,
        tolerate_node_failures: bool = False,
        *,
        published: bool = False,
    ) -> Self:
        """
        Build the runtime from an ``InstanceSnapshot`` (specs, not YAML dicts).

        The instance level (identity, years, features, dimensions, params,
        scenarios, impact overviews, normalizations) is constructed natively
        from the typed snapshot. Node/action/edge construction still consumes
        YAML-shaped dicts through the node-scope shim in
        ``nodes/instance_from_db.py``; that remainder migrates with the
        ``NodeMeta``-native node construction.
        """
        payload_refs = None
        if published:
            from nodes.datasets import DatasetPayloadRef

            payload_refs = [
                DatasetPayloadRef(
                    payload_id=pin.revision_id,
                    dataset_pk=0,
                    dataset_uuid=str(pin.dataset_uuid),
                    identifier=pin.identifier or str(pin.dataset_uuid),
                    content_hash=pin.content_hash,
                    generation=None,
                    forecast_from=pin.forecast_from,
                )
                for pin in snapshot.dataset_revisions
            ]
        return cls(
            snapshot=snapshot,
            tolerate_node_failures=tolerate_node_failures,
            dataset_payload_refs=payload_refs,
        )

    @classmethod
    def from_yaml(
        cls,
        filename: Path,
        fw_config: FrameworkConfig | None = None,
        tolerate_node_failures: bool = False,
    ) -> Self:
        yaml_fn = filename.resolve()
        yaml_conf = InstanceYAMLConfig.load_for_entrypoint(yaml_fn)

        data = yaml_conf.data
        assert data is not None
        return cls(
            config=data,
            yaml_file_path=yaml_fn,
            fw_config=fw_config,
            config_mtime_hash=yaml_conf.meta.mtime_hash,
            tolerate_node_failures=tolerate_node_failures,
        )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        yaml_file_path: Path | None = None,
        fw_config: FrameworkConfig | None = None,
        config_mtime_hash: str | None = None,
        tolerate_node_failures: bool = False,
        dataset_payload_refs: list[Any] | None = None,
        *,
        snapshot: InstanceSnapshot | None = None,
    ):
        from .units import add_unit_translations

        add_unit_translations()
        self.tolerate_node_failures = tolerate_node_failures
        self.supplied_dataset_payload_refs = dataset_payload_refs
        self.config_mtime_hash = config_mtime_hash
        self._node_classes = {}
        self.snapshot = snapshot
        if snapshot is not None:
            assert config is None
            assert fw_config is None
            assert yaml_file_path is None
            self.yaml_file_path = None
            self.fw_config = None
            # Node/action/edge construction still reads dicts; the node-scope
            # shim fills these two keys and nothing else.
            from nodes.instance_from_db import snapshot_nodes_to_config_dicts

            nodes_list, actions_list = snapshot_nodes_to_config_dicts(snapshot)
            self.config = {'nodes': nodes_list, 'actions': actions_list}
            self.default_language = snapshot.metadata.primary_language
            self.other_languages = list(snapshot.metadata.other_languages)
            self.logger = logger.bind(instance=snapshot.metadata.identifier)
            with set_i18n_context(self.default_language, self.other_languages):
                self._init_instance_from_snapshot(snapshot)
            return
        assert config is not None
        self.yaml_file_path = yaml_file_path.absolute() if yaml_file_path else None
        self.config = config
        self.fw_config = fw_config
        self.default_language = config['default_language']
        self.other_languages = config.get('supported_languages', [])
        self.logger = logger.bind(instance=config['id'])
        with set_i18n_context(self.default_language, self.other_languages):
            self._init_instance()

    def setup_node_visualizations(self):
        for node_id, viz_config in self._node_visualizations.items():
            node = self.context.get_node(node_id)
            self._make_node_visualizations(node, viz_config)

    def load_db_datasets(self):
        from kausal_common.datasets.models import Dataset as DBDatasetModel

        from nodes.models import InstanceConfig

        if self.supplied_dataset_payload_refs is not None:
            from nodes.datasets import RevisionDatasetPayloadStore

            self.db_datasets = {}
            self.db_dataset_refs = {ref.identifier: ref for ref in self.supplied_dataset_payload_refs}
            self.dataset_payload_store = RevisionDatasetPayloadStore(self.supplied_dataset_payload_refs)
            return

        try:
            ic = self.instance.config
        except InstanceConfig.DoesNotExist:
            self.db_datasets = {}
            return
        ds_objs = list(
            DBDatasetModel.objects.qs
            .for_instance_config(ic)
            .filter(is_external_placeholder=False, identifier__isnull=False)
            .only('uuid', 'identifier', 'last_modified_at', 'spec', 'is_external_placeholder')
        )
        self.db_datasets = {cast('str', ds.identifier): ds for ds in ds_objs}
        from nodes.dataset_materialization import ensure_dataset_materializations
        from nodes.datasets import CurrentDatasetPayloadStore, DatasetPayloadRef

        by_dataset = ensure_dataset_materializations(ds_objs)
        refs: list[DatasetPayloadRef] = []
        self.db_dataset_refs = {}
        for dataset in ds_objs:
            materialization = by_dataset.get(dataset.pk)
            if materialization is None:
                raise RuntimeError(f'Dataset {dataset.uuid} could not be materialized')
            assert dataset.identifier is not None
            ref = DatasetPayloadRef(
                payload_id=materialization.pk,
                dataset_pk=dataset.pk,
                dataset_uuid=str(dataset.uuid),
                identifier=dataset.identifier,
                content_hash=materialization.content_hash,
                generation=materialization.generation,
                forecast_from=materialization.forecast_from,
            )
            refs.append(ref)
            self.db_dataset_refs[dataset.identifier] = ref
        self.dataset_payload_store = CurrentDatasetPayloadStore(refs)

    def _init_instance(self) -> None:
        from nodes.context import Context
        from nodes.defs.instance_defs import ActionGroup

        from .instance import Instance

        config = self.config
        instance_id: str = config['id']
        fwc = self.fw_config
        if fwc is not None:
            instance_id = fwc.instance_config.identifier

        dataset_repo_config = config.get('dataset_repo')
        if dataset_repo_config is not None:
            dataset_repo_spec = DatasetRepoSpec.model_validate(dataset_repo_config)
        else:
            dataset_repo_spec = None

        agc_all = self.config.get('action_groups', [])
        agcs: list[ActionGroup] = []
        for idx, agc in enumerate(agc_all):
            ag = ActionGroup(
                id=agc['id'],
                name=make_trans_string(agc, 'name', required=True),
                color=agc.get('color'),
                order=idx,
            )
            agcs.append(ag)

        target_year = self.config['target_year']

        if fwc is None:
            owner = make_trans_string(self.config, 'owner', required=True)
            name = make_trans_string(self.config, 'name', required=True)
            max_hist_year: int | None = self.config.get('maximum_historical_year')
            min_hist_year: int = self.config['minimum_historical_year']
            reference_year = self.config.get('reference_year')
            if reference_year is None:
                raise ValueError(self, 'Reference year must be given for the instance.')
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
            reference_year = fwc.baseline_year
            if fwc.target_year is not None:
                target_year = fwc.target_year

        self.instance = Instance(
            id=instance_id,
            name=name,
            owner=owner,
            default_language=self.config['default_language'],
            action_groups=agcs,
            config_mtime_hash=self.config_mtime_hash,
            features=self.config.get('features', {}),
            terms=self.config.get('terms', {}),
            result_excels=[InstanceResultExcel.from_yaml_config(r) for r in self.config.get('result_excels', [])],
            yaml_file_path=self.yaml_file_path,
            pages=pages_from_config(self.config.get('pages', [])),
            maximum_historical_year=max_hist_year,
            minimum_historical_year=min_hist_year,
            reference_year=reference_year,
            supported_languages=cast(
                'list[str]',
                self.config.get('supported_languages') or [],
            ),
            theme_identifier=cast('str | None', self.config.get('theme_identifier')),
            # FIXME: The YAML file seems to specify what's supposed to be in InstanceConfig.lead_title (and other
            # attributes), but not under `instance` but under `pages` for a "page" whose `id' is `home`. It's a mess.
            **self._build_instance_args_from_home_page(),
        )

        model_end_year = self.config.get('model_end_year', target_year)
        sample_size = self.config.get('sample_size', 0)
        with start_span(name='create-context', op='function'):
            self.context = Context(
                instance=self.instance,
                dataset_repo_spec=dataset_repo_spec,
                target_year=target_year,
                model_end_year=model_end_year,
                sample_size=sample_size,
            )
        self._finish_init()

    def _finish_init(self) -> None:
        """Run the setup sequence shared by the config-dict and snapshot paths."""
        self.instance.set_context(self.context)
        # Make the fault-tolerance flag available throughout construction (setup_nodes/edges),
        # not just at compute time. See docs/architecture/fault-tolerance.md.
        self.context.tolerate_node_failures = self.tolerate_node_failures

        # Store input and output node configs for each created node, to be used in setup_edges().
        self._input_nodes = {}
        self._output_nodes = {}
        self._subactions = {}
        self._scenario_values = {}
        self._node_visualizations = {}
        self.db_datasets = {}
        self.db_dataset_refs = {}
        self.dataset_payload_store = None
        self.setup_validation_graph()
        self.setup_dimensions()
        self.generate_nodes_from_emission_sectors()
        self.setup_global_parameters()
        self.load_db_datasets()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_impact_overviews()
        self.setup_scenarios()
        self.setup_normalizations()
        self.setup_node_visualizations()
        self.setup_validations()

        for scenario in self.context.scenarios.values():
            if scenario.default:
                break
        else:
            raise Exception('No default scenario defined')
        self.context.activate_scenario(scenario)

    def _init_instance_from_snapshot(self, snapshot: InstanceSnapshot) -> None:
        """Build Instance and Context natively from the typed snapshot (no config dict)."""
        from nodes.context import Context
        from nodes.excel_results import InstanceResultExcel

        from .instance import Instance

        meta = snapshot.metadata
        spec = snapshot.spec
        years = spec.years

        if years.reference is None:
            raise ValueError('Reference year must be given for the instance.')
        if years.min_historical is None:
            raise ValueError('Minimum historical year must be given for the instance.')
        if years.target is None:
            raise ValueError('Target year must be given for the instance.')
        if meta.owner is None:
            raise ValueError('Owner must be given for the instance.')

        def ts(val: I18nString) -> TranslatedString:
            if isinstance(val, TranslatedString):
                return val
            return self.simple_trans_string(str(val))

        agcs = [ag.model_copy(update={'order': idx}) for idx, ag in enumerate(spec.action_groups)]

        # YAML-era convention preserved: the instance lead content comes from
        # the 'home' outcome page; the metadata-level copy serves editors.
        lead_args: dict[str, Any] = {}
        for page in spec.pages:
            if page.id == 'home':
                lead_args['lead_title'] = ts(page.lead_title) if page.lead_title is not None else None
                lead_args['lead_paragraph'] = ts(page.lead_paragraph) if page.lead_paragraph is not None else None
                break

        self.instance = Instance(
            id=meta.identifier,
            name=ts(meta.name),
            owner=ts(meta.owner),
            default_language=meta.primary_language,
            action_groups=agcs,
            config_mtime_hash=None,
            features=spec.features.model_copy(deep=True),
            terms=spec.terms.model_copy(deep=True),
            result_excels=[InstanceResultExcel.from_spec(r) for r in spec.result_excels],
            yaml_file_path=None,
            pages=[page.model_copy(deep=True) for page in spec.pages],
            maximum_historical_year=years.max_historical,
            minimum_historical_year=years.min_historical,
            reference_year=years.reference,
            supported_languages=list(meta.other_languages),
            theme_identifier=spec.theme_identifier,
            **lead_args,
        )

        target_year = years.target
        with start_span(name='create-context', op='function'):
            self.context = Context(
                instance=self.instance,
                dataset_repo_spec=spec.dataset_repo.model_copy(deep=True) if spec.dataset_repo is not None else None,
                target_year=target_year,
                model_end_year=years.model_end or target_year,
                sample_size=spec.sample_size,
            )

        self._stash_snapshot_bindings(snapshot)
        self._finish_init()

    def _stash_snapshot_bindings(self, snapshot: InstanceSnapshot) -> None:
        """Group dataset bindings per node for construction, the same way the config-dict shim grouped them."""
        from collections import defaultdict

        ports_by_node: defaultdict[UUID, list[DatasetPortSnapshot]] = defaultdict(list)
        for port in sorted(snapshot.dataset_bindings, key=lambda p: (p.node, p.dataset_index, str(p.port_id))):
            ports_by_node[port.node].append(port)
        self._snapshot_dataset_ports = dict(ports_by_node)

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
            'lead_title': make_trans_string(page, 'lead_title', default_language=default_language),
            'lead_paragraph': make_trans_string(page, 'lead_paragraph', default_language=default_language),
        }
