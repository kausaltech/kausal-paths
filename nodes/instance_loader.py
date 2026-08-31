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

from pydantic import BaseModel, Field, field_validator

import platformdirs
from loguru import logger
from rich import print
from ruamel.yaml import YAML as RuamelYAML  # noqa: N811
from sentry_sdk import start_span

from kausal_common.i18n.pydantic import TranslatedString, get_i18n_context, gettext_lazy as _, set_i18n_context

from nodes.actions.action import ActionNode
from nodes.exceptions import NodeError
from params.discover import discover_global_parameters

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from uuid import UUID

    from ruamel.yaml import CommentedMap

    from kausal_common.datasets.models import Dataset as DBDatasetModel
    from kausal_common.i18n.pydantic import I18nString

    from nodes.context import Context
    from nodes.datasets import Dataset
    from nodes.defs.node_defs import InputDatasetDef, NodeSpec
    from nodes.defs.transform_def import EdgeTransformOp
    from nodes.edges import Edge
    from nodes.explanations import NodeExplanationSystem
    from nodes.instance import Instance
    from nodes.instance_graph import InstanceGraph
    from nodes.instance_serialization import DatasetPortSnapshot, EdgeSnapshot, InstanceSnapshot, NodeSnapshot
    from nodes.models import InstanceConfig
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
        is_editable: bool | None = None,
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
            is_editable=is_editable,
        )

    @staticmethod
    def _merge_include_dataset_config(
        existing: list[CommentedMap],
        included: list[CommentedMap],
        dataset_replacements: list[dict[str, str]],
    ) -> None:
        """Merge module dataset metadata, remapping ids exactly like node bindings."""
        dataset_map = {replacement['from']: replacement['to'] for replacement in dataset_replacements}
        by_id = {dataset['id']: dataset for dataset in existing}
        for dataset in included:
            source_id = dataset['id']
            target_id = dataset_map.get(source_id, source_id)
            dataset['id'] = target_id
            current = by_id.get(target_id)
            if current is None:
                existing.append(dataset)
                by_id[target_id] = dataset
                continue
            # Instance-level declarations take precedence, while omitted
            # fields still inherit reusable metadata from the module.
            for key, value in dataset.items():
                if key not in current:
                    current[key] = value

    def _merge_config(
        self,
        existing: list[CommentedMap],
        newconf: list[CommentedMap],
        entity_type: str,
        apply_group: str | None = None,
        config_path: Path | None = None,
        allow_override: bool = False,
        is_editable: bool | None = None,
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
            if is_editable is not None:
                nc['is_editable'] = is_editable
            if config_path is not None:
                nc['config_location'] = ConfigLocation(file_path=str(config_path), line=nc.lc.line + 1, column=nc.lc.col)
            existing.append(nc)

    def _init_group(self, objs: list[CommentedMap]) -> None:
        for d in objs:
            d['config_location'] = ConfigLocation(file_path=str(self.meta.entrypoint.path), line=d.lc.line + 1, column=d.lc.col)

    @staticmethod
    def _set_merged_collections(
        data: dict[str, Any],
        *,
        nodes: list[CommentedMap],
        actions: list[CommentedMap],
        dimensions: list[CommentedMap],
        datasets: list[CommentedMap],
    ) -> None:
        data.update(nodes=nodes, actions=actions, dimensions=dimensions, datasets=datasets)

    @staticmethod
    def _get_include_nodes_editable(include: CommentedMap) -> bool | None:
        value = include.get('nodes_editable')
        if 'nodes_editable' in include and not isinstance(value, bool):
            msg = 'Include option nodes_editable must be a boolean'
            raise TypeError(msg)
        return value

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
        datasets = data.get('datasets', [])

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
            nodes_editable = self._get_include_nodes_editable(iconf)
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
                is_editable=nodes_editable,
            )
            self._merge_include_dataset_config(
                datasets,
                idata.get('datasets', []),
                dataset_replacements,
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
                is_editable=nodes_editable,
            )

        # Make sure that assignment works even if they are originally empty.
        self._set_merged_collections(data, nodes=nodes, actions=actions, dimensions=dimensions, datasets=datasets)

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
    snapshot: InstanceSnapshot
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
    # Groups are derived from native binding fields: a row whose pipeline
    # selects a metric is its own group; the column-less rows of one
    # (node, dataset) form one whole-frame group.
    _snapshot_dataset_groups: dict[UUID, list[tuple[InputDatasetDef, list[DatasetPortSnapshot]]]]
    _binding_group_index: dict[int, int]
    _instance_graph: InstanceGraph
    _runtime_datasets: dict[tuple[UUID, int], Dataset]

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

        uses_generic_dataset = issubclass(node_class, GenericNode) and not issubclass(node_class, AdditiveNode)

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
        class_interpolate = node_class.interpolates_input_datasets_by_default and not uses_generic_dataset
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
                # Snapshot-path entries arrive as typed defs, YAML-path entries as
                # dicts; either way an authored `interpolate` beats the class default.
                authored_interpolate = 'interpolate' in (ds.model_fields_set if isinstance(ds, InputDatasetDef) else ds)
                ds_def = InputDatasetDef.model_validate(ds)
                if ds_interpolate:
                    ds_def.interpolate = True
                elif class_interpolate and not authored_interpolate:
                    ds_def.interpolate = True

            # The class declares whether its datasets take framework measure
            # overlays; the tag is the per-binding opt-in for other classes.
            use_framework_ds = 'framework_measure_data' in ds_def.tags or (
                node_class.uses_framework_measure_data and self.context.framework_config_data is not None
            )
            use_obs_ds = 'observation_dataset' in ds_def.tags
            use_city_ds = 'city_data' in ds_def.tags
            ds_obj: Dataset | None = None
            if use_obs_ds:
                from frameworks.datasets import ObservationDataset

                ds_obj = ObservationDataset.from_def(ds_def, self.context)
            elif use_city_ds:
                from frameworks.datasets import FrameworkMeasureDVCDataset2

                # Prefer a DB-stored dataset when one exists for this instance.
                # FrameworkMeasureDVCDataset2 handles both cases: when db_dataset_obj is
                # provided it loads from DB, otherwise falls through to DVC. Either way,
                # The transformation pipeline and post-transform hook handle the uuid
                # dimension and any framework measure datapoint overrides correctly.
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
                if uses_generic_dataset:
                    # GenericDataset applies its unconditional interpolate/extend
                    # fills at execution time; framework and DB-backed replacements
                    # (ruled out above) deliberately don't get them.
                    ds_obj = GenericDataset.from_def(ds_def, self.context)
                else:
                    ds_obj = DVCDataset.from_def(ds_def, self.context)
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

        # Same dict shape as the YAML path used, but sourced from the typed
        # spec; copied so the shared snapshot is never mutated.
        dim_configs: list[dict[str, Any]] = [dict(dc) for dc in self.snapshot.spec.dimensions]
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
                    ag = next((ag for ag in self.instance.action_groups if ag.uuid == tc.group), None)
                    if ag is None:
                        self._init_failure(node, "Action group with UUID '%s' not found" % tc.group)
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
            # For ports without an authored identifier, the class's canonical
            # metric keys are recovered by column_id (same as the dict path).
            class_metrics_def: dict[str, NodeMetric] | None = getattr(node_class, 'output_metrics', None)
            col_to_class_key = {m.column_id: k for k, m in class_metrics_def.items()} if class_metrics_def else {}
            for port in spec.output_ports:
                column = str(port.column_id) if port.column_id is not None else None
                if column is None:
                    raise Exception('Node %s: multi-metric output port without column_id' % identifier)
                if port.quantity is None:
                    raise Exception('Node %s: output metric %s has no quantity' % (identifier, column))
                assert port.unit is not None
                key = port.identifier or col_to_class_key.get(column, column)
                metrics[key] = NodeMetric(unit=port.unit, quantity=port.quantity, id=key, column_id=column, label=port.label)

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

    def make_node_from_snapshot(self, node_class: type[Node], n: NodeSnapshot) -> Node:  # noqa: C901, PLR0912
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
        dataset_groups = self._snapshot_dataset_groups.get(n.uuid)
        if dataset_groups:
            ds_fragment['input_datasets'] = [ds_def for ds_def, _rows in dataset_groups]
        if extra.input_dataset_processors:
            ds_fragment['input_dataset_processors'] = list(extra.input_dataset_processors)
        if extra.historical_values:
            ds_fragment['historical_values'] = extra.historical_values
        if extra.forecast_values:
            ds_fragment['forecast_values'] = extra.forecast_values
        if extra.tags:
            ds_fragment['tags'] = list(extra.tags)
        datasets = self._make_node_datasets(ds_fragment, node_class, unit)
        if dataset_groups:
            if len(dataset_groups) != len(datasets):
                raise ValueError(
                    f'Node {identifier}: {len(dataset_groups)} dataset binding groups produced {len(datasets)} runtime datasets'
                )
            for group_index, dataset in enumerate(datasets):
                self._runtime_datasets[(n.uuid, group_index)] = dataset

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
            spec=spec,
        )
        if node.id in self._input_nodes or node.id in self._output_nodes:
            raise Exception('Node %s is already configured' % node.id)
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
        self._setup_nodes_from_snapshot(actions=False)

    @wrap_with_span('setup-actions', 'function')
    def setup_actions(self):
        self._setup_nodes_from_snapshot(actions=True)

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

        The target port's declared dimensions (plus legacy
        ``FlattenTransformation`` rows preserved in pre-step-2 snapshots) become
        bare exclude+flatten declarations, filters resolve categories and
        groups, and assigns pin one category.
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
        # Key order matters to the exporter (port dimensions follow declaration
        # order); the authored *op* order rides on ``Edge.source_transforms``.
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
        from nodes.defs.transform_def import modernized_transformations

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
            # Modernizing edge-storable ops yields edge-storable ops (legacy
            # kinds rewrite in place, flatten declarations drop out).
            source_transforms=cast('list[EdgeTransformOp]', modernized_transformations(list(transforms))),
        )

    def _setup_edges(self) -> None:
        self._setup_edges_from_snapshot()

    def _setup_runtime_inputs(self) -> None:  # noqa: C901, PLR0912
        """Attach graph bindings to runtime sources without mutating the cached graph models."""
        from collections import defaultdict

        from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef
        from nodes.instance_serialization import DatasetPortSnapshot, EdgeSnapshot
        from nodes.runtime_input import RuntimeInputBinding

        runtime_by_uuid = {
            meta.id: self.context.get_node(meta.identifier) for meta in self._instance_graph.nodes if meta.identifier is not None
        }
        bindings_by_target: defaultdict[UUID, list[RuntimeInputBinding]] = defaultdict(list)
        snapshot_bindings = list(self.snapshot.bindings_with_positions())
        if len(snapshot_bindings) != len(self._instance_graph.bindings):
            raise ValueError('InstanceGraph and source snapshot disagree on input binding count')

        for definition, (snapshot_binding, _position) in zip(self._instance_graph.bindings, snapshot_bindings, strict=True):
            try:
                target_meta = definition.target_node
            except ValueError:
                if self.tolerate_node_failures:
                    continue
                raise
            target = runtime_by_uuid.get(target_meta.id)
            if target is None or not target.input_port_declarations:
                # Unmigrated classes keep using the legacy edge/dataset views,
                # whose persisted port UUIDs need not match the exported spec.
                continue
            try:
                target_port = definition.target_port
            except ValueError:
                if self.tolerate_node_failures:
                    continue
                raise
            role = target_meta.role_for_input_port(target_port)
            if role is None:
                continue

            if isinstance(definition, EdgeBindingDef):
                if not isinstance(snapshot_binding, EdgeSnapshot):
                    raise TypeError(f'Binding {definition.id}: graph/snapshot kind mismatch')
                source = runtime_by_uuid[definition.source_node.id]
            elif isinstance(definition, DatasetBindingDef):
                if not isinstance(snapshot_binding, DatasetPortSnapshot):
                    raise TypeError(f'Binding {definition.id}: graph/snapshot kind mismatch')
                source = self._runtime_datasets[(snapshot_binding.node, self._binding_group_index[id(snapshot_binding)])]
            else:
                raise TypeError(f'Unsupported binding definition {type(definition).__name__}')

            bindings_by_target[target_meta.id].append(
                RuntimeInputBinding.from_graph_binding(
                    definition,
                    port_role=role,
                    source=source,
                    target=target,
                )
            )

        for node_id, node in runtime_by_uuid.items():
            bindings = bindings_by_target.get(node_id, [])
            fixed_role = node.legacy_fixed_dataset_input_role
            if fixed_role is not None:
                from nodes.datasets import FixedDataset

                bindings.extend(
                    RuntimeInputBinding.from_legacy_fixed_dataset(dataset, target=node, port_role=fixed_role)
                    for dataset in node.input_dataset_instances
                    if isinstance(dataset, FixedDataset)
                )
            node.bind_runtime_inputs(bindings, node_meta=self._instance_graph.node_by_id[node_id])

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

    def _snapshot_scenarios(self) -> list[Scenario]:
        """Runtime scenarios from the typed spec (param values re-cleaned like the dict path)."""
        from nodes.scenario import Scenario, ScenarioKind

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

    def setup_scenarios(self):
        from nodes.scenario import CustomScenario

        default_scenario = None

        scenarios = self._snapshot_scenarios()
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

    def setup_global_parameters(self):
        global_params = discover_global_parameters()

        context = self.context
        for spec_param in self.snapshot.spec.params:
            if spec_param.local_id not in global_params:
                raise Exception('Unknown global parameter: %s' % spec_param.local_id)
            param = spec_param.model_copy(deep=True)
            param.set_context(context)
            context.add_global_parameter(param)

    def setup_impact_overviews(self):
        from nodes.actions.action import ImpactOverview

        seen: set[str] = set()
        for overview_spec in self.snapshot.spec.impact_overviews:
            spec = overview_spec.model_copy(deep=True)
            assert spec.id is not None
            if spec.id in seen:
                raise ValueError(f"Duplicate impact overview id '{spec.id}'. Set an explicit 'id' field to disambiguate.")
            seen.add(spec.id)
            self.context.impact_overviews.append(ImpactOverview(spec, self.context))

    def setup_normalizations(self):
        from paths.refs import ValidationContext

        from nodes.defs.instance_defs import NormalizationSpec
        from nodes.normalization import Normalization

        # Re-validate against this context so node refs resolve here.
        spec_configs: list[dict[str, Any]] = [n.model_dump() for n in self.snapshot.spec.normalizations]
        for spec_config in spec_configs:
            normalization = Normalization(
                NormalizationSpec.model_validate(spec_config, context=ValidationContext(context=self.context)),
                self.context,
            )
            self.context.add_normalization(normalization.normalizer_node.id, normalization)

    def setup_node_explanations(self):
        """Install a lazy builder for the explanation system; nothing consumes it during loading."""
        from nodes.explanations import build_node_explanation_system

        snapshot = self.snapshot

        def build_from_snapshot(context: Context) -> NodeExplanationSystem:
            from nodes.instance_from_db import snapshot_nodes_to_config_dicts

            nodes_list, actions_list = snapshot_nodes_to_config_dicts(snapshot)
            return build_node_explanation_system(context, [*nodes_list, *actions_list])

        self.context._nes_factory = build_from_snapshot

    @classmethod
    def from_snapshot(
        cls,
        snapshot: InstanceSnapshot,
        tolerate_node_failures: bool = False,
        *,
        published: bool = False,
        instance_config: InstanceConfig | None = None,
    ) -> Self:
        """Build the runtime natively from an ``InstanceSnapshot`` (typed specs, no YAML dicts)."""
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
            instance_config=instance_config,
        )

    @classmethod
    def from_yaml(
        cls,
        filename: Path,
        tolerate_node_failures: bool = False,
        instance_config: InstanceConfig | None = None,
        snapshot_transform: Callable[[InstanceSnapshot], InstanceSnapshot] | None = None,
    ) -> Self:
        """
        Build the runtime from YAML through parse -> InstanceSnapshot -> native build.

        ``snapshot_transform`` lets a caller overlay the parsed snapshot before
        construction — the framework path uses it to apply ``FrameworkConfig``
        identity and year boundaries without the loader knowing about
        frameworks.
        """
        import uuid as uuid_mod

        from nodes.instance_parser import parse_instance_snapshot

        yaml_fn = filename.resolve()
        yaml_conf = InstanceYAMLConfig.load_for_entrypoint(yaml_fn)
        data = yaml_conf.data
        assert data is not None
        if instance_config is not None:
            instance_uuid = instance_config.uuid
        else:
            # The runtime never persists parse-invented UUIDs, so a deterministic
            # namespace from the instance identifier is sufficient; no DB lookup.
            instance_uuid = uuid_mod.uuid3(uuid_mod.NAMESPACE_URL, f'kausal-paths:instance:{data["id"]}')
        snapshot = parse_instance_snapshot(data, instance_uuid=instance_uuid)
        if snapshot_transform is not None:
            # The transform may rewrite I18n fields, which validate against the
            # active i18n context.
            with set_i18n_context(snapshot.metadata.primary_language, list(snapshot.metadata.other_languages)):
                snapshot = snapshot_transform(snapshot)
        return cls(
            snapshot=snapshot,
            yaml_file_path=yaml_fn,
            config_mtime_hash=yaml_conf.meta.mtime_hash,
            tolerate_node_failures=tolerate_node_failures,
            instance_config=instance_config,
        )

    def __init__(
        self,
        yaml_file_path: Path | None = None,
        config_mtime_hash: str | None = None,
        tolerate_node_failures: bool = False,
        dataset_payload_refs: list[Any] | None = None,
        instance_config: InstanceConfig | None = None,
        *,
        snapshot: InstanceSnapshot,
    ):
        from .units import add_unit_translations

        add_unit_translations()
        self.tolerate_node_failures = tolerate_node_failures
        self.supplied_dataset_payload_refs = dataset_payload_refs
        self.instance_config = instance_config
        self.config_mtime_hash = config_mtime_hash
        self._node_classes = {}
        self.snapshot = snapshot
        self.yaml_file_path = yaml_file_path.absolute() if yaml_file_path else None
        self.default_language = snapshot.metadata.primary_language
        self.other_languages = list(snapshot.metadata.other_languages)
        self.logger = logger.bind(instance=snapshot.metadata.identifier)
        with set_i18n_context(self.default_language, self.other_languages):
            self._init_instance_from_snapshot(snapshot)

    def setup_node_visualizations(self):
        for node_id, viz_config in self._node_visualizations.items():
            node = self.context.get_node(node_id)
            self._make_node_visualizations(node, viz_config)

    def load_db_datasets(self):
        from kausal_common.datasets.models import Dataset as DBDatasetModel

        if self.supplied_dataset_payload_refs is not None:
            from nodes.datasets import RevisionDatasetPayloadStore

            self.db_datasets = {}
            self.db_dataset_refs = {ref.identifier: ref for ref in self.supplied_dataset_payload_refs}
            self.dataset_payload_store = RevisionDatasetPayloadStore(self.supplied_dataset_payload_refs)
            return

        ic = self.instance.config
        if ic is None:
            # Standalone YAML tooling has no InstanceConfig owner to pass in.
            # Keep that compatibility lookup explicit at the loader boundary;
            # normal InstanceConfig construction binds the owner up front.
            from nodes.models import InstanceConfig

            try:
                ic = InstanceConfig.objects.get(identifier=self.instance.id)
            except InstanceConfig.DoesNotExist:
                self.db_datasets = {}
                return
            self.instance.config = ic
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

    def _finish_init(self) -> None:
        """Run the setup sequence for the natively built instance."""
        self.instance.set_context(self.context)
        # Make the fault-tolerance flag available throughout construction (setup_nodes/edges),
        # not just at compute time. See docs/architecture/fault-tolerance.md.
        self.context.tolerate_node_failures = self.tolerate_node_failures

        # Duplicate-node bookkeeping, plus per-node state consumed by later setup steps.
        self._input_nodes = {}
        self._output_nodes = {}
        self._subactions = {}
        self._scenario_values = {}
        self._node_visualizations = {}
        self.db_datasets = {}
        self.db_dataset_refs = {}
        self.dataset_payload_store = None
        self.setup_node_explanations()
        self.setup_dimensions()
        self.setup_global_parameters()
        self.load_db_datasets()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self._setup_runtime_inputs()
        self.setup_impact_overviews()
        self.setup_scenarios()
        self.setup_normalizations()
        self.setup_node_visualizations()

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
            config_mtime_hash=self.config_mtime_hash,
            features=spec.features.model_copy(deep=True),
            terms=spec.terms.model_copy(deep=True),
            result_excels=[InstanceResultExcel.from_spec(r) for r in spec.result_excels],
            yaml_file_path=self.yaml_file_path,
            pages=[page.model_copy(deep=True) for page in spec.pages],
            maximum_historical_year=years.max_historical,
            minimum_historical_year=years.min_historical,
            reference_year=years.reference,
            supported_languages=list(meta.other_languages),
            theme_identifier=spec.theme_identifier,
            config=self.instance_config,
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
        from uuid import NAMESPACE_URL, uuid5

        from nodes.defs.graph import DatasetMeta, DatasetMetricMeta
        from nodes.instance_graph import build_instance_graph

        graph_snapshot = snapshot
        is_yaml = getattr(self, 'yaml_file_path', None) is not None
        if is_yaml or not snapshot.datasets:
            catalog = list(snapshot.datasets)
            if not catalog and self.instance_config is not None and not is_yaml:
                from nodes.instance_serialization import build_instance_snapshot

                catalog.extend(build_instance_snapshot(self.instance_config).datasets)

            specs_by_node = {node.uuid: node.spec for node in snapshot.nodes if node.spec is not None}
            datasets_by_id = {dataset.id: dataset for dataset in catalog}
            datasets_by_identifier = {dataset.identifier: dataset for dataset in catalog if dataset.identifier is not None}
            for binding in snapshot.dataset_bindings:
                dataset = datasets_by_id.get(binding.dataset_uuid) if binding.dataset_uuid is not None else None
                if dataset is None:
                    dataset = datasets_by_identifier.get(binding.dataset)

                if dataset is None:
                    dataset_uuid = binding.dataset_uuid or uuid5(
                        NAMESPACE_URL,
                        f'kausal-paths:runtime-dataset:{snapshot.metadata.uuid}:{binding.dataset}',
                    )
                    dataset = DatasetMeta(
                        id=dataset_uuid,
                        identifier=binding.dataset,
                        schema_id=uuid5(NAMESPACE_URL, f'kausal-paths:runtime-dataset-schema:{dataset_uuid}'),
                        is_external_placeholder=True,
                    )
                    catalog.append(dataset)
                    datasets_by_id[dataset.id] = dataset
                    datasets_by_identifier[binding.dataset] = dataset

                metric = dataset.metric_by_id.get(binding.metric_uuid) if binding.metric_uuid is not None else None
                if metric is None:
                    metric = next((item for item in dataset.metrics if item.identifier == binding.metric), None)
                if metric is not None:
                    continue

                metric_uuid = binding.metric_uuid or uuid5(
                    NAMESPACE_URL,
                    f'kausal-paths:runtime-dataset-metric:{dataset.id}:{binding.metric}',
                )
                spec = specs_by_node[binding.node]
                unit = spec.input_port_by_id[binding.port_id].unit
                updated_dataset = dataset.model_copy(
                    update={
                        'metrics': (
                            *dataset.metrics,
                            DatasetMetricMeta(
                                id=metric_uuid,
                                identifier=binding.metric,
                                unit=str(unit) if unit is not None else '',
                            ),
                        )
                    }
                )
                catalog[catalog.index(dataset)] = updated_dataset
                datasets_by_id[updated_dataset.id] = updated_dataset
                if updated_dataset.identifier is not None:
                    datasets_by_identifier[updated_dataset.identifier] = updated_dataset

            graph_snapshot = snapshot.model_copy(update={'datasets': catalog})

        self._instance_graph = build_instance_graph(graph_snapshot)
        self._runtime_datasets = {}
        self._binding_group_index = {}
        self._snapshot_dataset_groups = self._group_dataset_bindings(snapshot)

    def _group_dataset_bindings(
        self, snapshot: InstanceSnapshot
    ) -> dict[UUID, list[tuple[InputDatasetDef, list[DatasetPortSnapshot]]]]:
        """
        Group dataset bindings per node from native fields only.

        A row whose pipeline selects a metric is a single-metric binding and
        forms its own group; the column-less rows of one (node, dataset) are the
        per-metric fan-out of a single whole-frame binding and collapse back
        into one group. Group order per node follows (input-port declaration
        order, per-port position), which is the authored order the fan-out was
        created in.
        """
        from collections import defaultdict

        from nodes.instance_serialization import DatasetPortSnapshot

        rows_by_node: defaultdict[UUID, list[tuple[DatasetPortSnapshot, int]]] = defaultdict(list)
        for item, position in snapshot.bindings_with_positions():
            if isinstance(item, DatasetPortSnapshot):
                rows_by_node[item.node].append((item, position))
        port_specs_by_node = {node.uuid: node.spec for node in snapshot.nodes if node.spec is not None}

        groups_by_node: dict[UUID, list[tuple[InputDatasetDef, list[DatasetPortSnapshot]]]] = {}
        for node_uuid, node_rows in rows_by_node.items():
            node_spec = port_specs_by_node.get(node_uuid)
            port_order = {port.id: index for index, port in enumerate(node_spec.input_ports)} if node_spec is not None else {}
            node_rows.sort(key=lambda entry: (port_order.get(entry[0].port_id, len(port_order)), entry[1], str(entry[0].port_id)))
            grouped: list[tuple[str, list[DatasetPortSnapshot]]] = []
            # dataset id -> (open group index, metrics seen in it). One whole-frame
            # binding fans out to one row per schema metric, so a metric repeating
            # within the open group can only mean a second binding of the same
            # dataset: close the group and start the next one.
            whole_frame_group: dict[str, tuple[int, set[str]]] = {}
            for row, _position in node_rows:
                if any(op.kind == 'select_metric' for op in row.spec.transformations):
                    grouped.append((row.dataset, [row]))
                    continue
                open_group = whole_frame_group.get(row.dataset)
                if open_group is None or row.metric in open_group[1]:
                    whole_frame_group[row.dataset] = (len(grouped), {row.metric})
                    grouped.append((row.dataset, [row]))
                else:
                    grouped[open_group[0]][1].append(row)
                    open_group[1].add(row.metric)
            node_groups: list[tuple[InputDatasetDef, list[DatasetPortSnapshot]]] = []
            for group_index, (dataset_id, group_rows) in enumerate(grouped):
                # Runtime-input resolution keys on object identity: the rows here
                # are the same binding objects bindings_with_positions() yields.
                for row in group_rows:
                    self._binding_group_index[id(row)] = group_index
                node_groups.append((group_rows[0].spec.to_input_dataset(id=dataset_id), group_rows))
            groups_by_node[node_uuid] = node_groups
        return groups_by_node
