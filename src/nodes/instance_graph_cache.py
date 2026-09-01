"""Source-aware construction and caching for :mod:`nodes.instance_graph`."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from django.core.cache import cache

from kausal_common.i18n.pydantic import set_i18n_context

from nodes.instance_graph import INSTANCE_GRAPH_FORMAT_VERSION, InstanceGraph, build_instance_graph
from nodes.instance_serialization import SNAPSHOT_SCHEMA_VERSION, InstanceSnapshot, build_instance_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from paths.context import PathsObjectCache

    from nodes.instance_loader import InstanceYAMLConfig
    from nodes.models import InstanceConfig, PreferredInstanceSource


type InstanceSourceKind = Literal['database-draft', 'database-published', 'yaml']
INSTANCE_GRAPH_CACHE_TIMEOUT = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class ResolvedInstanceSource:
    """Immutable identity of the exact structural snapshot source selected."""

    instance_uuid: str
    kind: InstanceSourceKind
    version: str
    revision_id: int | None = None

    @property
    def cache_key(self) -> str:
        return (
            f'instance-graph:v{INSTANCE_GRAPH_FORMAT_VERSION}:snapshot-v{SNAPSHOT_SCHEMA_VERSION}:'
            f'{self.instance_uuid}:{self.kind}:{self.version}'
        )


@dataclass(frozen=True)
class LoadedInstanceSnapshot:
    """A validated snapshot together with its pre-upgrade schema version."""

    snapshot: InstanceSnapshot
    source_schema_version: int


def resolve_instance_source(
    config: InstanceConfig,
    requested_source: PreferredInstanceSource,
) -> ResolvedInstanceSource:
    """Resolve draft/published/YAML selection before consulting either cache."""
    from nodes.models import PreferredInstanceSource

    if config.config_source != 'database':
        entrypoint = config.get_yaml_config_entrypoint()
        if entrypoint is not None:
            yaml_config = _load_yaml_config(config)
            version = yaml_config.meta.mtime_hash or yaml_config.meta.calculate_mtime_hash()
            return ResolvedInstanceSource(str(config.uuid), 'yaml', version)

    if requested_source == PreferredInstanceSource.PUBLISHED and config.live_revision_id is not None:
        revision = config.live_revision
        content = revision.content if revision is not None else None
        structured = (content or {}).get('model_snapshot', {}).get('structured')
        if structured is not None:
            return ResolvedInstanceSource(
                str(config.uuid),
                'database-published',
                str(config.live_revision_id),
                revision_id=config.live_revision_id,
            )

    return ResolvedInstanceSource(
        str(config.uuid),
        'database-draft',
        config.cache_invalidated_at.isoformat(),
    )


def get_instance_graph(
    config: InstanceConfig,
    requested_source: PreferredInstanceSource,
    *,
    object_cache: PathsObjectCache | None = None,
    refresh: bool = False,
    snapshot_loader: Callable[[], LoadedInstanceSnapshot] | None = None,
    resolved_source: ResolvedInstanceSource | None = None,
) -> InstanceGraph:
    """Return a graph from the request-local L1 or shared Django L2 cache."""
    source = resolved_source or resolve_instance_source(config, requested_source)
    local = object_cache.instance_graphs if object_cache is not None else None
    if not refresh and local is not None and (graph := local.get(source)) is not None:
        return graph

    if not refresh and (serialized := cache.get(source.cache_key)) is not None:
        graph = _load_graph(serialized)
    else:
        loaded = snapshot_loader() if snapshot_loader is not None else load_instance_snapshot(config, source)
        graph = _build_graph(config, source, loaded)
        cache.set(source.cache_key, _dump_graph(graph), timeout=INSTANCE_GRAPH_CACHE_TIMEOUT)

    if local is not None:
        local[source] = graph
    return graph


def _dump_graph(graph: InstanceGraph) -> bytes:
    """Use the established Python-mode validation path for translated fields."""
    data = graph.model_dump(mode='json')
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()


def _load_graph(serialized: bytes | str) -> InstanceGraph:
    return InstanceGraph.model_validate(json.loads(serialized))


def load_instance_snapshot(config: InstanceConfig, source: ResolvedInstanceSource) -> LoadedInstanceSnapshot:
    """Load and validate the exact snapshot identified by ``source``."""
    if source.kind == 'database-draft':
        if config.config_source != 'database':
            test_snapshot = _snapshot_from_preloaded_test_instance(config)
            if test_snapshot is not None:
                return LoadedInstanceSnapshot(test_snapshot, test_snapshot.schema_version)
        snapshot = build_instance_snapshot(config)
        return LoadedInstanceSnapshot(snapshot, snapshot.schema_version)

    if source.kind == 'database-published':
        snapshot, original_version = _published_snapshot(config, source)
    else:
        snapshot = _yaml_snapshot(config)
        original_version = snapshot.schema_version
    return LoadedInstanceSnapshot(snapshot, original_version)


def _build_graph(
    config: InstanceConfig,
    source: ResolvedInstanceSource,
    loaded: LoadedInstanceSnapshot,
) -> InstanceGraph:
    snapshot = loaded.snapshot
    original_version = loaded.source_schema_version

    if source.kind == 'database-draft' or (source.kind == 'database-published' and original_version >= 8):
        return build_instance_graph(snapshot)

    # Old published snapshots and YAML carry identifier references. Resolve
    # those once at this compatibility boundary against the persisted mirror.
    catalog_snapshot = build_instance_snapshot(config)
    return build_instance_graph(
        snapshot,
        legacy_dimensions=tuple(catalog_snapshot.dimensions),
        legacy_datasets=tuple(catalog_snapshot.datasets),
    )


def _published_snapshot(config: InstanceConfig, source: ResolvedInstanceSource) -> tuple[InstanceSnapshot, int]:
    revision = config.live_revision
    if revision is None or revision.pk != source.revision_id:
        raise RuntimeError(f'Published graph source changed while loading instance {config.uuid}')
    structured = ((revision.content or {}).get('model_snapshot') or {}).get('structured')
    if structured is None:
        raise RuntimeError(f'Instance revision {revision.pk} has no structured snapshot')

    source_version = structured.get('schema_version', 1)
    metadata = structured.get('metadata') or {}
    primary_language = metadata.get('primary_language', config.primary_language)
    other_languages = metadata.get('other_languages', config.other_languages or [])
    with set_i18n_context(primary_language, other_languages):
        snapshot = InstanceSnapshot.from_serialized_data(structured)
        if source_version < 6:
            config._complete_legacy_snapshot_content(snapshot)
    return snapshot, source_version


def _load_yaml_config(config: InstanceConfig) -> InstanceYAMLConfig:
    from nodes.instance_loader import InstanceYAMLConfig

    entrypoint = config.get_yaml_config_entrypoint()
    if entrypoint is None:
        raise ValueError(f'No YAML config entrypoint found for instance {config.identifier}')
    return InstanceYAMLConfig.load_for_entrypoint(entrypoint)


def _snapshot_from_preloaded_test_instance(config: InstanceConfig) -> InstanceSnapshot | None:
    """Honor the repository's runtime-only factory fixture without hydrating it again."""
    from nodes.defs.instance_defs import InstanceMetadata
    from nodes.models import _pytest_instances, make_minimal_instance_spec

    instance = _pytest_instances.get(config.identifier)
    if instance is None:
        return None
    metadata = InstanceMetadata.from_model(config)
    metadata.name = instance.name
    metadata.owner = instance.owner
    return InstanceSnapshot(
        metadata=metadata,
        spec=make_minimal_instance_spec(instance),
    )


def _yaml_snapshot(config: InstanceConfig) -> InstanceSnapshot:
    from nodes.instance_parser import parse_instance_snapshot
    from nodes.yaml_port_refs import build_yaml_port_reference_catalog

    yaml_config = _load_yaml_config(config)
    data = yaml_config.data
    assert data is not None
    node_uuids = {node.identifier: node.uuid for node in config.nodes.all().defer('spec')}
    return parse_instance_snapshot(
        data,
        instance_uuid=config.uuid,
        node_uuids=node_uuids,
        port_references=build_yaml_port_reference_catalog(config),
    )
