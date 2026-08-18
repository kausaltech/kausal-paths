"""
Create DB placeholder rows for the external (DVC) datasets a model references.

Two entrypoints, one body: :func:`sync_instance_dataset_placeholders` works
from a runtime ``Context``, :func:`sync_dataset_placeholders_from_snapshot`
from an :class:`~nodes.instance_serialization.InstanceSnapshot` plus the DB.
Both funnel into :class:`_PlaceholderEnv`, which is the whole of what
placeholder creation needs: a language, the model dimensions, the DVC repo
pointer, a way to load a DVC dataset, and the per-dataset metric unit hints
recovered from the node bindings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType

from kausal_common.datasets.models import (
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    Dimension,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.i18n.pydantic import TranslatedString, set_i18n_context

from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

    import dvc_pandas

    from nodes.context import Context
    from nodes.datasets import DVCDataset
    from nodes.defs.instance_defs import DatasetRepoSpec
    from nodes.defs.node_defs import DatasetPortSpec, NodeSpec
    from nodes.dimensions import Dimension as DimensionSpec, DimensionCategory as DimensionCategorySpec
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.models import InstanceConfig
    from nodes.node import Node
    from nodes.units import Unit


def _report(reporter: Callable[[str], None] | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


@dataclass
class _PlaceholderEnv:
    """The config-source-independent inputs of placeholder creation."""

    instance_config: InstanceConfig
    default_language: str
    dimensions: dict[str, DimensionSpec]
    repo_spec: DatasetRepoSpec | None
    load_dvc_dataset: Callable[[str], dvc_pandas.Dataset]
    metric_unit_hints: Callable[[str], dict[str, Unit]]
    """
    Fallback (column -> unit) mapping for a dataset, from the node bindings.

    Called lazily — only for datasets whose DVC metadata carries no units — so
    that a binding with a ``column`` but no unit fails only where the runtime
    path would also have failed.
    """


def _iter_mappable_index_columns(
    index_columns: Sequence[str | object], ds_id: str, reporter: Callable[[str], None] | None
) -> list[str]:
    cols: list[str] = []
    for col in index_columns:
        if isinstance(col, str):
            cols.append(col)
            continue
        _report(
            reporter,
            f"Skipping non-column external dataset index descriptor {col!r} for placeholder '{ds_id}'.",
        )
    return cols


def _collect_placeholder_metric_units_from_node_bindings(
    ctx: Context,
    ds_id: str,
) -> dict[str, Unit]:
    """
    Recover placeholder metrics for legacy emission datasets that do not store metric metadata.

    Older wide DVC datasets can still be referenced by nodes via ``column`` plus a fixed unit
    per column, even when the DVC dataset itself does not carry metric metadata. Until those
    datasets are migrated away, use the runtime node bindings as the fallback source of truth
    for placeholder metric creation.

    Column-less bindings (typical for legacy GPC-style datasets that share a single ``Value``
    column across many heterogeneous rows) are handled by deriving the effective column+unit
    from each referencing node's output metrics. When multiple nodes disagree on the unit for
    the same column — which is expected and not a bug for these wide datasets — we keep the
    first unit seen. ``DatasetMetric.unit`` doesn't drive runtime interpretation for legacy
    DVC datasets; nodes carry their own ``output_metric.unit``.
    """
    from nodes.datasets import DVCDataset

    metric_units: dict[str, Unit] = {}

    for node in ctx.nodes.values():
        for ds in node.input_dataset_instances:
            if not isinstance(ds, DVCDataset):
                continue
            if ds.id != ds_id:
                continue
            bindings = _node_binding_column_units(node, ds)
            _merge_binding_units(metric_units, bindings, ds_id=ds_id, has_column=ds.column is not None)

    return metric_units


def _merge_binding_units(
    metric_units: dict[str, Unit],
    bindings: dict[str, Unit],
    *,
    ds_id: str,
    has_column: bool,
) -> None:
    """Fold one binding's (column -> unit) pairs into the dataset-wide mapping (first unit wins)."""
    for column, unit in bindings.items():
        existing_unit = metric_units.get(column)
        if existing_unit is None:
            metric_units[column] = unit
            continue
        if has_column and unit != existing_unit:
            raise ValueError(f"Conflicting units for placeholder '{ds_id}' column '{column}': {existing_unit} vs {unit}")


def _node_binding_column_units(node: Node, ds: DVCDataset) -> dict[str, Unit]:
    """
    Return (column -> unit) pairs implied by ``node``'s binding to ``ds``.

    - Explicit ``ds.column`` bindings contribute one entry with ``ds.unit``.
    - Column-less bindings fall back to the node's output metrics: each metric
      with a non-null ``column_id`` contributes ``(column_id, metric.unit)``.
    """
    if ds.column is not None:
        if ds.unit is None:
            raise ValueError(f"Missing unit for placeholder '{ds.id}' on node {node.id}")
        return {ds.column: ds.unit}
    result: dict[str, Unit] = {}
    for metric in node.output_metrics.values():
        if metric.column_id is None:
            continue
        column = str(metric.column_id)
        if column not in result:
            result[column] = metric.unit
    return result


def make_external_dataset_ref(ctx: Context, ds_id: str) -> dict[str, str | None] | None:
    return _make_external_dataset_ref(ctx.dataset_repo_spec, ds_id)


def _make_external_dataset_ref(repo: DatasetRepoSpec | None, ds_id: str) -> dict[str, str | None] | None:
    if repo is None:
        return None
    return {
        'repo_url': repo.url,
        'commit': repo.commit,
        'dataset_id': ds_id,
    }


def _create_dataset_schema(
    instance_config: InstanceConfig,
    default_language: str,
    name_i18n: dict[str, str] | None,
) -> DatasetSchema:
    schema = DatasetSchema(
        time_resolution=DatasetSchema.TimeResolution.YEARLY,
    )
    if name_i18n is not None:
        if default_language not in name_i18n:
            # Fallback on whatever we find
            name_i18n[default_language] = next(iter(name_i18n.values()))
        name = TranslatedString(default_language=default_language, **name_i18n)
        name.set_modeltrans_field(schema, 'name', default_language)
    schema.save()
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ContentType.objects.get_for_model(instance_config),
        scope_id=instance_config.pk,
    )
    return schema


def _create_metric(
    col: str,
    unit: Unit | str,
    schema: DatasetSchema,
    default_language: str,
    label_i18n: dict[str, str] | None,
) -> DatasetMetric:
    if isinstance(unit, str):
        unit = unit_registry.parse_units(unit)
    metric = DatasetMetric(schema=schema, name=col, label=col, unit=str(unit))
    if label_i18n is not None:
        if default_language not in label_i18n:
            # Fallback on whatever we find
            label_i18n[default_language] = next(iter(label_i18n.values()))
        label = TranslatedString(default_language=default_language, **label_i18n)
        label.set_modeltrans_field(metric, 'label', default_language)
    metric.save()
    return metric


def _create_dimension_category(
    dimension: Dimension,
    default_language: str,
    spec: DimensionCategorySpec,
) -> DimensionCategory:
    cat = DimensionCategory(dimension=dimension, identifier=spec.id)
    label = spec.label
    assert isinstance(label, TranslatedString)
    label.set_modeltrans_field(cat, 'label', default_language)
    cat.save()
    return cat


def _create_dimension(
    schema: DatasetSchema,
    instance_config: InstanceConfig,
    default_language: str,
    spec: DimensionSpec,
) -> Dimension:
    dimension = Dimension()
    label = spec.label
    assert isinstance(label, TranslatedString)
    label.set_modeltrans_field(dimension, 'name', default_language)
    dimension.save()
    DatasetSchemaDimension.objects.create(schema=schema, dimension=dimension)
    for cat_spec in spec.categories:
        _create_dimension_category(
            dimension=dimension,
            default_language=default_language,
            spec=cat_spec,
        )
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(instance_config),
        scope_id=instance_config.pk,
        identifier=spec.id,
    )
    return dimension


def _get_or_create_dimension(
    schema: DatasetSchema,
    instance_config: InstanceConfig,
    default_language: str,
    spec: DimensionSpec,
) -> Dimension:
    scope = (
        DimensionScope.objects
        .filter(
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=spec.id,
        )
        .select_related('dimension')
        .first()
    )
    if scope is None:
        return _create_dimension(
            schema=schema,
            instance_config=instance_config,
            default_language=default_language,
            spec=spec,
        )
    DatasetSchemaDimension.objects.get_or_create(schema=schema, dimension=scope.dimension)
    return scope.dimension


def _resolve_placeholder_metric_units(
    env: _PlaceholderEnv,
    dvc_ds: dvc_pandas.Dataset,
    ds_id: str,
    reporter: Callable[[str], None] | None,
) -> dict[str, Unit | str]:
    """Prefer the DVC dataset's own units; fall back to the node bindings' unit hints."""
    units: dict[str, Unit | str] = dict(dvc_ds.units or {})
    if units:
        return units
    hints = env.metric_unit_hints(ds_id)
    if hints:
        _report(reporter, f"Falling back to node dataset references for placeholder '{ds_id}' metrics.")
    return dict(hints)


def _sync_dataset_placeholder(  # noqa: C901
    env: _PlaceholderEnv,
    ds_id: str,
    *,
    force: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> tuple[Dataset | None, bool]:
    from django.contrib.contenttypes.models import ContentType

    instance_config = env.instance_config
    default_language = env.default_language

    try:
        dvc_ds = env.load_dvc_dataset(ds_id)
    except Exception as e:
        _report(reporter, f"Error loading DVC dataset '{ds_id}': {e}")
        return None, False

    dvc_metadata = dvc_ds.metadata or {}
    metric_units = _resolve_placeholder_metric_units(env, dvc_ds, ds_id, reporter)
    index_columns = _iter_mappable_index_columns(dvc_ds.index_columns or [], ds_id, reporter)

    existing = (
        Dataset.objects
        .get_queryset()
        .for_instance_config(instance_config)
        .filter(
            identifier=ds_id,
        )
        .select_related('schema')
        .first()
    )
    if existing is not None:
        should_recreate = False
        if existing.is_external_placeholder:
            schema = existing.schema
            has_metrics = schema is not None and schema.metrics.exists()
            should_recreate = force or (metric_units and not has_metrics)
        if existing.is_external_placeholder and should_recreate:
            schema = existing.schema
            if schema is not None and schema.datasets.count() > 1:
                raise RuntimeError(f"Dataset '{existing}' cannot be recreated because its schema is shared with other datasets.")
            if not force:
                _report(reporter, f"Recreating external placeholder dataset '{ds_id}' because its schema has no metrics.")
            existing.delete()
            if schema is not None:
                schema.delete()
        else:
            _report(
                reporter,
                f"Dataset '{existing}' with identifier '{ds_id}' already exists for instance '{instance_config}'; "
                + 'skipping placeholder creation.',
            )
            return existing, False

    schema = _create_dataset_schema(
        instance_config=instance_config,
        default_language=default_language,
        name_i18n=dvc_metadata.get('name'),
    )
    dataset = Dataset.objects.create(
        identifier=ds_id,
        schema=schema,
        external_ref=_make_external_dataset_ref(env.repo_spec, ds_id),
        is_external_placeholder=True,
        scope_content_type=ContentType.objects.get_for_model(instance_config),
        scope_id=instance_config.pk,
    )

    metrics_meta = {
        (m.get('column_id') or m.get('id')): m for m in dvc_metadata.get('metrics') or [] if m.get('column_id') or m.get('id')
    }
    for col, unit in metric_units.items():
        _create_metric(
            col=col,
            unit=unit,
            schema=schema,
            default_language=default_language,
            label_i18n=metrics_meta.get(col, {}).get('label'),
        )

    for col in index_columns:
        if col not in env.dimensions:
            _report(
                reporter,
                f"Skipping external dataset index column '{col}' for placeholder '{ds_id}' "
                + 'because it does not map to a model dimension.',
            )
            continue
        _get_or_create_dimension(
            schema=schema,
            instance_config=instance_config,
            default_language=default_language,
            spec=env.dimensions[col],
        )

    _report(reporter, f"Created external placeholder dataset '{ds_id}'")
    return dataset, True


def _sync_dataset_placeholders(
    env: _PlaceholderEnv,
    ds_ids: Sequence[str],
    *,
    force: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> list[str]:
    created_dataset_ids: list[str] = []
    for ds_id in ds_ids:
        _dataset, created = _sync_dataset_placeholder(
            env,
            ds_id,
            force=force,
            reporter=reporter,
        )
        if created:
            created_dataset_ids.append(ds_id)
    return created_dataset_ids


# ---------------------------------------------------------------------------
# Runtime-context entrypoints
# ---------------------------------------------------------------------------


def _env_from_context(instance_config: InstanceConfig, ctx: Context) -> _PlaceholderEnv:
    return _PlaceholderEnv(
        instance_config=instance_config,
        default_language=ctx.instance.default_language,
        dimensions=dict(ctx.dimensions),
        repo_spec=ctx.dataset_repo_spec,
        load_dvc_dataset=ctx.load_dvc_dataset,
        metric_unit_hints=lambda ds_id: _collect_placeholder_metric_units_from_node_bindings(ctx, ds_id),
    )


def sync_dataset_placeholder(
    instance_config: InstanceConfig,
    ctx: Context,
    ds_id: str,
    *,
    force: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> tuple[Dataset | None, bool]:
    return _sync_dataset_placeholder(
        _env_from_context(instance_config, ctx),
        ds_id,
        force=force,
        reporter=reporter,
    )


def sync_instance_dataset_placeholders(
    instance_config: InstanceConfig,
    ctx: Context,
    *,
    force: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> list[str]:
    return _sync_dataset_placeholders(
        _env_from_context(instance_config, ctx),
        sorted(ctx.get_all_dvc_dataset_ids()),
        force=force,
        reporter=reporter,
    )


# ---------------------------------------------------------------------------
# Snapshot entrypoint
# ---------------------------------------------------------------------------


def _build_dataset_repo(repo_spec: DatasetRepoSpec) -> dvc_pandas.Repository:
    """Mirror ``Context.dataset_repo``: same credentials, same target commit."""
    import dvc_pandas

    creds = dvc_pandas.RepositoryCredentials(
        git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
        git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
        git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
        git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
    )
    repo = dvc_pandas.Repository(
        repo_url=repo_spec.url,
        dvc_remote=repo_spec.dvc_remote,
        repo_credentials=creds,
    )
    repo.set_target_commit(repo_spec.commit)
    return repo


@dataclass
class _SnapshotDvcLoader:
    """
    Load DVC datasets from a repo spec, without a runtime ``Context``.

    Reproduces ``Context.load_dvc_dataset``: in-memory memoization, and a
    single bulk ``load_datasets`` of everything the model needs the first time
    a dataset turns out not to be in the DVC cache (individual DVC operations
    are slow enough that loading all at once wins).
    """

    repo_spec: DatasetRepoSpec | None
    dataset_ids: Sequence[str]
    _loaded: dict[str, dvc_pandas.Dataset] = field(default_factory=dict, init=False)
    _repo: dvc_pandas.Repository | None = field(default=None, init=False)
    _loaded_all: bool = field(default=False, init=False)

    @property
    def repo(self) -> dvc_pandas.Repository:
        if self.repo_spec is None:
            raise RuntimeError('Dataset repository not set')
        if self._repo is None:
            self._repo = _build_dataset_repo(self.repo_spec)
        return self._repo

    def load(self, ds_id: str) -> dvc_pandas.Dataset:
        ds = self._loaded.get(ds_id)
        if ds is not None:
            return ds
        if self.repo_spec is None:
            raise RuntimeError('Dataset repository not set')
        if not self.repo.has_dataset(ds_id):
            raise Exception('Dataset %s not found in DVC repo' % ds_id)
        if not self._loaded_all and not self.repo.is_dataset_cached(ds_id):
            self.repo.load_datasets(list(self.dataset_ids))
            self._loaded_all = True
        ds = self.repo.load_dataset(ds_id)
        self._loaded[ds_id] = ds
        return ds


def _binding_columns(spec: DatasetPortSpec, node_spec: NodeSpec | None) -> list[str]:
    """
    Return the metric columns a dataset binding contributes.

    Mirror of the runtime pair ``DVCDataset.column`` / node output metrics (see
    ``_node_binding_column_units``): an explicit ``column`` is the only column,
    otherwise the columns are the owning node's output-port ``column_id``s.
    """
    if spec.column is not None:
        return [spec.column]
    if node_spec is None:
        return []
    columns: list[str] = []
    seen: set[str] = set()
    for port in node_spec.output_ports:
        if port.column_id is None:
            continue
        column = str(port.column_id)
        if column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def _output_port_unit(node_spec: NodeSpec | None, column: str) -> Unit | None:
    if node_spec is None:
        return None
    for port in node_spec.output_ports:
        if port.column_id is not None and str(port.column_id) == column:
            return port.unit
    return None


@dataclass
class _SnapshotBinding:
    """One node-to-dataset binding, collapsed from the snapshot's per-column port rows."""

    node: UUID
    dataset: str
    spec: DatasetPortSpec


def _snapshot_bindings(snapshot: InstanceSnapshot) -> list[_SnapshotBinding]:
    """
    Collapse the snapshot's per-column port rows back into bindings, in node order.

    ``dataset_ports`` holds one row per (binding, column); a binding is
    identified by (node, dataset_index), so the rows collapse back into the
    bindings the runtime iterates over.
    """
    bindings: dict[tuple[UUID, int], _SnapshotBinding] = {}
    for port in snapshot.dataset_bindings:
        key = (port.node, port.dataset_index)
        if key in bindings:
            continue
        bindings[key] = _SnapshotBinding(node=port.node, dataset=port.dataset, spec=port.spec)
    return list(bindings.values())


class _SnapshotUnitHints:
    """
    Snapshot equivalent of ``_collect_placeholder_metric_units_from_node_bindings``.

    The bindings are grouped by dataset up front; the units are resolved per
    dataset on call, so a binding whose unit is missing raises only for the
    datasets that actually need the fallback.
    """

    def __init__(self, snapshot: InstanceSnapshot) -> None:
        self._node_specs: dict[UUID, NodeSpec | None] = {n.uuid: n.spec for n in snapshot.nodes}
        self._by_dataset: dict[str, list[_SnapshotBinding]] = {}
        for binding in _snapshot_bindings(snapshot):
            self._by_dataset.setdefault(binding.dataset, []).append(binding)

    def __call__(self, ds_id: str) -> dict[str, Unit]:
        metric_units: dict[str, Unit] = {}
        for binding in self._by_dataset.get(ds_id, []):
            node_spec = self._node_specs.get(binding.node)
            spec = binding.spec
            bindings: dict[str, Unit] = {}
            for column in _binding_columns(spec, node_spec):
                if spec.column is not None:
                    if spec.unit is None:
                        raise ValueError(f"Missing unit for placeholder '{ds_id}' on node {binding.node}")
                    bindings[column] = spec.unit
                    continue
                unit = _output_port_unit(node_spec, column)
                if unit is not None and column not in bindings:
                    bindings[column] = unit
            _merge_binding_units(metric_units, bindings, ds_id=ds_id, has_column=spec.column is not None)
        return metric_units


def _dimensions_from_snapshot(snapshot: InstanceSnapshot) -> dict[str, DimensionSpec]:
    """Rebuild the runtime ``Dimension`` objects the snapshot's spec carries as raw dicts."""
    from nodes.dimensions import Dimension as DimensionSpecModel

    dims: dict[str, DimensionSpec] = {}
    for dim_config in snapshot.spec.dimensions:
        dim = DimensionSpecModel.model_validate(dim_config)
        dims[str(dim.id)] = dim
    return dims


def _dvc_dataset_ids_from_snapshot(instance_config: InstanceConfig, snapshot: InstanceSnapshot) -> set[str]:
    """
    Return the dataset ids that load over DVC, i.e. ``Context.get_all_dvc_dataset_ids``.

    Every dataset binding loads from DVC unless the loader would resolve it to
    a ``DBDataset`` instead: that needs ``use_datasets_from_db`` and a real
    (non-placeholder, identifier-bearing) DB dataset for the instance — the
    same queryset as ``InstanceLoader.load_db_datasets``. Existing placeholders
    are deliberately not in that set; they take the DVC path so their schema
    gets refreshed.

    A few runtime bindings stay ``DVCDataset`` even when a DB dataset exists —
    tagged ones (``city_data``, ``observation_dataset``,
    ``framework_measure_data``), every binding of a framework-backed instance,
    and the emission-sector nodes (built before ``load_db_datasets`` runs) — so
    this excludes slightly more than the runtime does. That makes no difference
    to the outcome: an id with a real DB dataset for the instance also matches
    the "already exists" lookup in :func:`_sync_dataset_placeholder`, which
    skips creation. Only the DVC load (and its report line) is avoided.
    """
    ds_ids = {port.dataset for port in snapshot.dataset_bindings}
    if not ds_ids or not snapshot.spec.features.use_datasets_from_db:
        return ds_ids
    db_ds_ids = set(
        Dataset.objects
        .get_queryset()
        .for_instance_config(instance_config)
        .filter(is_external_placeholder=False, identifier__in=ds_ids)
        .values_list('identifier', flat=True)
    )
    return ds_ids - db_ds_ids


def sync_dataset_placeholders_from_snapshot(
    instance_config: InstanceConfig,
    snapshot: InstanceSnapshot,
    reporter: Callable[[str], None] | None = None,
) -> list[str]:
    """
    Create external-dataset placeholders for a parsed instance snapshot.

    Snapshot-and-DB equivalent of :func:`sync_instance_dataset_placeholders`;
    needs no runtime ``Context``. Returns the ids of the placeholders created.
    """
    metadata = snapshot.metadata
    with set_i18n_context(metadata.primary_language, list(metadata.other_languages)):
        ds_ids = sorted(_dvc_dataset_ids_from_snapshot(instance_config, snapshot))
        loader = _SnapshotDvcLoader(repo_spec=snapshot.spec.dataset_repo, dataset_ids=ds_ids)
        env = _PlaceholderEnv(
            instance_config=instance_config,
            default_language=metadata.primary_language,
            dimensions=_dimensions_from_snapshot(snapshot),
            repo_spec=snapshot.spec.dataset_repo,
            load_dvc_dataset=loader.load,
            metric_unit_hints=_SnapshotUnitHints(snapshot),
        )
        return _sync_dataset_placeholders(env, ds_ids, reporter=reporter)
