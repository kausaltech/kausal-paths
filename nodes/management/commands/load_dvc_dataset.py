from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

import polars as pl
from rich import print

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    DatasetSourceReference,
    DataSource,
    Dimension,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.i18n.pydantic import TranslatedString

from common import polars as ppl
from nodes.constants import (
    FORECAST_COLUMN,
    RESERVED_ROW_COLUMNS,
    SOURCE_TARGET_DATA_POINT,
    SOURCE_TARGET_DATASET,
    YEAR_COLUMN,
)
from nodes.dataset_materialization import refresh_dataset_materialization
from nodes.dataset_placeholders import make_external_dataset_ref, sync_dataset_placeholder
from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from pathlib import Path

    from nodes.context import Context
    from nodes.defs.instance_defs import DatasetRepoSpec
    from nodes.dimensions import Dimension as DimensionSpec, DimensionCategory as DimensionCategorySpec
    from nodes.units import Unit

# Must match notebooks/upload_new_dataset.py's SOURCE_NAME_SEPARATOR: a 'Source' cell may
# join multiple citation names when a value was derived from more than one.
SOURCE_NAME_SEPARATOR = '; '

# A 'Comment' cell may likewise carry several distinct notes about one data point -- a
# source-cell reference, the inventory's own wording, an adopted-error note -- each of
# which should become its own DataPointComment.
#
# Deliberately NOT '; ', unlike SOURCE_NAME_SEPARATOR. Source names are short identifiers;
# comments are prose, and prose contains semicolons: 162 of the 12,899 comment cells
# already in data/ contain '; ' inside a single sentence, and splitting on it would
# fragment them. ' ;; ' cannot occur by accident in ordinary text.
COMMENT_SEPARATOR = ' ;; '

# RESERVED_ROW_COLUMNS comes from nodes.constants, which is also what `upload_new_dataset`
# writes by and what `DVCDataset` drops on load. It used to be copied here and there with a
# "must match" comment; one definition is what actually makes them match.


def source_target(fields: dict[str, str | None]) -> str:
    """
    Read what one metadata['sources'] entry attaches to.

    A missing key means data_point: that is what every .dvc file written before
    dataset-level sources existed says, and it is what those files meant.
    """
    return fields.get('target') or SOURCE_TARGET_DATA_POINT


@dataclass(frozen=True)
class ResolvedSource:
    """A DataSource row plus what its references attach to."""

    source: DataSource
    target: str


def dataset_level_source_names(sources_meta: list[dict[str, str | None]] | None) -> list[str]:
    """Names of the sources that attach to the dataset as a whole. Any number of them may."""
    return sorted(
        str(fields['name'])
        for fields in sources_meta or []
        if fields.get('name') and source_target(fields) == SOURCE_TARGET_DATASET
    )


@dataclass
class RepoProvenance:
    """Which DVC commit this run will read from, and which one the other config source names."""

    used_source: str
    """'yaml' or 'db' -- where the effective pin came from."""

    yaml_commit: str | None
    db_commit: str | None
    spec: DatasetRepoSpec | None

    @property
    def sources_disagree(self) -> bool:
        return self.yaml_commit is not None and self.db_commit is not None and self.yaml_commit != self.db_commit


def _yaml_repo_spec(ic: InstanceConfig) -> DatasetRepoSpec | None:
    """Read ``dataset_repo`` straight from the instance's YAML entrypoint, without loading the model."""
    from nodes.defs.instance_defs import DatasetRepoSpec
    from nodes.instance_loader import InstanceYAMLConfig

    path: Path | None = ic.get_yaml_config_entrypoint()
    if path is None:
        return None
    yaml_conf = InstanceYAMLConfig.load_for_entrypoint(path)
    repo = (yaml_conf.data or {}).get('dataset_repo')
    if not repo:
        return None
    return DatasetRepoSpec.model_validate(repo)


def resolve_repo_provenance(ic: InstanceConfig, ctx: Context, repo_from: str) -> RepoProvenance:
    """
    Decide which dataset-repo pin to import from, and record the one we are not using.

    ``ctx`` already carries the pin belonging to the instance's declared ``config_source``
    (YAML file or DB spec). The point of resolving both is that importing data from a
    different commit than the model expects is silent: it surfaces later, and far away,
    as a missing metric. ``repo_from='auto'`` keeps the declared source -- which side is
    authoritative changes as the model editor takes over -- but a disagreement is always
    reported.
    """
    ctx_spec: DatasetRepoSpec | None = ctx.dataset_repo_spec
    declared_is_yaml = ic.config_source == 'yaml'

    yaml_spec = ctx_spec if declared_is_yaml else _yaml_repo_spec(ic)
    db_spec = ctx_spec if not declared_is_yaml else (ic.spec.dataset_repo if ic.spec is not None else None)

    if repo_from == 'yaml':
        spec, used = yaml_spec, 'yaml'
    elif repo_from == 'db':
        spec, used = db_spec, 'db'
    else:
        spec, used = ctx_spec, ('yaml' if declared_is_yaml else 'db')

    return RepoProvenance(
        used_source=used,
        yaml_commit=yaml_spec.commit if yaml_spec is not None else None,
        db_commit=db_spec.commit if db_spec is not None else None,
        spec=spec,
    )


def apply_repo_provenance(ctx: Context, provenance: RepoProvenance) -> None:
    """Point ``ctx`` at the resolved pin, dropping the cached repo if it was already built."""
    if provenance.spec is None or provenance.spec is ctx.dataset_repo_spec:
        return
    ctx.dataset_repo_spec = provenance.spec
    ctx.__dict__.pop('dataset_repo', None)


@dataclass
class DatasetPlan:
    """What a sync would do to one dataset, computed before anything is written."""

    ds_id: str
    incoming_commit: str | None
    existing_pk: int | None = None
    existing_uuid: str | None = None
    current_commit: str | None = None
    current_data_points: int = 0
    incoming_data_points: int = 0
    kept_metrics: list[str] = field(default_factory=list)
    added_metrics: list[str] = field(default_factory=list)
    dropped_metrics: list[tuple[str, int]] = field(default_factory=list)
    """(metric name, number of dataset ports binding it) for columns no longer in the DVC data."""

    current_dataset_sources: list[str] = field(default_factory=list)
    """Names of the DataSources currently attached to the dataset row itself."""

    incoming_dataset_sources: list[str] = field(default_factory=list)
    """Names the incoming metadata attaches to the dataset row. The import replaces one set with the other."""

    is_placeholder: bool = False
    schema_shared_with: int = 0

    @property
    def is_new(self) -> bool:
        return self.existing_pk is None

    @property
    def blockers(self) -> list[str]:
        """
        Reasons this sync must not proceed.

        A metric that the model still binds cannot lose its column: the node would
        keep a port pointing at data that no longer arrives. Better to say so here,
        naming the bindings, than to let it surface later as a missing metric during
        ``sync_instance_to_db`` -- or not at all, as silently empty input.

        "Binds" means a ``NodeInputPortBinding`` row; each holds a PROTECTed reference,
        so an under-count turns a clean refusal into a ``ProtectedError`` halfway
        through the sync.
        """
        return [
            f'metric {name!r} would be dropped but {n} model binding(s) still bind it' for name, n in self.dropped_metrics if n
        ]


def build_dataset_plan(
    ds_id: str,
    dataset: Dataset | None,
    incoming_metric_cols: list[str],
    incoming_data_points: int,
    incoming_commit: str | None,
    incoming_dataset_sources: list[str] | None = None,
) -> DatasetPlan:
    """Diff the DB state of one dataset against the DVC data about to be imported."""
    from nodes.models import NodeInputPortBinding

    plan = DatasetPlan(
        ds_id=ds_id,
        incoming_commit=incoming_commit,
        incoming_data_points=incoming_data_points,
        added_metrics=list(incoming_metric_cols),
        incoming_dataset_sources=list(incoming_dataset_sources or []),
    )
    if dataset is None:
        return plan

    plan.existing_pk = dataset.pk
    plan.existing_uuid = str(dataset.uuid)
    plan.current_commit = (dataset.external_ref or {}).get('commit')
    plan.current_data_points = dataset.data_points.count()
    plan.is_placeholder = dataset.is_external_placeholder
    plan.current_dataset_sources = sorted(
        ref.data_source.name for ref in DatasetSourceReference.objects.filter(dataset=dataset).select_related('data_source')
    )
    schema = dataset.schema
    if schema is None:
        return plan

    plan.schema_shared_with = schema.datasets.count()
    existing = {m.name: m for m in schema.metrics.all() if m.name}
    incoming = set(incoming_metric_cols)
    plan.kept_metrics = sorted(name for name in existing if name in incoming)
    plan.added_metrics = sorted(name for name in incoming if name not in existing)
    # Every binding holds a PROTECTed reference to its metric, and every one has to be
    # counted: missing one means promising the sync can proceed and then dying on the
    # delete, with the transaction rolled back and the operator none the wiser about
    # what to clear.
    plan.dropped_metrics = sorted(
        (
            name,
            NodeInputPortBinding.objects.filter(metric=metric).count(),
        )
        for name, metric in existing.items()
        if name not in incoming
    )
    return plan


def print_dataset_source_plan(plan: DatasetPlan) -> None:
    """
    Render the dataset-level source lines of one dataset's plan.

    They are replaced wholesale on every import, so the removals need saying: they are
    otherwise invisible until someone notices the provenance is gone.
    """
    current = set(plan.current_dataset_sources)
    incoming = set(plan.incoming_dataset_sources)
    for name in sorted(current & incoming):
        print(f'  source keep  {name} [dim](dataset-level)[/dim]')
    for name in sorted(incoming - current):
        print(f'  source [green]add[/green]   {name} [dim](dataset-level)[/dim]')
    for name in sorted(current - incoming):
        print(f'  source [yellow]drop[/yellow]  {name} [dim](dataset-level)[/dim]')


def print_dataset_plan(plan: DatasetPlan) -> None:
    """Render one dataset's plan; the same output precedes an apply run and stands alone under --plan."""
    print(f'[bold]{plan.ds_id}[/bold]')
    if plan.is_new:
        print('  row          [green]new[/green]')
    else:
        kind = ' (external placeholder)' if plan.is_placeholder else ''
        print(f'  row          pk={plan.existing_pk} uuid={plan.existing_uuid}{kind}')
    commit_from = plan.current_commit or 'unrecorded'
    if plan.is_new:
        print(f'  commit       {plan.incoming_commit}')
    elif commit_from == plan.incoming_commit:
        print(f'  commit       {commit_from} [dim](unchanged)[/dim]')
    else:
        print(f'  commit       {commit_from} [yellow]->[/yellow] {plan.incoming_commit}')
    print(f'  data points  {plan.current_data_points} -> {plan.incoming_data_points}')
    if plan.kept_metrics:
        print(f'  metrics kept {", ".join(plan.kept_metrics)}')
    if plan.added_metrics:
        print(f'  metrics [green]add[/green]  {", ".join(plan.added_metrics)}')
    for name, port_count in plan.dropped_metrics:
        suffix = f' [red](bound by {port_count} dataset port(s))[/red]' if port_count else ''
        print(f'  metrics [yellow]drop[/yellow] {name}{suffix}')
    print_dataset_source_plan(plan)
    for problem in plan.blockers:
        print(f'  [red]blocker[/red]      {problem}')


def _translated_metadata(values: dict[str, str], default_language: str) -> TranslatedString:
    values = values.copy()
    if default_language not in values:
        values[default_language] = values.get('en') or next(iter(values.values()))
    return TranslatedString(default_language=default_language, **values)


def _label_from_identifier(identifier: str) -> str:
    return identifier.replace('_', ' ').replace('-', ' ').title()


def _parse_column_dimension_mappings(values: list[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for value in values:
        if '=' not in value:
            raise ValueError("--create-dimension-from-column values must use the syntax 'column=dimension_id'")
        column, dimension_id = value.split('=', maxsplit=1)
        column = column.strip()
        dimension_id = dimension_id.strip()
        if not column or not dimension_id:
            raise ValueError("--create-dimension-from-column values must use the syntax 'column=dimension_id'")
        mappings[column] = dimension_id
    return mappings


@lru_cache
def get_dimension(instance_config: InstanceConfig, identifier: str) -> Dimension:
    scope = DimensionScope.objects.get(
        scope_content_type=ContentType.objects.get_for_model(instance_config),
        scope_id=instance_config.pk,
        identifier=identifier,
    )
    return scope.dimension


@lru_cache
def get_dimension_category(instance_config: InstanceConfig, dimension_identifier: str, identifier: str) -> DimensionCategory:
    dimension = get_dimension(instance_config, dimension_identifier)
    return DimensionCategory.objects.get(dimension=dimension, identifier=identifier)


class Command(BaseCommand):
    help = 'Create a dataset in DB based on a DVC dataset'

    # Map dimension identifiers to a dict mapping dimension category identifiers to DimensionCategory instances
    dimension_categories: dict[str, dict[str, DimensionCategory]] = {}

    def add_arguments(self, parser):
        parser.add_argument('instance', metavar='INSTANCE_ID', type=str, nargs=1)
        parser.add_argument('datasets', metavar='DATASET_ID', type=str, nargs='*')
        parser.add_argument('--all', action='store_true', help='Sync all datasets')
        parser.add_argument(
            '--metadata-only',
            action='store_true',
            help='Create placeholder dataset, schema and metric objects without importing datapoints',
        )
        parser.add_argument(
            '--ignore-prefix',
            action='append',
            help='Ignore datasets with IDs starting with the specified prefix. Can be used multiple times.',
        )
        parser.add_argument(
            '--create-dimension-from-column',
            action='append',
            default=[],
            metavar='COLUMN=DIMENSION_ID',
            help=(
                'Create or update a scoped dimension from a DVC index column, then import that column under '
                'DIMENSION_ID. Can be used multiple times.'
            ),
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Replace the data of datasets that already exist in the DB (the row itself is kept)',
        )
        parser.add_argument(
            '--plan',
            action='store_true',
            help='Diagnose only: report what exists and what this run would change, without writing anything',
        )
        parser.add_argument(
            '--repo-from',
            choices=['auto', 'yaml', 'db'],
            default='auto',
            help=(
                "Which dataset-repo pin to import from: the instance's declared config source (auto, the default), "
                'the YAML file, or the DB spec.'
            ),
        )
        parser.add_argument(
            '--recreate',
            action='store_true',
            help='With --force, delete and recreate the dataset row instead of refreshing it in place (mints a new UUID)',
        )

    @transaction.atomic
    def sync_dataset(  # noqa: C901, PLR0912, PLR0915
        self,
        instance_config: InstanceConfig,
        ctx: Context,
        ds_id: str,
        force: bool = False,
        metadata_only: bool = False,
        create_dimensions_from_columns: dict[str, str] | None = None,
        plan_only: bool = False,
        recreate: bool = False,
    ):
        create_dimensions_from_columns = create_dimensions_from_columns or {}
        if metadata_only:
            if plan_only:
                print(f'{ds_id}: would sync placeholder metadata only')
                return
            sync_dataset_placeholder(instance_config, ctx, ds_id, force=force, reporter=print)
            return

        dvc_ds = ctx.load_dvc_dataset(ds_id)
        df = ppl.from_dvc_dataset(dvc_ds)
        self.rename_value_columns(df)
        df_metadata = df.get_meta()
        dvc_metadata = dvc_ds.metadata or {}

        identifier = ds_id
        scope_content_type = ContentType.objects.get_for_model(instance_config)
        get_kwargs = dict(
            scope_content_type=scope_content_type,
            scope_id=instance_config.pk,
            identifier=identifier,
        )
        dataset: Dataset | None = None
        schema: DatasetSchema | None = None
        datasets = list(
            Dataset.objects
            .get_queryset()
            .for_instance_config(instance_config)
            .filter(identifier=identifier)
            .select_related('schema')[:2]
        )
        if len(datasets) > 1:
            raise RuntimeError(f"Multiple datasets with identifier '{identifier}' exist for instance '{instance_config}'.")
        if datasets:
            dataset = datasets[0]

        plan = build_dataset_plan(
            ds_id=ds_id,
            dataset=dataset,
            incoming_metric_cols=list(df_metadata.metric_cols),
            incoming_data_points=sum(df[col].drop_nulls().len() for col in df_metadata.metric_cols),
            incoming_commit=(make_external_dataset_ref(ctx, ds_id) or {}).get('commit'),
            incoming_dataset_sources=dataset_level_source_names(dvc_metadata.get('sources')),
        )
        print_dataset_plan(plan)
        if plan_only:
            return
        if plan.blockers:
            raise CommandError(
                f'{ds_id}: refusing to sync -- ' + '; '.join(plan.blockers) + '. Update the model bindings first, '
                'or keep the column in the DVC data.'
            )

        if dataset is not None:
            if dataset.is_external_placeholder:
                print(f"Dataset '{dataset}' with identifier '{identifier}' is an external placeholder. Replacing.")
                dataset.is_external_placeholder = False
                dataset.scope_content_type = scope_content_type
                dataset.scope_id = instance_config.pk
                dataset.external_ref = make_external_dataset_ref(ctx, ds_id)
                dataset.save(update_fields=['is_external_placeholder', 'scope_content_type', 'scope_id', 'external_ref'])
                schema = dataset.schema
                assert schema is not None
            elif not force:
                print(f"Dataset '{dataset}' with identifier '{identifier}' exists for instance '{instance_config}'. Aborting.")
                print('Pass --force to replace its data.')
                return
            elif recreate:
                schema = dataset.schema
                assert schema is not None
                if schema.datasets.count() > 1:
                    print('Dataset exists already, but schema is linked to other datasets as well. Aborting.')
                    return
                print(f"Deleting existing dataset '{dataset}'")
                dataset.delete()
                print(f"Deleting dataset schema '{schema}'")
                schema.delete()
                # Both objects are now unsaved (the collector nulls their pks), so drop the
                # references: the code below recreates them only when they are None.
                dataset = None
                schema = None
            else:
                schema = self.refresh_dataset_in_place(
                    dataset=dataset,
                    ctx=ctx,
                    ds_id=ds_id,
                    plan=plan,
                    dvc_metadata=dvc_metadata,
                    default_language=ctx.instance.default_language,
                )

        if schema is None:
            schema = self.create_dataset_schema(
                instance_config=instance_config,
                default_language=ctx.instance.default_language,
                name_i18n=dvc_metadata['name'],
                description_i18n=dvc_metadata.get('description'),
            )
        if dataset is None:
            dataset = Dataset.objects.create(**get_kwargs, schema=schema, external_ref=make_external_dataset_ref(ctx, ds_id))
            print(f"Created dataset '{dataset}'")

        # Match DB metric columns (DVC units keys) to meta: column_id is the physical column name; id is optional slug.
        metrics_meta = {
            (m.get('column_id') or m.get('id')): m for m in dvc_metadata.get('metrics') or [] if m.get('column_id') or m.get('id')
        }

        metrics = {m.name: m for m in schema.metrics.all() if m.name}
        # Map metric identifiers (column names) to Metric instances
        metrics.update({
            col: self.create_metric(
                col=col,
                unit=df_metadata.units[col],
                schema=schema,
                default_language=ctx.instance.default_language,
                label_i18n=metrics_meta.get(col, {}).get('label'),
            )
            for col in df_metadata.metric_cols
            if col not in metrics
        })
        # A reused metric keeps its row, so a unit change in the DVC data would otherwise be
        # silently ignored -- the values would land under the old unit.
        for col in df_metadata.metric_cols:
            metric = metrics[col]
            incoming_unit = str(df_metadata.units[col])
            if metric.pk is not None and metric.unit != incoming_unit:
                print(f"Metric '{col}': unit {metric.unit} -> {incoming_unit}")
                metric.unit = incoming_unit
                metric.save(update_fields=['unit'])

        df, column_dimensions = self.sync_dimensions(
            schema=schema,
            instance_config=instance_config,
            ctx=ctx,
            df=df,
            create_dimensions_from_columns=create_dimensions_from_columns,
        )

        for col, dt in df.schema.items():
            if dt == pl.Categorical:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
        self.create_data_points(
            instance_config,
            df,
            dataset,
            metrics,
            column_dimensions=column_dimensions,
            sources_meta=dvc_metadata.get('sources'),
        )
        refresh_dataset_materialization(dataset)

    def create_dataset_schema(
        self,
        instance_config: InstanceConfig,
        default_language: str,
        name_i18n: dict[str, str] | None,
        description_i18n: dict[str, str] | None = None,
    ) -> DatasetSchema:
        schema = DatasetSchema(
            time_resolution=DatasetSchema.TimeResolution.YEARLY,  # TODO: allow other granularities
            # unit=?,  # What the hell is this for in DatasetSchema?
        )
        if name_i18n is not None:
            name = _translated_metadata(name_i18n, default_language)
            name.set_modeltrans_field(schema, 'name', default_language)
        if description_i18n:
            schema.description = description_i18n.get(default_language) or next(iter(description_i18n.values()))
        schema.save()
        print(f"Created dataset schema '{schema}'")
        print(f"Setting scope of schema '{schema}' to '{instance_config}'")
        DatasetSchemaScope.objects.create(
            schema=schema,
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
        )
        return schema

    def refresh_dataset_in_place(
        self,
        dataset: Dataset,
        ctx: Context,
        ds_id: str,
        plan: DatasetPlan,
        dvc_metadata: dict,
        default_language: str,
    ) -> DatasetSchema:
        """
        Replace a dataset's contents while keeping the row itself.

        Deleting and recreating the row is the older strategy, and it is destructive in
        ways that are easy to miss: ``DatasetPort``, ``NodeDataset`` and
        ``InstanceRevisionDatasetPin`` all reference the row under ``PROTECT``, and the
        new row gets a fresh UUID, which orphans the dataset references stored in
        published instance revisions. Keeping the pk and UUID leaves every one of those
        intact; only the data underneath changes.
        """
        schema = dataset.schema
        assert schema is not None

        deleted, _ = dataset.data_points.all().delete()
        print(f'Deleted {deleted} row(s) of existing data')

        for name, port_count in plan.dropped_metrics:
            assert not port_count, 'bound metrics are refused before we get here'
            if plan.schema_shared_with > 1:
                print(f"Keeping stale metric '{name}': the schema is shared with other datasets")
                continue
            DatasetMetric.objects.filter(schema=schema, name=name).delete()
            print(f"Deleted stale metric '{name}' (no longer in the DVC data)")

        # Restamp provenance, so the row records the commit its data actually came from.
        dataset.external_ref = make_external_dataset_ref(ctx, ds_id)
        dataset.save(update_fields=['external_ref'])

        name_i18n = dvc_metadata.get('name')
        if name_i18n is not None:
            _translated_metadata(name_i18n, default_language).set_modeltrans_field(schema, 'name', default_language)
        description_i18n = dvc_metadata.get('description')
        if description_i18n:
            schema.description = description_i18n.get(default_language) or next(iter(description_i18n.values()))
        schema.save()
        return schema

    def get_or_create_data_sources(
        self, instance_config: InstanceConfig, sources_meta: list[dict[str, str | None]] | None
    ) -> dict[str, ResolvedSource]:
        """
        Resolve a dvc_metadata['sources'] list into DataSource rows, keyed by name.

        Identity is the name within the instance's scope, so a source already known to the
        instance keeps its row -- and with it every reference, from this dataset and from
        every other one that cites it. Its descriptive fields are refreshed from the
        incoming metadata, because for an imported dataset the registry in DVC is the
        source of truth and an edition that never updates is worse than none. Anything that
        must survive as a distinct source (a superseded edition still cited elsewhere)
        needs a distinct name.
        """
        if not sources_meta:
            return {}
        scope_content_type = ContentType.objects.get_for_model(instance_config)
        result: dict[str, ResolvedSource] = {}
        for fields in sources_meta:
            name = fields['name']
            assert name is not None
            incoming = {
                'authority': fields.get('authority'),
                'url': fields.get('url'),
                'description': fields.get('description'),
                'edition': fields.get('edition'),
            }
            source, created = DataSource.objects.get_or_create(
                scope_content_type=scope_content_type,
                scope_id=instance_config.pk,
                name=name,
                defaults=incoming,
            )
            if not created:
                changed = [f for f, value in incoming.items() if getattr(source, f) != value]
                if changed:
                    for f in changed:
                        setattr(source, f, incoming[f])
                    # `last_modified_at` is auto_now, and auto_now only fires for a field the
                    # update_fields list names -- so an unlisted one silently keeps the old
                    # timestamp, and the row would claim it had not changed.
                    source.save(update_fields=[*changed, 'last_modified_at'])
                    print(f"Source '{name}': updated {', '.join(sorted(changed))} from the DVC metadata")
            result[name] = ResolvedSource(source=source, target=source_target(fields))
        return result

    def link_data_point_sources(self, data_point: DataPoint, source_cell: str, data_sources: dict[str, ResolvedSource]) -> None:
        """Link data_point to each DataSource named in source_cell (SOURCE_NAME_SEPARATOR-joined for >1 citation)."""
        for name in source_cell.split(SOURCE_NAME_SEPARATOR):
            resolved = data_sources.get(name)
            if resolved is None:
                print(f"Source '{name}' not found in dvc_metadata['sources']; skipping.")
            elif resolved.target != SOURCE_TARGET_DATA_POINT:
                # Refused rather than honoured: the same source would then be attached both
                # to the dataset and to its rows, which is a duplicated claim, not a stronger
                # one. `build_sources_metadata` rejects this combination at upload time, so
                # reaching here means hand-edited metadata.
                print(f"Source '{name}' is declared dataset-level but cited by a row; not linking it to the data point.")
            else:
                DatasetSourceReference.objects.create(data_point=data_point, data_source=resolved.source)

    def sync_dataset_source_references(self, dataset: Dataset, data_sources: dict[str, ResolvedSource]) -> None:
        """
        Replace the dataset's dataset-level source references with the ones the metadata declares.

        Replace, not add: unlike the per-point references -- which CASCADE away with the data
        points that `refresh_dataset_in_place` deletes -- a reference hanging off the Dataset
        row survives a re-import, so appending would grow a duplicate on every ``--force``,
        and a source dropped from the registry would linger forever.
        """
        deleted, _ = DatasetSourceReference.objects.filter(dataset=dataset).delete()
        names = sorted(name for name, resolved in data_sources.items() if resolved.target == SOURCE_TARGET_DATASET)
        for name in names:
            DatasetSourceReference.objects.create(dataset=dataset, data_source=data_sources[name].source)
        if names or deleted:
            print(f'Dataset-level sources: {", ".join(names) if names else "(none)"}')

    def create_data_point_comments(self, data_point: DataPoint, comment_cell: str) -> None:
        """Create one DataPointComment per note in comment_cell (COMMENT_SEPARATOR-joined for >1)."""
        for part in comment_cell.split(COMMENT_SEPARATOR):
            text = part.strip()
            if text:
                DataPointComment.objects.create(data_point=data_point, text=text)

    def create_data_points(
        self,
        instance_config: InstanceConfig,
        df: ppl.PathsDataFrame,
        dataset: Dataset,
        metrics: dict[str, DatasetMetric],
        *,
        column_dimensions: dict[str, str] | None = None,
        sources_meta: list[dict[str, str | None]] | None = None,
    ):
        """
        Create the dataset's data points, and with them everything the metadata says about provenance.

        The dataset-level source references are (re)built here too, rather than beside the
        caller: both levels are resolved from the one ``sources_meta`` list, and doing them
        together is what keeps a source from being attached twice under two different names
        for the same row set.
        """
        column_dimensions = column_dimensions or {}
        data_sources = self.get_or_create_data_sources(instance_config, sources_meta)
        self.sync_dataset_source_references(dataset, data_sources)
        meta = df.get_meta()
        table = JSONDataset.serialize_df(df)
        # We might not need to serialize `df` to create the data points, but I didn't check what the manipulations
        # of `df` above and the serialization do, so I'll take the serialization like the old version of
        # this management command did.
        # 'Source'/'Comment' (or, for plain_csv_wide, 'Description') are reserved per-row columns:
        # not dimensions, read back here into DataSource/DataPointComment links.
        #
        # Only one comment column is read. A file carrying both 'Comment' and 'Description'
        # would silently lose one, so that is rejected rather than resolved by column order;
        # put every note in 'Comment', COMMENT_SEPARATOR-joined.
        source_col = next((c for c in df.columns if c.lower() == 'source'), None)
        comment_cols = [c for c in df.columns if c.lower() in ('comment', 'description')]
        if len(comment_cols) > 1:
            raise CommandError(
                f'Dataset carries more than one comment column ({", ".join(comment_cols)}). '
                f"Use 'Comment' only, joining several notes with '{COMMENT_SEPARATOR}'."
            )
        comment_col = comment_cols[0] if comment_cols else None
        num_created = 0
        for row in table['data']:
            year_val = row['Year']
            year = date(year=year_val, month=1, day=1)
            for metric_identifier, metric in metrics.items():
                value = row[metric_identifier]
                # A valueless cell is created as a DataPoint with a null value rather than
                # skipped. `DataPoint.value` is nullable, GraphQL types it `float | None`, and
                # DataAvailabilityNode tests `is_not_null()` — so an empty cell reads as
                # "no data" while still existing as a row the city can see, comment on and
                # fill in. Skipping it lost the cell, its dimension categories, its source
                # link and its comment, which is why BISKO template datasets had to ship
                # zeros: a pre-filled 0 is indistinguishable from a municipality-confirmed 0,
                # and the certifier's Pruefschritt 1.4 tests exactly that.
                data_point = DataPoint.objects.create(
                    dataset=dataset,
                    date=year,
                    metric=metric,
                    value=value,
                )
                num_created += 1
                for column in meta.dim_ids:
                    dimension_identifier = column_dimensions.get(column, column)
                    dim_cat_identifier = row[column]
                    if dim_cat_identifier:
                        try:
                            cat = get_dimension_category(instance_config, dimension_identifier, dim_cat_identifier)
                        except DimensionCategory.DoesNotExist:
                            print(f"Dimension category '{dim_cat_identifier}' not found. Did you run --update-instance?")
                            raise
                        data_point.dimension_categories.add(cat)
                if source_col and row.get(source_col):
                    self.link_data_point_sources(data_point, row[source_col], data_sources)
                if comment_col and row.get(comment_col):
                    self.create_data_point_comments(data_point, row[comment_col])
        print(f'Created {num_created} data points')

    def rename_value_columns(self, df: ppl.PathsDataFrame):
        meta = df.get_meta()
        value_columns = (c for c in df.columns if c not in meta.metric_cols and c not in meta.dim_ids)
        for col in value_columns:
            if col.lower() == FORECAST_COLUMN.lower():
                if col != FORECAST_COLUMN:
                    df = df.rename({col: FORECAST_COLUMN})
            elif col.lower() == YEAR_COLUMN.lower():
                if col != YEAR_COLUMN:
                    df = df.rename({col: YEAR_COLUMN})
            elif col.lower() in RESERVED_ROW_COLUMNS:
                pass
            else:
                print(df)
                raise Exception(f'Unknown column {col}')

    def get_category_identifiers_from_column(self, df: ppl.PathsDataFrame, column: str) -> list[str]:
        return sorted(str(value) for value in df[column].drop_nulls().unique().to_list())

    def sync_dimensions(
        self,
        schema: DatasetSchema,
        instance_config: InstanceConfig,
        ctx: Context,
        df: ppl.PathsDataFrame,
        create_dimensions_from_columns: dict[str, str],
    ) -> tuple[ppl.PathsDataFrame, dict[str, str]]:
        df_metadata = df.get_meta()
        dim_ids = set(df_metadata.dim_ids)
        for column in create_dimensions_from_columns:
            if column not in dim_ids:
                raise ValueError(
                    f"Column '{column}' is not a dimension/index column in the DVC dataset. "
                    + f'Available dimension columns: {", ".join(sorted(dim_ids))}'
                )

        column_dimensions: dict[str, str] = {}
        for col in df_metadata.dim_ids:
            if dimension_identifier := create_dimensions_from_columns.get(col):
                self.remove_schema_dimensions_for_column(
                    schema=schema,
                    instance_config=instance_config,
                    column_name=col,
                    keep_dimension_identifier=dimension_identifier,
                )
                self.get_or_create_dimension_from_column(
                    schema=schema,
                    instance_config=instance_config,
                    column_name=col,
                    dimension_identifier=dimension_identifier,
                    category_identifiers=self.get_category_identifiers_from_column(df, col),
                )
                column_dimensions[col] = dimension_identifier
                continue

            self.get_or_create_dimension(
                schema=schema,
                instance_config=instance_config,
                default_language=ctx.instance.default_language,
                spec=ctx.dimensions[col],
            )
            # Does the following ever do anything to the column `col` in `df` other than converting strings to
            # `pl.Categorical`?
            new_col = ctx.dimensions[col].series_to_ids_pl(df[col], allow_null=True)
            # Let's throw in an assert and find out.
            # assert new_col.equals(df[col])
            df = df.with_columns(new_col)

        return df, column_dimensions

    def create_metric(
        self, col: str, unit: Unit, schema: DatasetSchema, default_language: str, label_i18n: dict[str, str] | None
    ) -> DatasetMetric:
        metric = DatasetMetric(schema=schema, name=col, label=col, unit=str(unit))
        if label_i18n is not None:
            label = _translated_metadata(label_i18n, default_language)
            label.set_modeltrans_field(metric, 'label', default_language)
        metric.save()
        print(f"Created metric '{metric}' and linking it to schema '{schema}'")
        return metric

    def get_or_create_dimension(
        self, schema: DatasetSchema, instance_config: InstanceConfig, default_language: str, spec: DimensionSpec
    ) -> Dimension:
        try:
            existing_scope = DimensionScope.objects.get(
                scope_content_type=ContentType.objects.get_for_model(instance_config),
                scope_id=instance_config.pk,
                identifier=spec.id,
            )
        except DimensionScope.DoesNotExist:
            return self.create_dimension(schema, instance_config, default_language, spec)
        print(
            f"There is already a dimension with identifier '{spec.id}' for '{instance_config}'; "
            + 'skipping creation of Dimension, DimensionCategory and DimensionScope instances and '
            + f"linking the existing dimension to the schema '{schema}'"
        )
        dimension = existing_scope.dimension
        if schema.dimensions.filter(dimension=dimension).exists():
            print(f"Dimension '{dimension}' is already linked to schema '{schema}'")
            return dimension
        print(f"Linking dimension '{existing_scope.dimension}' to schema '{schema}'")
        DatasetSchemaDimension.objects.create(schema=schema, dimension=existing_scope.dimension)
        return existing_scope.dimension

    def remove_schema_dimensions_for_column(
        self,
        schema: DatasetSchema,
        instance_config: InstanceConfig,
        column_name: str,
        keep_dimension_identifier: str,
    ) -> None:
        scope_content_type = ContentType.objects.get_for_model(instance_config)
        for schema_dim in schema.dimensions.select_related('dimension'):
            scope = DimensionScope.objects.filter(
                dimension=schema_dim.dimension,
                scope_content_type=scope_content_type,
                scope_id=instance_config.pk,
            ).first()
            if scope is None or scope.identifier == keep_dimension_identifier:
                continue
            dimension_column = schema_dim.column_name or scope.identifier
            if dimension_column != column_name:
                continue
            print(
                f"Removing dimension '{schema_dim.dimension}' from schema '{schema}' "
                + f"because column '{column_name}' is now mapped to '{keep_dimension_identifier}'"
            )
            schema_dim.delete()

    def get_or_create_dimension_from_column(
        self,
        schema: DatasetSchema,
        instance_config: InstanceConfig,
        column_name: str,
        dimension_identifier: str,
        category_identifiers: list[str],
    ) -> Dimension:
        try:
            existing_scope = DimensionScope.objects.get(
                scope_content_type=ContentType.objects.get_for_model(instance_config),
                scope_id=instance_config.pk,
                identifier=dimension_identifier,
            )
        except DimensionScope.DoesNotExist:
            dimension = Dimension.objects.create(name=_label_from_identifier(dimension_identifier))
            DimensionScope.objects.create(
                dimension=dimension,
                scope_content_type=ContentType.objects.get_for_model(instance_config),
                scope_id=instance_config.pk,
                identifier=dimension_identifier,
            )
            print(f"Created dimension '{dimension}' from dataset column '{column_name}'")
        else:
            dimension = existing_scope.dimension
            print(f"There is already a dimension with identifier '{dimension_identifier}' for '{instance_config}'")

        existing_categories = set(dimension.categories.values_list('identifier', flat=True))
        created_count = 0
        for category_identifier in category_identifiers:
            if category_identifier in existing_categories:
                continue
            DimensionCategory.objects.create(
                dimension=dimension,
                identifier=category_identifier,
                label=_label_from_identifier(category_identifier),
            )
            created_count += 1
        if created_count:
            print(f"Created {created_count} categories for dimension '{dimension_identifier}'")
        schema_dim = schema.dimensions.filter(dimension=dimension).first()
        column_name_to_store = column_name if column_name != dimension_identifier else None
        if schema_dim is None:
            print(f"Linking dimension '{dimension}' to schema '{schema}'")
            DatasetSchemaDimension.objects.create(
                schema=schema,
                dimension=dimension,
                column_name=column_name_to_store,
            )
        elif schema_dim.column_name != column_name_to_store:
            schema_dim.column_name = column_name_to_store
            schema_dim.save(update_fields=['column_name'])
            print(f"Updated dimension '{dimension}' column mapping in schema '{schema}'")
        else:
            print(f"Dimension '{dimension}' is already linked to schema '{schema}'")
        return dimension

    def create_dimension(
        self, schema: DatasetSchema, instance_config: InstanceConfig, default_language: str, spec: DimensionSpec
    ) -> Dimension:
        dimension = Dimension()
        label = spec.label
        assert isinstance(label, TranslatedString)
        label.set_modeltrans_field(dimension, 'name', default_language)
        dimension.save()
        print(f"Created dimension '{dimension}' and linking it to schema '{schema}'")
        DatasetSchemaDimension.objects.create(schema=schema, dimension=dimension)
        for cat_spec in spec.categories:
            self.create_dimension_category(
                dimension=dimension,
                default_language=default_language,
                spec=cat_spec,
            )
        print(f"Setting scope of dimension '{dimension}' to '{instance_config}'")
        DimensionScope.objects.create(
            dimension=dimension,
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=spec.id,
        )
        return dimension

    def create_dimension_category(
        self, dimension: Dimension, default_language: str, spec: DimensionCategorySpec
    ) -> DimensionCategory:
        cat = DimensionCategory(dimension=dimension, identifier=spec.id)
        label = spec.label
        assert isinstance(label, TranslatedString)
        label.set_modeltrans_field(cat, 'label', default_language)
        cat.save()
        print(f"Created dimension category '{cat}'")
        return cat

    def report_provenance(self, ic: InstanceConfig, provenance: RepoProvenance) -> None:
        """Say which commit the data will come from, and flag a disagreement between the two sources."""
        spec = provenance.spec
        if spec is None:
            print('[yellow]No dataset repository configured for this instance[/yellow]')
            return
        print(f'Instance   {ic.identifier} (config_source={ic.config_source})')
        print(f'Repository {spec.url}')
        print(f'Commit     {spec.commit} [dim](from {provenance.used_source})[/dim]')
        if provenance.sources_disagree:
            other = 'db' if provenance.used_source == 'yaml' else 'yaml'
            other_commit = provenance.db_commit if other == 'db' else provenance.yaml_commit
            print(
                f'[yellow]Warning:[/yellow] the {other} config pins a different commit ({other_commit}). '
                f'Data will be imported from the {provenance.used_source} pin above; '
                f'pass --repo-from {other} to use the other one.'
            )

    def handle(self, *args, **options):  # noqa: C901, PLR0912
        instance_id = options['instance'][0]
        ic = InstanceConfig.objects.get(identifier=instance_id)
        ctx = ic.get_instance().context

        provenance = resolve_repo_provenance(ic, ctx, options['repo_from'])
        apply_repo_provenance(ctx, provenance)
        self.report_provenance(ic, provenance)

        if not options['datasets']:
            dvc_dataset_ids = sorted(ctx.get_all_dvc_dataset_ids())
            if not dvc_dataset_ids:
                # With `use_datasets_from_db`, any identifier that already has a DB row loads as a
                # DBDataset, so it never appears here. An empty list then means "everything is
                # already imported", not "this instance has no datasets" -- say so, rather than
                # silently doing nothing.
                print(
                    '[yellow]No DVC-backed datasets found for this instance.[/yellow] '
                    'If it uses datasets from the DB, every identifier already has a row; '
                    'name the datasets explicitly to re-import them.'
                )
                return
            if not options['all']:
                print('Available datasets:')
                for ds_id in dvc_dataset_ids:
                    print(ds_id)
                return
            ds_ids = dvc_dataset_ids
        else:
            ds_ids = options['datasets']

        ignore_prefixes = options.get('ignore_prefix') or []
        create_dimensions_from_columns = _parse_column_dimension_mappings(options['create_dimension_from_column'])

        # Ensure all prefixes end with / character for filtering
        normalized_prefixes = []
        for prefix in ignore_prefixes:
            if not prefix.endswith('/'):
                normalized_prefixes.append(f'{prefix}/')
            else:
                normalized_prefixes.append(prefix)

        if normalized_prefixes:
            filtered_ds_ids = [ds_id for ds_id in ds_ids if not any(ds_id.startswith(prefix) for prefix in normalized_prefixes)]

            ignored_count = len(ds_ids) - len(filtered_ds_ids)
            if ignored_count > 0:
                display_prefixes = [p[:-1] for p in normalized_prefixes]
                print(f'Ignoring {ignored_count} dataset(s) with prefix(es): {", ".join(display_prefixes)}')

            ds_ids = filtered_ds_ids

        plan_only = options['plan']
        for ds_id in ds_ids:
            with transaction.atomic():
                self.sync_dataset(
                    ic,
                    ctx,
                    ds_id,
                    force=options['force'],
                    metadata_only=options['metadata_only'],
                    create_dimensions_from_columns=create_dimensions_from_columns,
                    plan_only=plan_only,
                    recreate=options['recreate'],
                )
                if plan_only:
                    # Nothing was written, but a plan run must not leave anything behind even
                    # if a code path below the diff decided to create something.
                    transaction.set_rollback(True)
        if plan_only:
            print('\n[bold]--plan: nothing was written.[/bold] Re-run with --force to apply.')
