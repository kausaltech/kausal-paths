"""
Rewrite stored dataset payloads to `DatasetSnapshot` v2 and instance revisions to snapshot v15.

See docs/plans/node-owned-datasets.md, step 4. The materializations are derived state and
are deleted: `ensure_dataset_materializations` rebuilds them from the live rows on first use.
The published dataset revisions are the record of what was published, so they are
upgraded, taking their structure from the instance revision that pinned them and the uuids
of the rows they still describe. Both hash chains that cover them are then restamped: the
pins' content hashes in the instance revisions, and the template content hash each
revision composed from a template records.

The code this calls (`datasets.legacy_snapshot`, `upgrade_dataset_catalog_v15`) goes when
the migrations are next squashed.
"""

from typing import Any

from django.db import migrations


def _scope_instance_pk(dataset: Any, apps: Any) -> int | None:
    model = dataset.scope_content_type.model
    if model == 'instanceconfig':
        return dataset.scope_id
    if model == 'nodeconfig':
        node = apps.get_model('nodes', 'NodeConfig').objects.filter(pk=dataset.scope_id).first()
        return node.instance_id if node is not None else None
    return None


def _live_dimensions(dataset: Any, instance_pk: int | None, apps: Any) -> list[Any]:
    """Return the dimensions of the dataset's schema, named as its instance names them."""
    from nodes.defs.graph import DimensionCategoryMeta, DimensionMeta

    DatasetSchemaDimension = apps.get_model('datasets', 'DatasetSchemaDimension')
    DimensionScope = apps.get_model('datasets', 'DimensionScope')
    result = []
    for schema_dimension in DatasetSchemaDimension.objects.filter(schema_id=dataset.schema_id).select_related('dimension'):
        dimension = schema_dimension.dimension
        scopes = list(DimensionScope.objects.filter(dimension=dimension).exclude(identifier=None).order_by('pk'))
        own = [s for s in scopes if s.scope_content_type.model == 'instanceconfig' and s.scope_id == instance_pk]
        scope = (own or scopes or [None])[0]
        result.append(
            DimensionMeta(
                id=dimension.uuid,
                identifier=scope.identifier if scope is not None else str(dimension.uuid),
                categories=tuple(
                    DimensionCategoryMeta(id=category.uuid, identifier=category.identifier)
                    for category in dimension.categories.all()
                ),
            )
        )
    return result


def _live_meta(dataset: Any, apps: Any) -> Any:
    """Return the dataset's structure as its rows describe it now."""
    from datasets.validation_rules import validation_rule_adapter
    from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, ValidationRuleMeta

    DatasetMetric = apps.get_model('datasets', 'DatasetMetric')
    DatasetSchemaDimension = apps.get_model('datasets', 'DatasetSchemaDimension')
    metrics = []
    for metric in DatasetMetric.objects.filter(schema_id=dataset.schema_id).order_by('order', 'pk'):
        spec = metric.spec or {}
        metrics.append(
            DatasetMetricMeta(
                id=metric.uuid,
                identifier=metric.name or metric.label or str(metric.uuid),
                label=metric.label or None,
                unit=metric.unit,
                quantity=spec.get('quantity'),
                order=metric.order,
                validation_rules=tuple(
                    ValidationRuleMeta(id=rule.uuid, rule=validation_rule_adapter.validate_python(rule.rule))
                    for rule in metric.validation_rules.order_by('order', 'pk')
                ),
                quality_of=spec.get('quality_of'),
            )
        )
    return DatasetMeta(
        id=dataset.uuid,
        identifier=dataset.identifier,
        schema_id=dataset.schema.uuid,
        metrics=tuple(metrics),
        declared_dimension_ids=tuple(
            DatasetSchemaDimension.objects
            .filter(schema_id=dataset.schema_id)
            .order_by('order', 'pk')
            .values_list('dimension__uuid', flat=True)
        ),
    )


def upgrade(apps: Any, schema_editor: Any) -> None:  # noqa: C901, PLR0912, PLR0915
    from kausal_common.i18n.pydantic import set_i18n_context

    from datasets.legacy_snapshot import LiveIdentities, complete_catalog_entry, upgrade_dataset_snapshot_v1
    from datasets.materialization import hash_dataset_content
    from datasets.snapshot import DatasetSnapshot
    from nodes.instance_serialization import SNAPSHOT_SCHEMA_VERSION, InstanceSnapshot
    from nodes.template_graph import snapshot_content_hash

    alias = schema_editor.connection.alias
    ContentType = apps.get_model('contenttypes', 'ContentType')
    Revision = apps.get_model('wagtailcore', 'Revision')
    Dataset = apps.get_model('datasets', 'Dataset')
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    DataQualityLevel = apps.get_model('frameworks', 'DataQualityLevel')
    models = {
        name: apps.get_model('datasets', name)
        for name in ('DataPoint', 'DataPointDimensionCategory', 'DataPointComment', 'DatasetSourceReference')
    }

    apps.get_model('nodes', 'DatasetMaterialization').objects.using(alias).all().delete()

    dataset_ct = ContentType.objects.using(alias).filter(app_label='datasets', model='dataset').first()
    instance_ct = ContentType.objects.using(alias).filter(app_label='nodes', model='instanceconfig').first()
    if dataset_ct is None or instance_ct is None:
        return
    languages = {
        pk: (lang or 'en', others or [])
        for pk, lang, others in InstanceConfig.objects.using(alias).values_list('pk', 'primary_language', 'other_languages')
    }

    # The instance revisions, upgraded to v15, and what each pinned dataset revision's structure was.
    instance_revisions = list(
        Revision.objects.using(alias).filter(content_type=instance_ct, content__model_snapshot__structured__isnull=False)
    )
    snapshots: dict[int, Any] = {}
    pinned: dict[int, tuple[Any, list[Any], tuple[str, list[str]]]] = {}
    for revision in instance_revisions:
        language = languages.get(int(revision.object_id), ('en', []))
        with set_i18n_context(*language):
            snapshot = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'], compose=False)
        snapshots[revision.pk] = snapshot
        catalog = {dataset.id: dataset for dataset in snapshot.all_datasets()}
        for pin in snapshot.dataset_revisions:
            if pin.dataset_uuid in catalog and pin.revision_id not in pinned:
                pinned[pin.revision_id] = (catalog[pin.dataset_uuid], list(snapshot.dimensions), language)

    scores = {str(uuid): float(score) for uuid, score in DataQualityLevel.objects.using(alias).values_list('uuid', 'score')}
    upgraded: dict[int, Any] = {}
    hashes: dict[int, str] = {}
    for revision in Revision.objects.using(alias).filter(content_type=dataset_ct).order_by('pk'):
        dataset = (
            Dataset.objects.using(alias).filter(pk=int(revision.object_id)).select_related('schema', 'scope_content_type').first()
        )
        content = revision.content
        base, dimensions, pinned_language = pinned.get(revision.pk, (None, None, None))
        instance_pk = _scope_instance_pk(dataset, apps) if dataset is not None else None
        language = pinned_language or (languages.get(instance_pk, ('en', [])) if instance_pk is not None else ('en', []))
        if content.get('schema_version', 1) < 2:
            if dataset is not None:
                live_dimensions = _live_dimensions(dataset, instance_pk, apps)
                dimensions = [*(dimensions or []), *(d for d in live_dimensions if d.id not in {x.id for x in dimensions or []})]
                if base is None:
                    base = _live_meta(dataset, apps)
            with set_i18n_context(*language):
                content = upgrade_dataset_snapshot_v1(
                    content,
                    base=base,
                    dimensions=dimensions or [],
                    identities=LiveIdentities.load(dataset.pk, dataset.uuid, models) if dataset is not None else None,
                    scores=scores,
                )
            revision.content = content
            revision.save(update_fields=['content'])
        with set_i18n_context(*language):
            upgraded[revision.pk] = DatasetSnapshot.model_validate(content)
        hashes[revision.pk] = hash_dataset_content(content)

    # The instance revisions: pin hashes, and the structure v15 adds to their catalog entries.
    for revision in instance_revisions:
        snapshot = snapshots[revision.pk]
        revisions_by_dataset = {pin.dataset_uuid: pin.revision_id for pin in snapshot.dataset_revisions}
        for pin in snapshot.dataset_revisions:
            if pin.revision_id in hashes:
                pin.content_hash = hashes[pin.revision_id]

        snapshot.datasets = [
            complete_catalog_entry(entry, upgraded.get(revisions_by_dataset.get(entry.id, -1))) for entry in snapshot.datasets
        ]
        for node in snapshot.nodes:
            node.datasets = [complete_catalog_entry(entry, upgraded.get(revisions_by_dataset.get(entry.id, -1))) for entry in node.datasets]

    # Template hashes last: a template's own revision has to be final before it is hashed.
    template_hashes = {pk: snapshot_content_hash(snapshot) for pk, snapshot in snapshots.items()}
    for revision in instance_revisions:
        snapshot = snapshots[revision.pk]
        if snapshot.template_revision_id in template_hashes:
            snapshot.template_content_hash = template_hashes[snapshot.template_revision_id]
        revision.content['model_snapshot'] = {
            **revision.content['model_snapshot'],
            'schema_version': SNAPSHOT_SCHEMA_VERSION,
            'structured': snapshot.model_dump(mode='json'),
        }
        revision.save(update_fields=['content'])


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0084_dataset_validation_rules_hash'),
        ('datasets', '0040_dimension_category_short_label'),
        ('frameworks', '0034_bisko_quality_scheme_identifier'),
        ('wagtailcore', '0097_baselogentry_uuid_action_timestamp_indexes'),
    ]

    operations = [
        migrations.RunPython(upgrade, migrations.RunPython.noop),
    ]
