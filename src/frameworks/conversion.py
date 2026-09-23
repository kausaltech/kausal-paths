"""Explicit migration from copied graphs to a framework release. Never runs on ordinary reads."""

import re
from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from wagtail.models import Revision

from kausal_common.datasets.models import (
    DataPoint,
    DataPointDimensionCategory,
    Dataset,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.i18n.pydantic import set_i18n_context

from frameworks.models import FrameworkConfig
from nodes.dataset_materialization import ensure_dataset_materializations, refresh_dataset_materialization
from nodes.defs.port_def import InputPortDef
from nodes.instance_graph import build_instance_graph
from nodes.instance_serialization import InputBindingSnapshot, InstanceSnapshot, build_instance_snapshot
from nodes.models import InputPortBindingSet, NodeConfig, NodeInputPortBinding
from nodes.template_reference_data import remap_json
from nodes.template_settings import InheritedNodeSettings
from nodes.units import unit_registry
from pages.models import OutcomePage

if TYPE_CHECKING:
    from pydantic import JsonValue

    from frameworks.models import Framework
    from nodes.defs.graph import DatasetMeta
    from nodes.defs.instance_defs import DatasetRepoSpec
    from nodes.defs.port_def import OutputPortDef
    from nodes.instance_serialization import NodeSnapshot
    from nodes.models import InstanceConfig


def _port_key(port: InputPortDef | OutputPortDef) -> tuple[str | None, str | None, str | None]:
    return port.identifier, port.role, str(port.unit) if port.unit is not None else None


def _node_identities(base: InstanceSnapshot, local: InstanceSnapshot) -> dict[str, str]:
    identities: dict[str, str] = {}
    by_name = {node.identifier: node for node in base.nodes}
    for node in local.nodes:
        shared = by_name.get(node.identifier)
        if shared is None:
            continue
        identities[str(node.uuid)] = str(shared.uuid)
        assert node.spec is not None
        assert shared.spec is not None
        for attribute in ('input_ports', 'output_ports'):
            available = list(getattr(shared.spec, attribute))
            for port in getattr(node.spec, attribute):
                match = next((candidate for candidate in available if _port_key(candidate) == _port_key(port)), None)
                if match is None:
                    raise ValueError(f'{node.identifier} has an undeclared framework port: {_port_key(port)}')
                if attribute == 'output_ports' and port.model_dump(
                    mode='json', exclude={'id', 'label', 'is_editable'}
                ) != match.model_dump(mode='json', exclude={'id', 'label', 'is_editable'}):
                    raise ValueError(f'{node.identifier} changes a framework output contract')
                identities[str(port.id)] = str(match.id)
                available.remove(match)
    return identities


def normalize_legacy_dataset_ports(instance: InstanceConfig) -> None:
    """Give legacy dataset bindings an explicit, named input-port definition before migration."""

    for node in instance.nodes.all():
        if node.spec is None:
            continue
        ports = {port.id: port for port in node.spec.input_ports}
        for binding in node.input_bindings.filter(dataset__isnull=False).select_related('metric'):
            assert binding.metric is not None
            port = ports.get(binding.port_id)
            if port is None:
                port = InputPortDef(
                    id=binding.port_id,
                    identifier=re.sub(r'[^A-Za-z0-9_]', '_', binding.metric.name or 'value'),
                    unit=unit_registry.parse_units(binding.metric.unit),
                )
                node.spec.input_ports.append(port)
                ports[port.id] = port
        NodeConfig.objects.filter(pk=node.pk).update(spec=node.spec)


def _extend_category_vocabulary(template: InstanceConfig, examples: list[InstanceConfig], report: list[str]) -> None:
    template_dimensions = {scope.identifier: scope.dimension for scope in DimensionScope.objects.for_instance_config(template)}
    for example in examples:
        for scope in DimensionScope.objects.for_instance_config(example):
            target = template_dimensions.get(scope.identifier)
            if target is None:
                continue
            for category in scope.dimension.categories.all():
                if target.categories.filter(identifier=category.identifier).exists():
                    continue
                DimensionCategory.objects.create(
                    dimension=target,
                    identifier=category.identifier,
                    label=category.label,
                    i18n=category.i18n,
                    spec=category.spec,
                    order=target.categories.count(),
                )
                report.append(f'Added framework category {scope.identifier}/{category.identifier}')


def _extend_input_ports(
    template: InstanceConfig,
    base: InstanceSnapshot,
    local: InstanceSnapshot,
    report: list[str],
) -> None:
    shared_by_name = {node.identifier: node for node in base.nodes}
    for node in local.nodes:
        shared = shared_by_name.get(node.identifier)
        if shared is None or node.spec is None or shared.spec is None:
            continue
        available = list(shared.spec.input_ports)
        for port in node.spec.input_ports:
            match = next((candidate for candidate in available if _port_key(candidate) == _port_key(port)), None)
            if match is not None:
                available.remove(match)
                continue
            added = port.model_copy(deep=True)
            added.id = uuid5(shared.uuid, f'extension:{port.id}')
            added.binding_owner = 'instance'
            shared.spec.input_ports.append(added)
            report.append(f'Added local input {node.identifier}/{added.identifier or added.id}')
        NodeConfig.objects.filter(instance=template, uuid=shared.uuid).update(spec=shared.spec)


def _expose_changed_inputs(
    template: InstanceConfig,
    base: InstanceSnapshot,
    local: InstanceSnapshot,
    report: list[str],
) -> None:
    identities = _node_identities(base, local)
    local_nodes = {node.uuid: node for node in base.nodes}
    grouped = defaultdict(list)
    for binding in local.bindings:
        mapped = InputBindingSnapshot.model_validate(remap_json(binding.model_dump(mode='json'), identities))
        grouped[(mapped.node_id, mapped.port_id)].append(mapped)
    base_groups = defaultdict(list)
    for binding in base.bindings:
        base_groups[(binding.node_id, binding.port_id)].append(binding)
    for node_id, port_id in grouped.keys() | base_groups.keys():
        bindings = grouped[(node_id, port_id)]
        shared = local_nodes.get(node_id)
        if shared is None or shared.spec is None:
            continue

        def signature(binding: InputBindingSnapshot) -> JsonValue:
            data = binding.model_dump(mode='json', exclude={'uuid'})
            source = data['source']
            for field in ('dataset_uuid', 'metric_uuid', 'dataset_revision'):
                source.pop(field, None)
            return data

        local = any(
            b.dataset_source is not None and b.dataset_source.dataset.startswith('kommune/')
            for b in base_groups[(node_id, port_id)]
        )
        if not local and [signature(b) for b in bindings] == [signature(b) for b in base_groups[(node_id, port_id)]]:
            continue
        port = next((port for port in shared.spec.input_ports if port.id == port_id), None)
        if port is None:
            raise ValueError(
                f'Port mapping failed for {shared.identifier}: {port_id}, expected {[p.id for p in shared.spec.input_ports]}'
            )
        if port.binding_owner != 'instance':
            port.binding_owner = 'instance'
            report.append(f'Exposed local input {shared.identifier}/{port.identifier or port.id}')
            NodeConfig.objects.filter(instance=template, uuid=node_id).update(spec=shared.spec)


LOCAL_DATA_SLOT_PREFIX = 'kommune/'


@transaction.atomic
def declare_local_data_slots(framework: Framework) -> list[str]:
    """
    Give the municipality the inputs the template reads from its `kommune/*` data slots.

    A `kommune/` dataset is, by the template's convention, the placeholder for
    data the municipality supplies itself; cities replace it via
    `dataset_replacements`. Every template input bound only to such datasets is
    therefore instance-owned. This is a rule over the template, not an inference
    from example instances, and stands in until the YAML declares ownership on
    the port itself.
    """
    template = framework.template_instance
    if template is None:
        raise ValueError('Framework has no template')
    normalize_legacy_dataset_ports(template)
    snapshot = build_instance_snapshot(template)
    sources = defaultdict(list)
    for binding in snapshot.bindings:
        sources[(binding.node_id, binding.port_id)].append(binding.dataset_source)
    report: list[str] = []
    for node in snapshot.nodes:
        if node.spec is None:
            continue
        changed = False
        for port in node.spec.input_ports:
            bound = sources.get((node.uuid, port.id), [])
            is_slot = bool(bound) and all(
                source is not None and source.dataset.startswith(LOCAL_DATA_SLOT_PREFIX) for source in bound
            )
            if is_slot and port.binding_owner != 'instance':
                port.binding_owner = 'instance'
                changed = True
                report.append(f'Local data slot {node.identifier}/{port.identifier or port.id}')
        if changed:
            NodeConfig.objects.filter(instance=template, uuid=node.uuid).update(spec=node.spec)
    template.invalidate_cache()
    return report


@transaction.atomic
def prepare_template_inputs(framework: Framework, examples: list[InstanceConfig]) -> list[str]:
    """
    Prepare declarations from explicitly selected migration examples, before publishing a release.

    Adds missing optional input ports and exposes input selections that differ in
    the examples. It never alters a calculation class, formula, or output port.
    The returned report records every newly exposed port for method review.
    """
    template = framework.template_instance
    if template is None:
        raise ValueError('Framework has no template')
    report: list[str] = []
    _extend_category_vocabulary(template, examples, report)
    normalize_legacy_dataset_ports(template)
    for example in examples:
        normalize_legacy_dataset_ports(example)
    for example in examples:
        local = build_instance_snapshot(example)
        base = build_instance_snapshot(template)
        _extend_input_ports(template, base, local, report)
        base = build_instance_snapshot(template)
        _expose_changed_inputs(template, base, local, report)
    template.invalidate_cache()
    return report


@transaction.atomic
def share_template_catalogue(framework: Framework) -> None:
    template = framework.template_instance
    if template is None:
        raise ValueError('Framework has no template')
    framework_ct = ContentType.objects.get_for_model(framework)
    for scope in DimensionScope.objects.for_instance_config(template):
        DimensionScope.objects.get_or_create(
            scope_content_type=framework_ct,
            scope_id=framework.pk,
            identifier=scope.identifier,
            defaults={'dimension': scope.dimension, 'order': scope.order},
        )
    for dataset in Dataset.objects.for_instance_config(template):
        DatasetSchemaScope.objects.get_or_create(schema=dataset.schema, scope_content_type=framework_ct, scope_id=framework.pk)


def _adopt_dimensions(instance: InstanceConfig, framework: Framework) -> dict[str, str]:
    framework_ct = ContentType.objects.get_for_model(framework)
    shared = {
        scope.identifier: scope for scope in DimensionScope.objects.filter(scope_content_type=framework_ct, scope_id=framework.pk)
    }
    identities: dict[str, str] = {}
    datasets = Dataset.objects.for_instance_config(instance)
    for local in DimensionScope.objects.for_instance_config(instance):
        target = shared.get(local.identifier)
        if target is None or target.dimension_id == local.dimension_id:
            continue
        cats = {cat.identifier: cat for cat in target.dimension.categories.all()}
        if not set(local.dimension.categories.values_list('identifier', flat=True)) <= set(cats):
            raise ValueError(f'Category vocabulary differs for {local.identifier}')
        identities[str(local.dimension.uuid)] = str(target.dimension.uuid)
        for category in local.dimension.categories.all():
            replacement = cats[category.identifier]
            identities[str(category.uuid)] = str(replacement.uuid)
            DataPointDimensionCategory.objects.filter(data_point__dataset__in=datasets, dimension_category=category).update(
                dimension_category=replacement,
            )
        DatasetSchemaDimension.objects.filter(schema__datasets__in=datasets, dimension=local.dimension).update(
            dimension=target.dimension
        )
        local.delete()
    for schema in DatasetSchema.objects.filter(datasets__in=datasets).distinct():
        DatasetSchema.objects.filter(pk=schema.pk).update(
            category_domain=remap_json(schema.category_domain.model_dump(mode='json'), identities),
        )
    return identities


def _adopt_schemas(instance: InstanceConfig, framework: Framework, identities: dict[str, str]) -> None:
    template = framework.template_instance
    assert template is not None
    template_datasets = list(Dataset.objects.for_instance_config(template).select_related('schema'))
    for dataset in Dataset.objects.for_instance_config(instance).select_related('schema'):
        schema = dataset.schema
        assert schema is not None

        # Natural dataset names and external references identify the intended schema;
        # replacement datasets may have different identifiers but identical contracts.
        def signature(candidate: DatasetSchema) -> tuple[object, ...]:
            return (
                candidate.time_resolution,
                tuple(candidate.dimensions.order_by('order').values_list('dimension_id', 'column_name')),
                tuple(
                    (m.name, m.unit, m.spec, list(m.validation_rules.values_list('rule', flat=True)))
                    for m in candidate.metrics.order_by('order')
                ),
                remap_json(candidate.category_domain.model_dump(mode='json'), identities),
            )

        shape = signature(schema)
        matches = [ds for ds in template_datasets if ds.schema is not None and signature(ds.schema) == shape]
        preferred = next((ds for ds in matches if ds.identifier == dataset.identifier), None)
        if preferred is None and len({ds.schema_id for ds in matches}) == 1:
            preferred = matches[0]
        if preferred is None or preferred.schema_id == schema.pk:
            continue
        target_schema = preferred.schema
        assert target_schema is not None
        if (
            Dataset.objects
            .filter(
                schema=target_schema,
                scope_content_type_id=dataset.scope_content_type_id,
                scope_id=dataset.scope_id,
            )
            .exclude(pk=dataset.pk)
            .exists()
        ):
            # Keep the original schema when another dataset in this scope
            # already occupies the shared one.
            continue
        targets = {metric.name: metric for metric in target_schema.metrics.all()}
        for metric in schema.metrics.all():
            target = targets[metric.name]
            identities[str(metric.uuid)] = str(target.uuid)
            DataPoint.objects.filter(dataset=dataset, metric=metric).update(metric=target)
            NodeInputPortBinding.objects.filter(instance=instance, dataset=dataset, metric=metric).update(metric=target)
        identities[str(schema.uuid)] = str(target_schema.uuid)
        Dataset.objects.filter(pk=dataset.pk).update(schema=target_schema)


def _node_settings(original: NodeSnapshot, shared: NodeSnapshot) -> InheritedNodeSettings:
    assert original.spec is not None
    assert shared.spec is not None
    settings = InheritedNodeSettings(node_uuid=shared.uuid, goals=original.spec.goals, layout=original.layout)
    parameters = {parameter.local_id: parameter for parameter in shared.spec.params}
    for parameter in original.spec.params:
        base = parameters.get(parameter.local_id)
        if base is None:
            raise ValueError(f'Unknown shared parameter {original.identifier}/{parameter.local_id}')
        if parameter.type == 'reference' and base.type != 'reference':
            if not base.is_customizable:
                raise ValueError('Cannot override a fixed framework parameter')
            settings.parameter_sources[parameter.local_id] = parameter.target_id
        elif parameter.model_dump().get('value') != base.model_dump().get('value'):
            if not base.is_customizable:
                raise ValueError('Cannot override a fixed framework parameter')
            settings.parameter_values[parameter.local_id] = parameter.model_dump().get('value')
    return settings


def _same_dataset_repo(left: DatasetRepoSpec | None, right: DatasetRepoSpec | None) -> bool:
    if left is None or right is None:
        return left is right
    return (left.url, left.commit) == (right.url, right.commit)


def _same_external_dataset(left: dict[str, str | None] | None, right: dict[str, str | None] | None) -> bool:
    """
    Whether two placeholders name the same external dataset.

    A placeholder is read at its instance's dataset-repo pin, so its own
    ``commit`` is only the provenance stamp of when the row was created and
    must not decide equality; the caller compares the pins instead.
    """
    if left is None or right is None:
        return False
    return (left.get('repo_url'), left.get('dataset_id')) == (right.get('repo_url'), right.get('dataset_id'))


def _superseded_datasets(instance: InstanceConfig, base: InstanceSnapshot) -> list[tuple[Dataset, DatasetMeta]]:
    """
    Return the instance's datasets that the template supplies identically, each with its template dataset.

    Defaults are reused only when their actual input payloads agree; otherwise the local data is retained.
    """
    defaults = {dataset.identifier: dataset for dataset in base.datasets}
    revisions = {pin.dataset_uuid: pin.revision_id for pin in base.dataset_revisions}
    same_pin = _same_dataset_repo(instance.ensure_spec().dataset_repo, base.spec.dataset_repo)
    superseded: list[tuple[Dataset, DatasetMeta]] = []
    for dataset in Dataset.objects.for_instance_config(instance).select_related('schema'):
        target = defaults.get(dataset.identifier)
        if target is None or dataset.schema is None:
            continue
        if dataset.identifier and dataset.identifier.startswith(LOCAL_DATA_SLOT_PREFIX):
            # The municipality's own data, even while it still equals the template's
            # placeholder values: it must stay the dataset its inputs read, or edits
            # to it would have no effect.
            continue
        if dataset.is_external_placeholder:
            equal = (
                target.is_external_placeholder and same_pin and _same_external_dataset(dataset.external_ref, target.external_ref)
            )
        elif target.id in revisions:
            materialization = ensure_dataset_materializations([dataset])[dataset.pk]
            equal = materialization.content.get('data') == Revision.objects.get(pk=revisions[target.id]).content.get('data')
        else:
            equal = False
        if not equal:
            continue
        target_metrics = {metric.identifier for metric in target.metrics}
        if set(dataset.schema.metrics.values_list('name', flat=True)) != target_metrics:
            continue
        superseded.append((dataset, target))
    return superseded


def _dataset_identities(superseded: list[tuple[Dataset, DatasetMeta]]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for dataset, target in superseded:
        assert dataset.schema is not None
        target_metrics = {metric.identifier: metric for metric in target.metrics}
        identities[str(dataset.uuid)] = str(target.id)
        for metric in dataset.schema.metrics.all():
            identities[str(metric.uuid)] = str(target_metrics[metric.name].id)
    return identities


def _delete_datasets(instance: InstanceConfig, datasets: list[Dataset]) -> list[str]:
    """
    Delete local copies the instance no longer reads, with schemas nothing else uses.

    Refuses a dataset any effective binding still references, so only true
    leftovers go: after conversion the instance reads the template's copy.
    """
    import json

    referenced = json.dumps([b.model_dump(mode='json') for b in build_instance_snapshot(instance).bindings])
    removed: list[str] = []
    for dataset in datasets:
        if str(dataset.uuid) in referenced:
            continue
        schema = dataset.schema
        identifier = dataset.identifier or str(dataset.uuid)
        dataset.delete()
        if (
            schema is not None
            and not schema.datasets.exists()
            and not DatasetSchemaScope.objects
            .filter(schema=schema)
            .exclude(scope_content_type=ContentType.objects.get_for_model(instance), scope_id=instance.pk)
            .exists()
        ):
            schema.delete()
        removed.append(identifier)
    return removed


def _bind_local_data_slots(instance: InstanceConfig) -> int:
    """
    Make every input that reads a `kommune/*` slot read the instance's own copy of it, where it has one.

    The template may read a slot from more ports than the instance's copied
    graph did (availability and data-quality siblings); those would otherwise
    fall back to the template's placeholder copy, so one slot would be read
    from two datasets. Returns the number of ports redirected.
    """
    own = {
        dataset.identifier: dataset
        for dataset in Dataset.objects.for_instance_config(instance).select_related('schema')
        if dataset.identifier and dataset.identifier.startswith(LOCAL_DATA_SLOT_PREFIX)
    }

    def is_foreign_copy(binding: InputBindingSnapshot) -> bool:
        source = binding.dataset_source
        return source is not None and source.dataset in own and source.dataset_uuid != own[source.dataset].uuid

    by_port: dict[tuple[UUID, UUID], list[InputBindingSnapshot]] = defaultdict(list)
    for binding in build_instance_snapshot(instance).bindings:
        by_port[(binding.node_id, binding.port_id)].append(binding)
    redirected = 0
    for (node_uuid, port_uuid), bindings in by_port.items():
        if not any(is_foreign_copy(binding) for binding in bindings):
            continue
        replacement = []
        for binding in bindings:
            source = binding.dataset_source
            if source is None or not is_foreign_copy(binding):
                replacement.append(binding)
                continue
            local = own[source.dataset]
            assert local.schema is not None
            metric = local.schema.metrics.get(name=source.metric)
            update = {'dataset_uuid': local.uuid, 'metric_uuid': metric.uuid, 'dataset_revision': None}
            replacement.append(binding.model_copy(update={'source': source.model_copy(update=update)}))
        InputPortBindingSet.objects.update_or_create(
            instance=instance, node_uuid=node_uuid, port_uuid=port_uuid, defaults={'bindings': replacement}
        )
        redirected += 1
    return redirected


@transaction.atomic
def remove_superseded_datasets(instance: InstanceConfig) -> list[str]:
    """Delete a framework instance's leftover copies of datasets it now reads from its template revision."""
    if instance.template_revision is None:
        raise ValueError('%s does not inherit from a template' % instance.identifier)
    base = InstanceSnapshot.from_serialized_data(instance.template_revision.content['model_snapshot']['structured'])
    removed = _delete_datasets(instance, [dataset for dataset, _target in _superseded_datasets(instance, base)])
    instance.invalidate_cache()
    return removed


def _conversion_node_settings(before: InstanceSnapshot, shared: dict[str | None, NodeSnapshot]) -> list[InheritedNodeSettings]:
    settings = []
    for node in before.nodes:
        target = shared.get(node.identifier)
        if target is None:
            continue
        assert node.spec is not None
        assert target.spec is not None
        left = node.spec.model_dump(mode='json', exclude={'input_ports', 'output_ports', 'params', 'goals'})
        right = target.spec.model_dump(mode='json', exclude={'input_ports', 'output_ports', 'params', 'goals'})
        if left != right:
            raise ValueError(f'{node.identifier} changes the framework calculation definition')
        settings.append(_node_settings(node, target))
    return settings


def _convert_bindings(
    config: InstanceConfig,
    before: InstanceSnapshot,
    base: InstanceSnapshot,
    identities: dict[str, str],
) -> None:
    groups = defaultdict(list)
    for binding in before.bindings:
        mapped = InputBindingSnapshot.model_validate(remap_json(binding.model_dump(mode='json'), identities))
        mapped.uuid = uuid4()
        groups[(mapped.node_id, mapped.port_id)].append(mapped)
    base_groups = defaultdict(list)
    for binding in base.bindings:
        base_groups[(binding.node_id, binding.port_id)].append(binding)
    # Every input of a local node is represented here as well, so shared
    # sources never require an FK to a mutable template NodeConfig.
    for node_id, port_id in groups.keys() | base_groups.keys():
        bindings = groups[(node_id, port_id)]
        base_bindings = base_groups.get((node_id, port_id), [])

        def comparable(binding: InputBindingSnapshot) -> dict[str, JsonValue]:
            data = binding.model_dump(mode='json', exclude={'uuid'})
            data['source'].pop('dataset_revision', None)
            return data

        if [comparable(b) for b in bindings] != [comparable(b) for b in base_bindings]:
            InputPortBindingSet.objects.create(instance=config, node_uuid=node_id, port_uuid=port_id, bindings=bindings)


@transaction.atomic
def convert_to_framework(instance: InstanceConfig, framework: Framework, revision: Revision) -> dict[str, int]:
    """Adopt shared identities and remove copied nodes, preserving local bindings and data."""
    if instance.config_source != 'database':
        raise ValueError('Sync and verify the database-backed instance before converting it')
    if instance.has_framework_config():
        raise ValueError('Instance already has framework membership; use an explicit release upgrade instead')

    base = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'])
    normalize_legacy_dataset_ports(instance)
    with set_i18n_context(instance.primary_language, instance.other_languages):
        before = build_instance_snapshot(instance)
        identities = _node_identities(base, before)
        shared = {node.identifier: node for node in base.nodes}
        settings = _conversion_node_settings(before, shared)
        identities.update(_adopt_dimensions(instance, framework))
        _adopt_schemas(instance, framework, identities)
        superseded = _superseded_datasets(instance, base)
        identities.update(_dataset_identities(superseded))
        FrameworkConfig.objects.create(
            framework=framework,
            instance_config=instance,
            organization_name=instance.organization.name,
        )
        instance.template_revision = revision
        instance.node_settings = settings
        instance.save(update_fields=['template_revision', 'node_settings'])
        _convert_bindings(instance, before, base, identities)
        NodeInputPortBinding.objects.filter(instance=instance).delete()
        shared_rows = instance.nodes.filter(identifier__in=shared)
        # Page FKs are authoring references; point them at the surviving template
        # row with the same shared identity before removing the copied rows.

        template = framework.template_instance
        assert template is not None
        template_nodes = {node.identifier: node for node in template.nodes.all()}
        for node in shared_rows:
            OutcomePage.objects.filter(outcome_node=node).update(outcome_node=template_nodes[node.identifier])
            instance.nodes.filter(indicator_node=node).update(indicator_node=template_nodes[node.identifier])
        count = shared_rows.count()
        shared_rows.delete()
        # Local specs can reference shared identities (e.g. indicator nodes).
        for node in instance.nodes.all():
            if node.spec is not None:
                type(node).objects.filter(pk=node.pk).update(spec=remap_json(node.spec.model_dump(mode='json'), identities))
        # The instance now reads these from the template; its identical copies would only
        # show up as editable datasets that feed nothing.
        removed_datasets = _delete_datasets(instance, [dataset for dataset, _target in superseded])
        _bind_local_data_slots(instance)

        for dataset in Dataset.objects.for_instance_config(instance).filter(is_external_placeholder=False):
            refresh_dataset_materialization(dataset, touch=False)
        instance.refresh_from_db()
        effective = build_instance_snapshot(instance)

        build_instance_graph(effective)
        instance.invalidate_cache()
        return {
            'shared_nodes': count,
            'local_nodes': instance.nodes.count(),
            'binding_overrides': instance.binding_overrides.count(),
            'removed_datasets': len(removed_datasets),
        }
