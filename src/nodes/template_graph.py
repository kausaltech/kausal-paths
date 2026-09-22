"""Compose a pinned framework method with a dependent instance's own graph and input selections."""

from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from wagtail.models import Revision

import networkx as nx

from kausal_common.datasets.models import Dataset
from kausal_common.i18n.pydantic import set_i18n_context

from nodes.constraints.validation import InstanceConstraintError, solve_instance_constraints
from nodes.dataset_materialization import ensure_dataset_materializations
from nodes.instance_graph import NodeEditContext, NodeMeta, build_instance_graph
from nodes.instance_graph_cache import resolve_instance_source
from nodes.instance_serialization import (
    DatasetMetricSource,
    DatasetRevisionPinSnapshot,
    InputBindingSnapshot,
    InstanceSnapshot,
    NodePortSource,
    NodeSnapshot,
    build_instance_snapshot,
    dataset_meta_from_model,
)
from nodes.models import InputPortBindingSet, InstanceConfig, InstanceRevisionDatasetPin, PreferredInstanceSource
from nodes.template_reference_data import release_reference_materializations
from params.param import ReferenceParameter

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.defs.graph import DatasetMeta, DimensionMeta
    from users.models import User


def compose_template_snapshot(
    instance: InstanceConfig,
    local: InstanceSnapshot,
    *,
    dataset_revision_pins: dict[int, DatasetRevisionPinSnapshot] | None = None,
) -> InstanceSnapshot:
    revision = instance.template_revision
    if revision is None:
        return local
    base = template_snapshot(instance)
    nodes = _released_nodes(base, revision.pk)
    shared_ids = {node.uuid for node in nodes}
    _apply_node_settings(instance, local, {node.uuid: node for node in nodes})
    shared_names = {node.identifier for node in nodes}
    if any(node.uuid in shared_ids or node.identifier in shared_names for node in local.nodes):
        raise ValueError('Local nodes cannot shadow framework nodes')
    nodes.extend(local.nodes)
    by_id = {node.uuid: node for node in nodes}
    bindings = [*base.bindings, *local.bindings]
    bindings = _apply_binding_overrides(instance, by_id, shared_ids, bindings)

    _validate_edges(by_id, bindings)

    dimensions = _compose_dimensions(base, local)

    datasets, used = _effective_datasets(instance, base, local, bindings)
    # Update only explicit publication pins; reference defaults retain the release's pins.
    pins_by_uuid = {pin.dataset_uuid: pin for pin in (dataset_revision_pins or {}).values()}
    pinned_bindings = []
    for binding in bindings:
        source = binding.source
        if isinstance(source, DatasetMetricSource) and source.dataset_uuid in pins_by_uuid:
            source = source.model_copy(update={'dataset_revision': pins_by_uuid[source.dataset_uuid].revision_id})
            pinned_bindings.append(binding.model_copy(update={'source': source}))
        else:
            pinned_bindings.append(binding)
    for dataset_uuid, pin in pins_by_uuid.items():
        if dataset_uuid in datasets:
            datasets[dataset_uuid] = datasets[dataset_uuid].model_copy(update={'revision_id': pin.revision_id})
    return local.model_copy(
        update={
            'template_revision_id': revision.pk,
            'nodes': nodes,
            'bindings': pinned_bindings,
            'dimensions': dimensions,
            'datasets': [dataset for uuid, dataset in datasets.items() if uuid in used],
            'dataset_revisions': list(
                {
                    pin.dataset_uuid: pin
                    for pin in [*base.dataset_revisions, *local.dataset_revisions]
                    if pin.dataset_uuid in used
                }.values()
            ),
        }
    )


def _effective_datasets(
    instance: InstanceConfig,
    base: InstanceSnapshot,
    local: InstanceSnapshot,
    bindings: list[InputBindingSnapshot],
) -> tuple[dict[UUID, DatasetMeta], set[UUID]]:
    datasets = {dataset.id: dataset for dataset in base.datasets}
    datasets.update({dataset.id: dataset for dataset in local.datasets})
    used = {
        b.source.dataset_uuid for b in bindings if isinstance(b.source, DatasetMetricSource) and b.source.dataset_uuid is not None
    }
    missing_datasets = (
        Dataset.objects
        .filter(uuid__in=used - datasets.keys())
        .select_related('schema')
        .prefetch_related(
            'schema__metrics__validation_rules',
            'schema__dimensions__dimension',
        )
    )
    for dataset in missing_datasets:
        datasets[dataset.uuid] = dataset_meta_from_model(dataset, primary_language=instance.primary_language)
    if used - datasets.keys() or any(
        isinstance(b.source, DatasetMetricSource) and b.source.dataset_uuid is None for b in bindings
    ):
        raise ValueError('A framework binding references an unknown dataset')
    names = [dataset.identifier for key, dataset in datasets.items() if key in used]
    if len(names) != len(set(names)):
        raise ValueError('Effective input datasets must have distinct identifiers')
    return datasets, used


def _released_nodes(base: InstanceSnapshot, revision_id: int) -> list[NodeSnapshot]:
    nodes = []
    for original in base.nodes:
        node = original.model_copy(deep=True)
        node.template_revision_id = revision_id
        nodes.append(node)
    return nodes


def _compose_dimensions(base: InstanceSnapshot, local: InstanceSnapshot) -> list[DimensionMeta]:
    dimensions = {dimension.id: dimension for dimension in base.dimensions}
    dimension_names = {dimension.identifier: dimension.id for dimension in base.dimensions}
    for dimension in local.dimensions:
        if dimension.identifier in dimension_names and dimension_names[dimension.identifier] != dimension.id:
            raise ValueError(f'Local dimension shadows framework dimension {dimension.identifier}')
        if dimension.id not in dimensions:
            dimensions[dimension.id] = dimension

    return list(dimensions.values())


def template_snapshot(instance: InstanceConfig) -> InstanceSnapshot:
    revision = instance.template_revision
    if revision is None or revision.content_type.pk != ContentType.objects.get_for_model(InstanceConfig).pk:
        raise ValueError('Template revision must belong to an instance')
    if str(instance.pk) == str(revision.object_id):
        raise ValueError('An instance cannot inherit itself')
    if instance.has_framework_config():
        template_id = instance.framework_config.framework.template_instance_id
        if str(template_id) != str(revision.object_id):
            raise ValueError('Template revision belongs to another framework')
    snapshot = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'])
    if snapshot.template_revision_id is not None:
        raise ValueError('Nested template inheritance is not supported')
    return snapshot


@transaction.atomic
def publish_template_instance(
    template: InstanceConfig,
    *,
    user: User | None = None,
    reference_data: dict[str, Dataset] | None = None,
) -> Revision:
    """Publish a complete template and atomically advance all dependent drafts."""

    template = InstanceConfig.objects.select_for_update().get(pk=template.pk)
    if template.config_source != 'database' or template.template_revision_id is not None:
        raise ValueError('A template must be a standalone database-backed instance')
    dependents = list(
        InstanceConfig.objects
        .select_for_update()
        .filter(
            template_revision__content_type=ContentType.objects.get_for_model(InstanceConfig),
            template_revision__object_id=str(template.pk),
        )
        .order_by('pk')
    )
    with set_i18n_context(template.primary_language, template.other_languages):
        template.validate_draft_constraints()
        revision = _freeze_template_revision(template, user, reference_data or {})
    for instance in dependents:
        with set_i18n_context(instance.primary_language, instance.other_languages):
            source = resolve_instance_source(instance, PreferredInstanceSource.DRAFT)
            before = solve_instance_constraints(instance, build_instance_graph(build_instance_snapshot(instance)), source)
            instance.template_revision = revision
            instance.save(update_fields=['template_revision'])
            after = solve_instance_constraints(instance, build_instance_graph(build_instance_snapshot(instance)), source)
            new_conflicts = tuple(conflict for conflict in after.conflicts if conflict not in before.conflicts)
            if new_conflicts:
                raise InstanceConstraintError(new_conflicts)
            instance.invalidate_cache()
    template.publish(revision, user=user)
    template.invalidate_cache()
    return revision


@transaction.atomic
def replace_input_port_bindings(
    instance: InstanceConfig,
    node_uuid: UUID,
    port_uuid: UUID,
    bindings: list[InputBindingSnapshot] | None,
) -> None:
    """Replace a port's input selection; None restores the template/local default. Caller authorizes and audits the edit."""

    instance = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
    if instance.is_locked:
        raise ValidationError('Instance is locked')
    if instance.template_revision_id is None:
        raise ValidationError('Instance does not use a published template revision')
    config = instance
    snapshot = build_instance_snapshot(instance)
    _require_editable_port(snapshot, node_uuid, port_uuid)
    before = build_instance_graph(build_instance_snapshot(instance))
    if bindings is None:
        config.binding_overrides.filter(node_uuid=node_uuid, port_uuid=port_uuid).delete()
    else:
        permitted_datasets = set(Dataset.objects.for_instance_config(instance).values_list('uuid', flat=True))
        permitted_datasets.update(dataset.id for dataset in template_snapshot(instance).datasets)
        for binding in bindings:
            if isinstance(binding.source, DatasetMetricSource) and binding.source.dataset_uuid not in permitted_datasets:
                raise ValidationError('Dataset belongs to another instance')
        InputPortBindingSet.objects.update_or_create(
            instance=instance,
            node_uuid=node_uuid,
            port_uuid=port_uuid,
            defaults={'bindings': bindings},
        )
    try:
        after = build_instance_graph(build_instance_snapshot(instance))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    source = resolve_instance_source(instance, PreferredInstanceSource.DRAFT)
    baseline = solve_instance_constraints(instance, before, source)
    candidate = solve_instance_constraints(instance, after, source)
    conflicts = [conflict for conflict in candidate.conflicts if conflict not in baseline.conflicts]
    if conflicts:
        raise InstanceConstraintError(tuple(conflicts))
    instance.invalidate_cache()


def _require_editable_port(snapshot: InstanceSnapshot, node_uuid: UUID, port_uuid: UUID) -> None:
    node = next((node for node in snapshot.nodes if node.uuid == node_uuid), None)
    if node is None or node.spec is None:
        raise ValidationError('Unknown input node')
    port = next((port for port in node.spec.input_ports if port.id == port_uuid), None)
    if port is None:
        raise ValidationError('Unknown input port')
    if not NodeMeta.can_edit(
        NodeEditContext(can_change_instance=True, is_draft=True, is_template=False),
        inherited=node.template_revision_id is not None,
        binding_owner=port.binding_owner,
    ):
        raise ValidationError('Template controls bindings of this port')


def _validate_edges(by_id: dict[UUID, NodeSnapshot], bindings: list[InputBindingSnapshot]) -> None:
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    for binding in bindings:
        if binding.node_id not in by_id:
            raise ValueError(f'Missing binding target {binding.node_id}')
        if isinstance(binding.source, NodePortSource):
            source = by_id.get(binding.source.node_id)
            if source is None or source.spec is None or not any(p.id == binding.source.port_id for p in source.spec.output_ports):
                raise ValueError('Binding references an output outside the effective graph')
            graph.add_edge(source.uuid, binding.node_id)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('The composed framework graph contains a cycle')


def _apply_node_settings(config: InstanceConfig, local: InstanceSnapshot, shared_nodes: dict[UUID, NodeSnapshot]) -> None:
    for settings in config.node_settings:
        node = shared_nodes.get(settings.node_uuid)
        if node is None or node.spec is None:
            raise ValueError('Node settings reference a node outside the published template revision')
        if settings.goals is not None:
            node.spec.goals = settings.goals
        if settings.layout is not None:
            node.layout = settings.layout
        parameters = {parameter.local_id: parameter for parameter in node.spec.params}
        for identifier in settings.parameter_values.keys() | settings.parameter_sources.keys():
            parameter = parameters.get(identifier)
            if parameter is None or not parameter.is_customizable:
                raise ValueError(f'Parameter {identifier} is controlled by the framework')
            if identifier in settings.parameter_sources:
                target = next((p for p in local.spec.params if p.local_id == settings.parameter_sources[identifier]), None)
                if target is None or target.type != parameter.type:
                    raise ValueError('A parameter source must reference a compatible instance parameter')
                parameters[identifier] = ReferenceParameter(local_id=identifier, target_id=target.local_id)
            else:
                value = settings.parameter_values[identifier]
                parameter.set(value)
        node.spec.params = list(parameters.values())


def _apply_binding_overrides(
    config: InstanceConfig,
    by_id: dict[UUID, NodeSnapshot],
    shared_ids: set[UUID],
    bindings: list[InputBindingSnapshot],
) -> list[InputBindingSnapshot]:
    for override in config.binding_overrides.all():
        node = by_id.get(override.node_uuid)
        if node is None or node.spec is None:
            raise ValueError(f'Binding override targets missing node {override.node_uuid}')
        port = next((port for port in node.spec.input_ports if port.id == override.port_uuid), None)
        if port is None:
            raise ValueError(f'Binding override targets missing port {override.port_uuid}')
        if not NodeMeta.can_edit(
            NodeEditContext(can_change_instance=True, is_draft=True, is_template=False),
            inherited=node.uuid in shared_ids,
            binding_owner=port.binding_owner,
        ):
            raise ValueError(f'Template controls bindings of {node.identifier}/{port.id}')
        if not port.multi and len(override.bindings) > 1:
            raise ValueError('A single input port cannot accept multiple bindings')
        for position, binding in enumerate(override.bindings):
            if (binding.node_id, binding.port_id, binding.position) != (node.uuid, port.id, position):
                raise ValueError('Override bindings must target their owning port in dense position order')
        bindings = [b for b in bindings if (b.node_id, b.port_id) != (node.uuid, port.id)]
        bindings.extend(override.bindings)

    return bindings


def _freeze_template_revision(template: InstanceConfig, user: User | None, reference_data: dict[str, Dataset]) -> Revision:
    """Freeze method inputs without treating empty local slots as completed reporting data."""

    snapshot = build_instance_snapshot(template)
    datasets = list(
        Dataset.objects
        .select_for_update(of=('self',))
        .exclude(
            identifier__in=[key for key, ds in reference_data.items() if ds.is_external_placeholder],
        )
        .filter(
            Q(is_external_placeholder=False)
            | Q(identifier__in=[key for key, ds in reference_data.items() if not ds.is_external_placeholder]),
            uuid__in=[dataset.id for dataset in snapshot.datasets],
        )
    )
    materializations = ensure_dataset_materializations([dataset for dataset in datasets if not dataset.is_external_placeholder])
    if reference_data:
        materializations = release_reference_materializations(
            datasets, materializations, {key: ds for key, ds in reference_data.items() if not ds.is_external_placeholder}
        )
    content_type = ContentType.objects.get_for_model(Dataset)
    pins = {}
    revisions = {}
    for dataset in datasets:
        materialization = materializations[dataset.pk]
        revision = Revision.objects.create(
            content_type=content_type,
            base_content_type=content_type,
            object_id=str(dataset.pk),
            user=user,
            object_str=dataset.identifier or str(dataset.uuid),
            content=materialization.content,
        )
        revisions[dataset.pk] = revision
        pins[dataset.pk] = DatasetRevisionPinSnapshot(
            dataset_uuid=dataset.uuid,
            identifier=dataset.identifier,
            revision_id=revision.pk,
            content_hash=materialization.content_hash,
            generation=materialization.generation,
            forecast_from=materialization.forecast_from,
        )
    template._publication_dataset_revision_pins = pins
    try:
        revision = template.save_revision(user=user)
    finally:
        template._publication_dataset_revision_pins = None
    # A historical reference edition may materialize a formerly external placeholder.
    snapshot = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'])
    pinned_ids = {pin.dataset_uuid for pin in pins.values()}
    snapshot.datasets = [
        dataset.model_copy(update={'is_external_placeholder': False}) if dataset.id in pinned_ids else dataset
        for dataset in snapshot.datasets
    ]
    for dataset in snapshot.datasets:
        source = reference_data.get(dataset.identifier or '')
        if source is not None and source.is_external_placeholder:
            snapshot.datasets = [
                item.model_copy(
                    update={'external_ref': source.external_ref, 'is_external_placeholder': True, 'revision_id': None}
                )
                if item.id == dataset.id
                else item
                for item in snapshot.datasets
            ]
    revision.content['model_snapshot']['structured'] = snapshot.model_dump(mode='json')
    revision.save(update_fields=['content'])
    InstanceRevisionDatasetPin.objects.bulk_create([
        InstanceRevisionDatasetPin(
            instance_config=template,
            instance_revision=revision,
            dataset=dataset,
            dataset_revision=revisions[dataset.pk],
            dataset_uuid=dataset.uuid,
            identifier=dataset.identifier,
            forecast_from=materializations[dataset.pk].forecast_from,
            shape_profiles=materializations[dataset.pk].shape_profiles,
        )
        for dataset in datasets
    ])
    return revision


def lock_template_for_publication(instance_id: int) -> None:
    """Acquire template before dependent instance locks, matching template publication's lock order."""

    selected = InstanceConfig.objects.select_related('template_revision').get(pk=instance_id)
    if selected.template_revision is not None:
        InstanceConfig.objects.select_for_update().get(pk=selected.template_revision.object_id)
