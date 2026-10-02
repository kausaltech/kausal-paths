"""Compose a pinned framework method with a dependent instance's own graph and input selections."""

import hashlib
import json
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from wagtail.models import Revision

import networkx as nx

from kausal_common.datasets.models import Dataset
from kausal_common.i18n.pydantic import set_i18n_context

from datasets.catalogue import dataset_meta_from_model
from datasets.materialization import ensure_dataset_materializations
from nodes.constraints.validation import InstanceConstraintError, solve_instance_constraints
from nodes.instance_graph import NodeEditContext, NodeMeta, build_instance_graph
from nodes.instance_graph_cache import resolve_instance_source
from nodes.instance_serialization import (
    DatasetMetricSource,
    DatasetRevisionPinSnapshot,
    DefinitionOrigin,
    InputBindingOverrideSnapshot,
    InputBindingSnapshot,
    InstanceSnapshot,
    NodePortSource,
    NodeSnapshot,
    build_instance_snapshot,
)
from nodes.models import InputPortBindingSet, InstanceConfig, InstanceRevisionDatasetPin, PreferredInstanceSource
from nodes.template_reference_data import release_reference_materializations
from nodes.template_spec import (
    DECLARATION_LISTS,
    compose_instance_spec,
    declaration_identity,
    parameters_by_id,
    validate_spec_references,
)
from params.base import ParameterOwner
from params.param import ReferenceParameter, ValidationError as ParameterValidationError

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.defs.graph import DatasetMeta, DimensionMeta
    from nodes.template_settings import InheritedNodeSettings
    from params import Parameter
    from users.models import User


def snapshot_content_hash(snapshot: InstanceSnapshot) -> str:
    """Portable edition identity: structure and dataset content, independent of database revision IDs."""
    data = snapshot.model_dump(mode='json')
    data.pop('snapshot_kind', None)
    for pin in data['dataset_revisions']:
        pin.pop('revision_id', None)
    for dataset in [*data['datasets'], *(item for node in data['nodes'] for item in node['datasets'])]:
        dataset.pop('revision_id', None)
    for binding in data['bindings']:
        binding['source'].pop('dataset_revision', None)
    content = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(content).hexdigest()


def compose_template_snapshot(
    instance: InstanceConfig,
    local: InstanceSnapshot,
    *,
    dataset_revision_pins: dict[int, DatasetRevisionPinSnapshot] | None = None,
    compose: bool = True,
) -> InstanceSnapshot:
    """Capture local authoring inputs before optionally composing their runtime view."""
    revision = instance.template_revision
    if revision is None:
        return local
    base = template_snapshot(instance)
    local = local.model_copy(
        deep=True,
        update={
            'template_revision_id': revision.pk,
            'template_content_hash': snapshot_content_hash(base),
            'node_settings': [settings.model_copy(deep=True) for settings in instance.node_settings],
            'binding_overrides': [
                InputBindingOverrideSnapshot(node_uuid=item.node_uuid, port_uuid=item.port_uuid, bindings=item.bindings)
                for item in instance.binding_overrides.all()
            ],
        },
    )
    # Catalog explicit local selections, including datasets not bound by local ORM rows.
    selections = [*local.bindings, *(b for item in local.binding_overrides for b in item.bindings)]
    datasets, used = _effective_datasets(instance, base, local, selections)
    inherited = {item.id for item in base.all_datasets()}
    owned = {item.id for node in local.nodes for item in node.datasets}
    local.datasets = list(
        {
            **{item.id: item for item in local.datasets if item.id not in inherited},
            **{key: item for key, item in datasets.items() if key in used and key not in inherited and key not in owned},
        }.values()
    )
    local.dimensions = [item for item in local.dimensions if item.id not in {dimension.id for dimension in base.dimensions}]
    pins = {pin.dataset_uuid: pin for pin in (dataset_revision_pins or {}).values()}
    local.datasets = [
        item.model_copy(update={'revision_id': pins[item.id].revision_id}) if item.id in pins else item for item in local.datasets
    ]
    for override in local.binding_overrides:
        override.bindings = [
            binding.model_copy(
                update={
                    'source': binding.source.model_copy(
                        update={
                            'dataset_revision': pins[binding.source.dataset_uuid].revision_id,
                        }
                    )
                }
            )
            if isinstance(binding.source, DatasetMetricSource) and binding.source.dataset_uuid in pins
            else binding
            for binding in override.bindings
        ]
    local._template = base
    return local.resolve(base) if compose else local


def resolve_template_snapshot(local: InstanceSnapshot, *, template: InstanceSnapshot | None = None) -> InstanceSnapshot:
    """Pure composition of local authoring state and one immutable template snapshot."""
    base = template or local._template
    if base is None:
        revision = Revision.objects.get(pk=local.template_revision_id)
        if revision.content_type.pk != ContentType.objects.get_for_model(InstanceConfig).pk:
            raise ValueError('Template revision must belong to an instance')
        base = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'], compose=False)
    if base.template_revision_id is not None or base.metadata.uuid == local.metadata.uuid:
        raise ValueError('Nested or self template inheritance is not supported')
    if local.template_content_hash != snapshot_content_hash(base):
        raise ValueError('Pinned template content does not match the authoring snapshot')
    assert local.template_revision_id is not None
    nodes = _released_nodes(base, local.template_revision_id)
    shared_ids = {node.uuid for node in nodes}
    _apply_node_settings(local.node_settings, [*base.spec.params, *local.spec.params], {node.uuid: node for node in nodes})
    shared_names = {node.identifier for node in nodes}
    if any(node.uuid in shared_ids or node.identifier in shared_names for node in local.nodes):
        raise ValueError('Local nodes cannot shadow framework nodes')
    nodes.extend(node.model_copy(deep=True) for node in local.nodes)
    by_id = {node.uuid: node for node in nodes}
    bindings = _apply_binding_overrides(local.binding_overrides, by_id, shared_ids, [*base.bindings, *local.bindings])
    errors: list[str] = []
    bindings = _valid_binding_subset(by_id, bindings, errors)
    datasets = {item.id: item for item in [*base.all_datasets(), *local.all_datasets()]}
    used = {binding.source.dataset_uuid for binding in bindings if isinstance(binding.source, DatasetMetricSource)}
    if used - datasets.keys():
        raise ValueError('A framework binding references an unknown dataset')
    names = [dataset.identifier for key, dataset in datasets.items() if key in used and dataset.identifier is not None]
    if len(names) != len(set(names)):
        raise ValueError('Effective input datasets must have distinct identifiers')
    owned = {dataset.id for node in nodes for dataset in node.datasets}
    result = local.model_copy(
        update={
            'snapshot_kind': 'composed',
            'composition_errors': errors,
            'spec': compose_instance_spec(base.spec, local.spec, nodes, errors=errors),
            'nodes': nodes,
            'bindings': bindings,
            'dimensions': _compose_dimensions(base, local),
            'datasets': [dataset for key, dataset in datasets.items() if key in used and key not in owned],
            'dataset_revisions': list(
                {
                    pin.dataset_uuid: pin
                    for pin in [*base.dataset_revisions, *local.dataset_revisions]
                    if pin.dataset_uuid in used
                }.values()
            ),
        }
    )
    result._template = base
    result._provenance = definition_provenance(local, base, result)
    return result


def definition_provenance(  # noqa: C901, PLR0912
    local: InstanceSnapshot, base: InstanceSnapshot, effective: InstanceSnapshot
) -> dict[str, DefinitionOrigin]:
    """Track the source of declarations separately from the source of overridden values."""
    inherited = DefinitionOrigin(
        instance_uuid=base.metadata.uuid, revision_id=local.template_revision_id, content_hash=local.template_content_hash
    )
    own = DefinitionOrigin(instance_uuid=local.metadata.uuid)
    result: dict[str, DefinitionOrigin] = {'years': own, 'dataset_repo': inherited}
    for snapshot, origin in ((base, inherited), (local, own)):
        for node in snapshot.nodes:
            result[f'nodes/{node.uuid}'] = origin
        for field in (*DECLARATION_LISTS, 'params'):
            for declaration in getattr(snapshot.spec, field):
                identity = declaration.local_id if field == 'params' else declaration_identity(field, declaration)
                result[f'{field}/{identity}'] = origin
        for scenario in snapshot.spec.scenarios:
            result.setdefault(f'scenarios/{scenario.id}', origin)
            for identifier in scenario.param_values:
                result[f'scenarios/{scenario.id}/param_values/{identifier}'] = origin
        for binding in snapshot.bindings:
            result[f'bindings/{binding.uuid}'] = origin
    parameters = parameters_by_id(effective.spec, effective.nodes)
    inherited_scenarios = {item.id for item in base.spec.scenarios}
    for scenario in local.spec.scenarios:
        if scenario.id not in inherited_scenarios:
            continue
        for identifier in scenario.param_values:
            parameter = parameters.get(identifier)
            valid = parameter is not None and parameter.owner != ParameterOwner.FRAMEWORK
            if valid and parameter is not None:
                try:
                    parameter.clean(scenario.param_values[identifier])
                except ParameterValidationError:
                    valid = False
            if not valid or parameter is None or scenario.parameter_types.get(identifier, parameter.type) != parameter.type:
                key = f'scenarios/{scenario.id}/param_values/{identifier}'
                original = next(item for item in base.spec.scenarios if item.id == scenario.id)
                if identifier in original.param_values:
                    result[key] = inherited
                else:
                    result.pop(key, None)
    for node in effective.nodes:
        if node.spec is None:
            continue
        origin = result[f'nodes/{node.uuid}']
        for parameter in node.spec.params:
            result[f'params/{node.identifier}.{parameter.local_id}'] = origin
    default = next((scenario for scenario in effective.spec.scenarios if scenario.default), None)
    for identifier in parameters:
        origin = result[f'params/{identifier}']
        result[f'parameter_defaults/{identifier}'] = (
            result.get(f'scenarios/{default.id}/param_values/{identifier}', origin) if default is not None else origin
        )
    for field in ('theme_identifier', 'sample_size'):
        result[field] = own if field in local.spec.model_fields_set else inherited
    for field in ('features', 'terms'):
        for name in type(getattr(base.spec, field)).model_fields:
            result[f'{field}/{name}'] = own if name in getattr(local.spec, field).model_fields_set else inherited
    for settings in local.node_settings:
        for field in ('goals', 'layout', 'parameter_values', 'parameter_sources'):
            if getattr(settings, field):
                result[f'nodes/{settings.node_uuid}/{field}'] = own
    for override in local.binding_overrides:
        result[f'input_ports/{override.node_uuid}/{override.port_uuid}/bindings'] = own
    return result


def _effective_datasets(
    instance: InstanceConfig,
    base: InstanceSnapshot,
    local: InstanceSnapshot,
    bindings: list[InputBindingSnapshot],
) -> tuple[dict[UUID, DatasetMeta], set[UUID]]:
    datasets = {dataset.id: dataset for dataset in base.all_datasets()}
    datasets.update({dataset.id: dataset for dataset in local.all_datasets()})
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
    names = [dataset.identifier for key, dataset in datasets.items() if key in used and dataset.identifier is not None]
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
    """Publish a complete template without changing any dependent draft."""

    template = InstanceConfig.objects.select_for_update().get(pk=template.pk)
    if template.config_source != 'database' or template.template_revision_id is not None:
        raise ValueError('A template must be a standalone database-backed instance')
    with set_i18n_context(template.primary_language, template.other_languages):
        template.validate_draft_constraints()
        revision = _freeze_template_revision(template, user, reference_data or {})
    template.publish(revision, user=user)
    template.invalidate_cache()
    return revision


@transaction.atomic
def upgrade_template_instance(instance: InstanceConfig, revision: Revision, *, user: User | None = None) -> None:  # noqa: C901
    """Advance a draft independently; obsolete settings are pruned, other conflicts await repair."""
    if user is not None and not instance.permission_policy().user_has_perm(user, 'change', instance):
        raise ValidationError('You do not have permission to upgrade this instance')
    instance = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
    if instance.is_locked:
        raise ValidationError('Instance is locked')
    if instance.config_source != 'database':
        raise ValidationError('Only database-backed instances can be upgraded')
    previous = template_snapshot(instance)
    previous_revision = instance.template_revision
    if previous_revision is None or str(revision.object_id) != str(previous_revision.object_id):
        raise ValidationError('Upgrade revision must belong to the same template')
    instance.template_revision = revision
    candidate = template_snapshot(instance)
    nodes = {node.uuid: node for node in candidate.nodes}
    instance.node_settings = [settings for settings in instance.node_settings if settings.node_uuid in nodes]
    for settings in instance.node_settings:
        node = nodes[settings.node_uuid]
        assert node.spec is not None
        old_node = next(node for node in previous.nodes if node.uuid == settings.node_uuid)
        assert old_node.spec is not None
        old_params = {p.local_id: p for p in old_node.spec.params}
        valid = {
            p.local_id
            for p in node.spec.params
            if p.owner != ParameterOwner.FRAMEWORK and p.local_id in old_params and p.type == old_params[p.local_id].type
        }
        settings.parameter_values = {key: value for key, value in settings.parameter_values.items() if key in valid}
        settings.parameter_sources = {key: value for key, value in settings.parameter_sources.items() if key in valid}
    local_nodes = build_local_nodes(instance)
    instance.binding_overrides.exclude(node_uuid__in=[*nodes, *(node.uuid for node in local_nodes)]).delete()
    for binding in instance.binding_overrides.all():
        node = nodes.get(binding.node_uuid)
        if node is not None and node.spec is not None:
            port = next((port for port in node.spec.input_ports if port.id == binding.port_uuid), None)
            if port is None or port.binding_owner != 'instance':
                binding.delete()
    spec = instance.ensure_spec()
    old_params = parameters_by_id(previous.spec, previous.nodes)
    new_params = parameters_by_id(candidate.spec, candidate.nodes)
    new_params.update(parameters_by_id(spec, local_nodes))
    # A local entry for an inherited scenario holds only values for it, so it goes with the scenario;
    # left behind, it would read as a new local scenario without a name.
    retired = {scenario.id for scenario in previous.spec.scenarios} - {scenario.id for scenario in candidate.spec.scenarios}
    spec.scenarios = [override for override in spec.scenarios if override.id not in retired]
    for override in spec.scenarios:
        for identifier in list(override.param_values):
            parameter = new_params.get(identifier)
            old = old_params.get(identifier)
            if parameter is None or parameter.owner == ParameterOwner.FRAMEWORK or (old and old.type != parameter.type):
                override.param_values.pop(identifier)
                override.parameter_types.pop(identifier, None)
    instance.spec = spec
    instance.save(update_fields=['template_revision', 'node_settings', 'spec'])
    instance.invalidate_cache()


def build_local_nodes(instance: InstanceConfig) -> list[NodeSnapshot]:
    return [NodeSnapshot.from_model(node) for node in instance.nodes.all()]


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
    baseline_snapshot = build_instance_snapshot(instance)
    before = build_instance_graph(baseline_snapshot)
    if bindings is None:
        config.binding_overrides.filter(node_uuid=node_uuid, port_uuid=port_uuid).delete()
    else:
        permitted_datasets = set(Dataset.objects.for_instance_config(instance).values_list('uuid', flat=True))
        # A local node may bind its own datasets; a template node owns none here.
        local_node = instance.nodes.filter(uuid=node_uuid).first()
        if local_node is not None:
            permitted_datasets.update(Dataset.objects.qs.for_node(local_node).values_list('uuid', flat=True))
        permitted_datasets.update(dataset.id for dataset in template_snapshot(instance).all_datasets())
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
        candidate_snapshot = build_instance_snapshot(instance)
        new_errors = set(candidate_snapshot.composition_errors) - set(baseline_snapshot.composition_errors)
        if new_errors:
            raise ValidationError(sorted(new_errors))
        after = build_instance_graph(candidate_snapshot)
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


def _valid_binding_subset(
    by_id: dict[UUID, NodeSnapshot],
    bindings: list[InputBindingSnapshot],
    errors: list[str],
) -> list[InputBindingSnapshot]:
    """Keep a draft inspectable while retaining invalid selections in its authoring rows."""
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    valid = []
    for binding in bindings:
        if binding.node_id not in by_id:
            errors.append(f'Binding {binding.uuid}: missing binding target {binding.node_id}')
            continue
        if isinstance(binding.source, NodePortSource):
            source = by_id.get(binding.source.node_id)
            if source is None or source.spec is None or not any(p.id == binding.source.port_id for p in source.spec.output_ports):
                errors.append(f'Binding {binding.uuid}: output outside the effective graph')
                continue
            graph.add_edge(source.uuid, binding.node_id)
        valid.append(binding)
    cyclic_nodes = {
        node: component
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1 or graph.has_edge(next(iter(component)), next(iter(component)))
        for node in component
    }
    result = []
    for binding in valid:
        source = binding.source
        if (
            isinstance(source, NodePortSource)
            and source.node_id in cyclic_nodes
            and binding.node_id in cyclic_nodes[source.node_id]
        ):
            errors.append(f'Binding {binding.uuid}: the composed framework graph contains a cycle')
        else:
            result.append(binding)
    return result


def _apply_node_settings(
    settings_list: list[InheritedNodeSettings], global_parameters: list[Parameter], shared_nodes: dict[UUID, NodeSnapshot]
) -> None:
    for settings in settings_list:
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
            if parameter is None or parameter.owner == ParameterOwner.FRAMEWORK:
                raise ValueError(f'Parameter {identifier} is controlled by the framework')
            if identifier in settings.parameter_sources:
                target = next((p for p in global_parameters if p.local_id == settings.parameter_sources[identifier]), None)
                if target is None or target.type != parameter.type:
                    raise ValueError('A parameter source must reference a compatible instance parameter')
                parameters[identifier] = ReferenceParameter(local_id=identifier, target_id=target.local_id, owner=parameter.owner)
            else:
                value = settings.parameter_values[identifier]
                parameter.set(value)
        node.spec.params = list(parameters.values())


def _apply_binding_overrides(
    overrides: list[InputBindingOverrideSnapshot],
    by_id: dict[UUID, NodeSnapshot],
    shared_ids: set[UUID],
    bindings: list[InputBindingSnapshot],
) -> list[InputBindingSnapshot]:
    for override in overrides:
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
    validate_spec_references(snapshot.spec, snapshot.nodes)
    datasets = list(
        Dataset.objects
        .select_for_update(of=('self',))
        .exclude(
            identifier__in=[key for key, ds in reference_data.items() if ds.is_external_placeholder],
        )
        .filter(
            Q(is_external_placeholder=False)
            | Q(identifier__in=[key for key, ds in reference_data.items() if not ds.is_external_placeholder]),
            uuid__in=[dataset.id for dataset in snapshot.all_datasets()],
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
