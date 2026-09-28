"""Activate a German municipality as one BISKO framework instance."""

from typing import TYPE_CHECKING
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType
from django.db import transaction

from kausal_common.datasets.models import Dataset

from datasets.materialization import refresh_dataset_materialization
from datasets.snapshot import metric_column_id
from frameworks import submissions
from frameworks.models import Framework, FrameworkConfig
from frameworks.organization_access import organization_is_in_framework
from nodes.instance_serialization import DatasetMetricSource, InputBindingSnapshot, InstanceSnapshot
from nodes.models import DatasetMaterialization, InputPortBindingSet, InstanceConfig
from nodes.template_graph import template_snapshot
from orgs.models import Organization
from params.param import StringParameter

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.defs.instance_defs import InstanceModelSpec
    from users.models import User


class ActivationError(ValueError):
    pass


def _municipal_spec(snapshot: InstanceSnapshot, framework: Framework, ags: str) -> InstanceModelSpec:
    spec = snapshot.spec.model_copy(deep=True)
    spec.features.enable_user_management = framework.enable_user_management
    for name, value in (('ags_number', ags), ('lau_code', f'DE_{ags}')):
        parameter = next((param for param in spec.params if param.local_id == name), None)
        if parameter is None:
            spec.params.append(StringParameter(local_id=name, label=name, value=value))
        elif isinstance(parameter, StringParameter):
            parameter.set(value, notify=False)
        else:
            raise ActivationError(f'BISKO parameter {name} is not a string parameter.')
    return spec


def _ensure_local_inputs(instance: InstanceConfig) -> None:
    """Give each municipality empty local data slots and replace only local input bindings."""
    base = template_snapshot(instance)
    framework = instance.framework_config.framework
    template = framework.template_instance
    assert template is not None
    sources = {
        dataset.identifier: dataset
        for dataset in Dataset.objects.for_instance_config(template).filter(identifier__startswith='kommune/')
        if dataset.identifier and dataset.identifier.startswith('kommune/')
    }
    required = {ds.identifier for ds in base.all_datasets() if ds.identifier and ds.identifier.startswith('kommune/')}
    if not required <= set(sources):
        raise ActivationError(f'Published municipal datasets are missing from the database: {sorted(required - sources.keys())}')
    content_type = ContentType.objects.get_for_model(instance)
    local: dict[str, Dataset] = {}
    for identifier in sorted(required):
        source = sources[identifier]
        if source.schema_id is None:
            raise ActivationError(f'{identifier} has no dataset schema.')
        dataset, _ = Dataset.objects.get_or_create(
            scope_content_type=content_type,
            scope_id=instance.pk,
            identifier=identifier,
            defaults={'schema': source.schema, 'spec': dict(source.spec or {})},
        )
        if dataset.schema_id != source.schema_id or dataset.schema is None:
            raise ActivationError(f'{identifier} uses a different municipal schema; reconcile it explicitly.')
        if not dataset.data_points.exists():
            materialization = DatasetMaterialization.objects.filter(dataset=dataset).first()
            data = materialization.content.get('data') if materialization is not None else None
            fields = {field['name']: field for field in data['schema']['fields']} if data is not None else {}
            if (
                data is None
                or any(field['type'] == 'any' for field in fields.values())
                or any(
                    fields.get(metric_column_id(metric), {}).get('unit') != metric.unit for metric in dataset.schema.metrics.all()
                )
            ):
                refresh_dataset_materialization(dataset, touch=False)
        local[identifier] = dataset

    _override_municipal_bindings(instance, base, local)
    instance.invalidate_cache()


def _override_municipal_bindings(instance: InstanceConfig, base: InstanceSnapshot, local: dict[str, Dataset]) -> None:
    groups: dict[tuple[UUID, UUID], list[InputBindingSnapshot]] = {}
    for binding in base.bindings:
        groups.setdefault((binding.node_id, binding.port_id), []).append(binding)
    for (node_id, port_id), bindings in groups.items():
        if not any(isinstance(binding.source, DatasetMetricSource) and binding.source.dataset in local for binding in bindings):
            continue
        if InputPortBindingSet.objects.filter(instance=instance, node_uuid=node_id, port_uuid=port_id).exists():
            continue
        replacements = []
        for position, binding in enumerate(bindings):
            source = binding.source
            if isinstance(source, DatasetMetricSource) and source.dataset in local:
                source = source.model_copy(update={'dataset_uuid': local[source.dataset].uuid, 'dataset_revision': None})
            replacements.append(binding.model_copy(deep=True, update={'uuid': uuid4(), 'position': position, 'source': source}))
        InputPortBindingSet.objects.create(
            instance=instance,
            node_uuid=node_id,
            port_uuid=port_id,
            bindings=replacements,
        )


def _reconcile_instance(framework: Framework, instance: InstanceConfig, ags: str, *, actor: User | None) -> None:
    if instance.config_source != 'database':
        raise ActivationError('BISKO municipality must have a database-backed instance.')
    base = template_snapshot(instance)
    spec = _municipal_spec(base, framework, ags)
    current = instance.ensure_spec()
    # Preserve local settings and years; repair the municipality identity and licence feature.
    current.features.enable_user_management = spec.features.enable_user_management
    for name in ('ags_number', 'lau_code'):
        value = next(param.value for param in spec.params if param.local_id == name)
        parameter = next((param for param in current.params if param.local_id == name), None)
        if parameter is None:
            current.params.append(StringParameter(local_id=name, label=name, value=value))
        elif isinstance(parameter, StringParameter):
            parameter.set(value, notify=False)
        else:
            raise ActivationError(f'BISKO parameter {name} is not a string parameter.')
    instance.spec = current
    instance.save(update_fields=['spec'])
    _ensure_local_inputs(instance)
    if not instance.submissions.exists():
        period = current.years.max_historical or current.years.reference
        if period is not None:
            submissions.create_submission(instance, period_start=period, user=actor)


def _municipality_ags(framework: Framework, organization: Organization) -> str:
    if framework.identifier != 'bisko':
        raise ActivationError('Only BISKO municipalities can be activated here.')
    classification = organization.classification
    if classification is None or classification.identifier not in ('de_municipality', 'de_district_free_city'):
        raise ActivationError('Only municipalities and district-free cities can be activated.')
    if not organization_is_in_framework(framework, organization):
        raise ActivationError('Organization is outside BISKO coverage.')
    identifiers = dict(organization.identifiers.values_list('namespace__identifier', 'identifier'))
    ags = identifiers.get('ags')
    if ags is None or len(ags) != 8 or not ags.isdigit():
        raise ActivationError('Municipality has no valid eight-digit AGS.')
    return ags


@transaction.atomic
def activate_bisko_municipality(
    framework: Framework, organization: Organization, *, actor: User | None = None
) -> tuple[FrameworkConfig, bool]:
    """Create one local instance pinned to the published BISKO method, or return it."""
    organization = Organization.objects.select_for_update().get(pk=organization.pk)
    ags = _municipality_ags(framework, organization)

    existing = list(
        FrameworkConfig.objects.filter(framework=framework, instance_config__organization=organization).select_related(
            'instance_config'
        )[:2]
    )
    if len(existing) > 1:
        raise ActivationError('Municipality has multiple BISKO instances; resolve them before activation.')
    if existing:
        if existing[0].instance_config.template_revision_id is None:
            raise ActivationError('Existing BISKO instance does not inherit a published template; convert it explicitly.')
        try:
            template_snapshot(existing[0].instance_config)
        except ValueError as error:
            raise ActivationError(f'Existing BISKO instance has an invalid template revision: {error}') from error
        _reconcile_instance(framework, existing[0].instance_config, ags, actor=actor)
        return existing[0], False

    template = framework.template_instance
    if template is None or template.live_revision is None:
        raise ActivationError('Publish the BISKO template before activating municipalities.')
    revision = template.live_revision
    snapshot = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'])
    if snapshot.spec.years.reference is None:
        raise ActivationError('Published BISKO template has no reference year.')

    identifier = f'bisko-{ags}'
    if InstanceConfig.objects.filter(identifier=identifier).exists():
        raise ActivationError(f'Instance identifier {identifier} is already in use.')
    if InstanceConfig.objects.filter(organization=organization).exists():
        raise ActivationError('Municipality already has another model instance; attach or convert it explicitly.')
    suffix = f' ({ags})'
    name = f'{organization.name[: 150 - len(suffix)]}{suffix}'
    instance = InstanceConfig.objects.create(
        name=name,
        identifier=identifier,
        owner=organization.name,
        organization=organization,
        primary_language='de',
        other_languages=[],
        config_source='database',
        spec=_municipal_spec(snapshot, framework, ags),
        template_revision=revision,
        created_by=actor,
        last_modified_by=actor,
    )
    config = FrameworkConfig.objects.create(
        framework=framework,
        instance_config=instance,
        uuid=instance.uuid,
        organization_name=organization.name,
        organization_identifier=ags,
        created_by=actor,
        last_modified_by=actor,
    )
    _reconcile_instance(framework, instance, ags, actor=actor)
    return config, True
