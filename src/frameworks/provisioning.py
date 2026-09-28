"""Explicit, repeatable provisioning of framework catalogues."""

from decimal import Decimal
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.db import transaction

from frameworks.identity import ensure_municipal_organization
from frameworks.models import DataQualityLevel, DataQualityScheme, Framework, FrameworkConfig
from frameworks.quality_provisioning import provision_bisko_quality_projections
from nodes.defs.transform_def import FilterColumnOp
from nodes.models import InstanceConfig, NodeInputPortBinding
from orgs.models import Namespace, OrganizationClass

if TYPE_CHECKING:
    from nodes.defs.instance_defs import InstanceModelSpec

# Catalogue version, not a claim about a particular certification protocol edition.
BISKO_QUALITY_VERSION = '1'
BISKO_QUALITY_LEVELS = (
    ('A', 'Regionale Primärdaten', Decimal(1)),
    ('B', 'Hochrechnung regionaler Primärdaten', Decimal('0.5')),
    ('C', 'Regionale Kennwerte und Statistiken', Decimal('0.25')),
    ('D', 'Bundesweite Kennzahlen', Decimal(0)),
)

GERMAN_ORGANIZATION_CLASSES = (
    ('de_state', 'Bundesland'),
    ('de_district', 'Landkreis'),
    ('de_district_free_city', 'Kreisfreie Stadt'),
    ('de_samtgemeinde', 'Samtgemeinde'),
    ('de_verbandsgemeinde', 'Verbandsgemeinde'),
    ('de_amt', 'Amt'),
    ('de_municipality', 'Gemeinde'),
)
GERMAN_IDENTIFIER_NAMESPACES = (
    ('ars', 'Amtlicher Regionalschlüssel'),
    ('ags', 'Amtlicher Gemeindeschlüssel'),
)


def _remove_passenger_spec_references(spec: InstanceModelSpec) -> bool:
    changed = False
    for result in spec.result_excels:
        if result.node_ids is not None and 'passenger_kilometers_own' in result.node_ids:
            result.node_ids.remove('passenger_kilometers_own')
            changed = True
    for scenario in spec.scenarios:
        if scenario.param_values.pop('passenger_kilometers_own.formula', None) is not None:
            changed = True
    return changed


def _remove_unused_template_inputs(template: InstanceConfig) -> None:
    """
    Retire the passenger-kilometres display input from the BISKO template.

    Keep its dataset: older published revisions can still refer to it, and
    deleting the node makes the unbound dataset disappear from new snapshots.
    """
    node = template.nodes.filter(identifier='passenger_kilometers_own').first()
    dependents = InstanceConfig.objects.filter(
        template_revision__content_type=ContentType.objects.get_for_model(InstanceConfig),
        template_revision__object_id=str(template.pk),
    )
    if node is not None and NodeInputPortBinding.objects.filter(source_node=node).exists():
        raise ValueError('passenger_kilometers_own feeds another template node; resolve that dependency first.')
    for instance in dependents:
        if node is not None:
            settings = [item for item in instance.node_settings if item.node_uuid != node.uuid]
            if len(settings) != len(instance.node_settings):
                instance.node_settings = settings
                instance.save(update_fields=['node_settings'])
                instance.invalidate_cache()
            instance.binding_overrides.filter(node_uuid=node.uuid).delete()
        instance_spec = instance.spec
        if instance_spec is not None and _remove_passenger_spec_references(instance_spec):
            instance.spec = instance_spec
            instance.save(update_fields=['spec'])
            instance.invalidate_cache()
    if node is not None:
        node.delete()
        template.invalidate_cache()

    spec = template.spec
    if spec is None:
        return
    changed = _remove_passenger_spec_references(spec)
    params = [param for param in spec.params if param.local_id != 'municipality_name']
    if len(params) != len(spec.params):
        spec.params = params
        changed = True
    if changed:
        template.spec = spec
        template.save(update_fields=['spec'])
        template.invalidate_cache()


def _remove_redundant_municipality_filters(template: InstanceConfig) -> None:
    """Remove label filters from the three national inputs already selected by AGS."""
    identifiers = (
        'vehicle_kilometers_ifeu',
        'other_transport_energy_ifeu',
        'other_transport_energy_availability',
    )
    bindings = NodeInputPortBinding.objects.filter(instance=template, node__identifier__in=identifiers)
    changed = False
    for binding in bindings:
        filters = [op for op in binding.transformations if isinstance(op, FilterColumnOp)]
        name_filters = [op for op in filters if op.column == 'municipality' and op.ref == 'municipality_name']
        ags_filter = next((op for op in filters if op.column == 'ags' and op.ref == 'ags_number'), None)
        if name_filters and ags_filter is None:
            raise ValueError(f'{binding.node.identifier} has a municipality filter without an AGS filter.')
        if ags_filter is None:
            continue
        if name_filters:
            binding.transformations = [
                op.model_copy(update={'ref': None}) if op in name_filters else op for op in binding.transformations
            ]
        elif not any(op.column == 'municipality' for op in filters):
            transforms = list(binding.transformations)
            transforms.insert(transforms.index(ags_filter) + 1, FilterColumnOp(column='municipality'))
            binding.transformations = transforms
        else:
            continue
        binding.save(update_fields=['transformations'])
        changed = True
    if changed:
        template.invalidate_cache()


@transaction.atomic
def prepare_bisko_template(framework: Framework) -> None:
    """Reconcile retired BISKO inputs before publishing a new template revision."""
    template = framework.template_instance
    if framework.identifier != 'bisko' or template is None:
        raise ValueError('Expected a BISKO framework with a template instance.')
    _remove_unused_template_inputs(template)
    _remove_redundant_municipality_filters(template)


def provision_german_organization_catalogue() -> None:
    """Seed stable identifiers used by the BKG administrative tree importer."""
    for identifier, name in GERMAN_ORGANIZATION_CLASSES:
        OrganizationClass.objects.get_or_create(identifier=identifier, defaults={'name': name})
    for identifier, name in GERMAN_IDENTIFIER_NAMESPACES:
        Namespace.objects.get_or_create(identifier=identifier, defaults={'name': name, 'user_editable': False})


@transaction.atomic
def setup_bisko(*, template_identifier: str = 'bisko', instance_identifiers: tuple[str, ...] = ()) -> Framework:  # noqa: C901
    """
    Register BISKO and its quality scale without cloning graphs or creating pages.

    Attached instances are moved to the organization identified by their AGS
    (see `ensure_municipal_organization`).

    Only explicitly named database-backed framework instances are attached. Framework
    membership changes authorization, and YAML membership also changes the model
    entrypoint; it is never inferred from an instance's name.
    """
    template = InstanceConfig.objects.get(identifier=template_identifier)
    instances = list(InstanceConfig.objects.filter(identifier__in=instance_identifiers).select_for_update())
    missing = set(instance_identifiers) - {instance.identifier for instance in instances}
    if missing:
        raise ValueError(f'Unknown instances: {", ".join(sorted(missing))}')
    for instance in instances:
        if instance.config_source != 'database':
            raise ValueError(
                f'{instance.identifier} is YAML-backed; attaching it would replace its model entrypoint with bisko.yaml. '
                'Migrate and verify the database-backed model before attaching it.'
            )

    framework, _ = Framework.objects.get_or_create(
        identifier='bisko',
        defaults={
            'name': 'BISKO',
            'description': 'Bilanzierungs-Systematik Kommunal',
            'template_instance': template,
            'use_instance_subdomains': False,
            'enable_user_management': True,
            'max_user_accounts_per_instance': 5,
        },
    )
    if framework.template_instance_id != template.pk:
        raise ValueError('BISKO already has a different template; resolve that explicitly before provisioning.')
    if framework.max_user_accounts_per_instance is None:
        framework.max_user_accounts_per_instance = 5
        framework.save(update_fields=['max_user_accounts_per_instance'])

    provision_german_organization_catalogue()

    scheme, _ = DataQualityScheme.objects.get_or_create(
        framework=framework,
        identifier='bisko',
        version=BISKO_QUALITY_VERSION,
        defaults={'name': 'BISKO Datengüte'},
    )
    unexpected = set(scheme.levels.values_list('identifier', flat=True)) - {row[0] for row in BISKO_QUALITY_LEVELS}
    if unexpected:
        raise ValueError(f'Unexpected grades in BISKO quality scheme: {", ".join(sorted(unexpected))}')
    for order, (identifier, name, score) in enumerate(BISKO_QUALITY_LEVELS):
        level, _ = DataQualityLevel.objects.get_or_create(
            scheme=scheme,
            identifier=identifier,
            defaults={'name': name, 'score': score, 'order': order},
        )
        if level.score != score:
            raise ValueError(f'BISKO grade {identifier} has score {level.score}, expected {score}; refusing to overwrite it.')

    for instance in instances:
        config, _ = FrameworkConfig.objects.get_or_create(
            instance_config=instance,
            defaults={'framework': framework, 'organization_name': instance.organization.name},
        )
        if config.framework_id != framework.pk:
            raise ValueError(f'{instance.identifier} already belongs to another framework.')
        instance.refresh_from_db()
        ensure_municipal_organization(instance)

    provision_bisko_quality_projections(framework)
    return framework
