"""Explicit, repeatable provisioning of framework catalogues."""

from decimal import Decimal

from django.db import transaction

from frameworks.models import DataQualityLevel, DataQualityScheme, Framework, FrameworkConfig
from nodes.models import InstanceConfig

# Catalogue version, not a claim about a particular certification protocol edition.
BISKO_QUALITY_VERSION = '1'
BISKO_QUALITY_LEVELS = (
    ('A', 'Regionale Primärdaten', Decimal(1)),
    ('B', 'Hochrechnung regionaler Primärdaten', Decimal('0.5')),
    ('C', 'Regionale Kennwerte und Statistiken', Decimal('0.25')),
    ('D', 'Bundesweite Kennzahlen', Decimal(0)),
)


@transaction.atomic
def setup_bisko(*, template_identifier: str = 'bisko', instance_identifiers: tuple[str, ...] = ()) -> Framework:
    """
    Register BISKO and its quality scale without cloning graphs or creating pages.

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
        },
    )
    if framework.template_instance_id != template.pk:
        raise ValueError('BISKO already has a different template; resolve that explicitly before provisioning.')

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

    return framework
