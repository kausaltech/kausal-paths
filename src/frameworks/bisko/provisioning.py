"""Explicit, repeatable provisioning of framework catalogues."""

from decimal import Decimal
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.db import transaction

from kausal_common.datasets.models import Dataset

from datasets.materialization import refresh_dataset_materialization
from datasets.validation import dump_violations, evaluate_dataset_rules
from datasets.year_slots import ensure_empty_year
from frameworks.bisko.activation import municipality_nuts3
from frameworks.bisko.plausibility import provision_bisko_plausibility_ranges
from frameworks.bisko.quality import provision_bisko_quality_projections
from frameworks.bisko.weather import (
    WEATHER_DATASET,
    load_weather_source,
    mark_seeded_weather_defaults,
    seed_weather_defaults,
)
from frameworks.identity import ensure_municipal_organization
from frameworks.models import DataQualityLevel, DataQualityScheme, Framework, FrameworkConfig
from nodes.defs.transform_def import FilterColumnOp
from nodes.models import DatasetMaterialization, InstanceConfig, NodeInputPortBinding
from nodes.template_graph import template_snapshot
from orgs.models import Namespace, OrganizationClass, OrganizationIdentifier
from params.param import StringParameter

if TYPE_CHECKING:
    from wagtail.models import Revision

    from nodes.defs.graph import QualityLevelKey
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
    ('nuts3', 'NUTS-3-Region'),
)


def _check_nuts3_identifiers() -> None:
    districts = OrganizationIdentifier.objects.filter(
        namespace__identifier='ars',
        organization__classification__identifier__in=('de_district', 'de_district_free_city'),
    )
    missing = districts.exclude(organization__identifiers__namespace__identifier='nuts3')
    codes = list(missing.values_list('identifier', flat=True)[:6])
    if codes:
        raise ValueError(
            f'{missing.count()} German districts lack NUTS3 identifiers (ARS: {", ".join(codes)}). '
            'Regenerate the BKG Parquet with the updated kausal-importers and rerun import_bkg_organizations.'
        )


def _reconcile_bisko_nuts_codes(framework: Framework) -> None:
    configs = FrameworkConfig.objects.filter(
        framework=framework,
        instance_config__organization__classification__identifier='de_municipality',
    ).select_related('instance_config__organization')
    for config in configs:
        instance = config.instance_config
        spec = instance.spec
        if spec is None:
            raise ValueError(f'{instance.identifier} has no stored instance spec.')
        organization = instance.organization
        assert organization is not None
        nuts3 = municipality_nuts3(organization)
        if instance.template_revision_id is not None and not any(p.local_id == 'nuts_code' for p in spec.params):
            base = template_snapshot(instance)
            default = next((scenario for scenario in base.spec.scenarios if scenario.default), None)
            override = spec.local_scenario(default.id if default is not None else 'default')
            if override.param_values.get('nuts_code') == nuts3:
                continue
            override.param_values['nuts_code'] = nuts3
            override.parameter_types['nuts_code'] = 'string'
        else:
            parameter = next((param for param in spec.params if param.local_id == 'nuts_code'), None)
            if parameter is None:
                spec.params.append(StringParameter(local_id='nuts_code', label='NUTS-3 code', value=nuts3))
            elif isinstance(parameter, StringParameter):
                if parameter.value == nuts3:
                    continue
                parameter.set(nuts3, notify=False)
            else:
                raise ValueError(f'{instance.identifier} has a non-string nuts_code parameter.')
        instance.spec = spec
        instance.save(update_fields=['spec'])
        instance.invalidate_cache()


def _reconcile_bisko_weather_defaults(framework: Framework) -> None:
    template = framework.template_instance
    assert template is not None
    if not Dataset.objects.for_instance_config(template).filter(identifier=WEATHER_DATASET).exists():
        return
    configs = FrameworkConfig.objects.filter(
        framework=framework,
        instance_config__template_revision__isnull=False,
        instance_config__organization__classification__identifier='de_municipality',
    ).select_related('instance_config__organization')
    empty: list[tuple[InstanceConfig, Dataset]] = []
    for config in configs:
        instance = config.instance_config
        dataset = Dataset.objects.for_instance_config(instance).filter(identifier=WEATHER_DATASET).first()
        if dataset is None:
            raise ValueError(f'{instance.identifier} has no municipal weather dataset; reconcile its local inputs first.')
        if not dataset.data_points.exists():
            empty.append((instance, dataset))
        else:
            mark_seeded_weather_defaults(dataset)
    if not empty:
        return
    frame, revision = load_weather_source(framework)
    for instance, dataset in empty:
        organization = instance.organization
        assert organization is not None
        seed_weather_defaults(instance, dataset, frame, nuts3=municipality_nuts3(organization), source_revision=revision)


def _reconcile_bisko_empty_cells(framework: Framework) -> None:
    template = framework.template_instance
    assert template is not None
    sources = {
        dataset.identifier: dataset
        for dataset in Dataset.objects.for_instance_config(template).filter(identifier__startswith='kommune/')
        if dataset.identifier != WEATHER_DATASET
    }
    configs = FrameworkConfig.objects.filter(
        framework=framework,
        instance_config__template_revision__isnull=False,
        instance_config__organization__classification__identifier__in=('de_municipality', 'de_district_free_city'),
    ).select_related('instance_config')
    for config in configs:
        instance = config.instance_config
        year = instance.ensure_spec().years.max_historical
        if year is None:
            continue
        for dataset in Dataset.objects.for_instance_config(instance).filter(identifier__in=sources):
            assert dataset.identifier is not None
            if ensure_empty_year(dataset, year, prototype=sources[dataset.identifier]):
                continue
            materialization = DatasetMaterialization.objects.filter(dataset=dataset).first()
            if materialization is None or not any(
                violation.get('kind') == 'invalid_rule' for violation in materialization.validation_violations
            ):
                continue
            if dump_violations(evaluate_dataset_rules(dataset)) != materialization.validation_violations:
                refresh_dataset_materialization(dataset, touch=False)


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
    _check_nuts3_identifiers()
    _reconcile_bisko_nuts_codes(framework)
    _reconcile_bisko_weather_defaults(framework)
    _reconcile_bisko_empty_cells(framework)

    scheme, _ = DataQualityScheme.objects.get_or_create(
        framework=framework,
        identifier='quality',
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
    provision_bisko_plausibility_ranges(framework)
    return framework


@transaction.atomic
def reconcile_bisko_default_quality(
    framework: Framework,
    defaults: dict[str, QualityLevelKey],
    *,
    publish: bool = True,
    check_problems: bool = False,
    ignore_problems: bool = False,
) -> Revision | None:
    """Reconcile declared grades and release them once to template-dependent drafts."""
    from frameworks.evidence import DEFAULT_QUALITY_SPEC_KEY
    from nodes.instance_serialization import InstanceSnapshot, build_instance_snapshot
    from nodes.template_graph import publish_template_instance

    template = framework.template_instance
    if framework.identifier != 'bisko' or template is None:
        raise ValueError('Expected a BISKO framework with a template instance.')
    template = InstanceConfig.objects.select_for_update().get(pk=template.pk)
    datasets = list(Dataset.objects.for_instance_config(template).filter(identifier__in=defaults).select_for_update())
    expected = {}
    for dataset in datasets:
        assert dataset.identifier is not None
        default = defaults[dataset.identifier]
        expected[dataset.uuid] = default
        spec = dict(dataset.spec or {})
        declared = default.model_dump()
        if spec.get(DEFAULT_QUALITY_SPEC_KEY) != declared:
            spec[DEFAULT_QUALITY_SPEC_KEY] = declared
            dataset.spec = spec
            dataset.save(update_fields=['spec'])
    revision = template.live_revision
    if not publish or not expected:
        return revision
    # Unused declarations do not belong to the release; avoid republishing for them.
    current_ids = {d.id for d in build_instance_snapshot(template).all_datasets()}
    expected = {key: value for key, value in expected.items() if key in current_ids}
    if not expected:
        return revision
    released = {}
    if revision is not None:
        snapshot = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'])
        released = {d.id: d.default_quality for d in snapshot.all_datasets()}
    if revision is None or any(released.get(key) != value for key, value in expected.items()):
        if check_problems:
            from frameworks.bisko.validation import publish_checked_template

            return publish_checked_template(template, ignore_problems=ignore_problems)
        return publish_template_instance(template)
    return revision
