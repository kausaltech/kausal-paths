from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

import graphene
import strawberry as sb
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import CharField, Q
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from graphql import GraphQLError

import polars as pl
import sentry_sdk

from kausal_common.graphene import DjangoNode, DjangoNodeMeta
from kausal_common.models.general import public_fields
from kausal_common.models.uuid import UUID_PATTERN, query_pk_or_uuid, query_pk_or_uuid_or_identifier
from kausal_common.strawberry.registry import register_strawberry_type

from paths.graphql_types import resolve_unit

from nodes.constants import YEAR_COLUMN
from nodes.exceptions import NodeComputationError
from nodes.models import InstanceConfig

from .models import (
    Framework,
    FrameworkConfig,
    FrameworkDefaults,
    Measure,
    MeasureDataPoint,
    MeasureTemplate,
    MeasureTemplateDefaultDataPoint,
    MinMaxDefaultInt as MinMaxDefaultIntModel,
    Section,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from paths.types import PathsGQLInfo as GQLInfo

    from nodes.instance import Instance
    from nodes.units import Unit


strawberry = sb

class MeasureTemplateDefaultDataPointType(DjangoNode):
    class Meta(DjangoNodeMeta):
        model = MeasureTemplateDefaultDataPoint
        fields = public_fields(MeasureTemplateDefaultDataPoint)


class MeasureTemplateType(DjangoNode):
    default_data_points = graphene.List(graphene.NonNull(MeasureTemplateDefaultDataPointType), required=True)
    measure = graphene.Field(lambda: MeasureType, framework_config_id=graphene.ID(required=True), required=False)

    class Meta(DjangoNodeMeta):
        model = MeasureTemplate
        fields = public_fields(MeasureTemplate)

    @staticmethod
    def resolve_unit(root: MeasureTemplate, info: GQLInfo) -> Unit:
        return resolve_unit(root.unit, info)

    @staticmethod
    def resolve_default_data_points(root: MeasureTemplate, info: GQLInfo) -> list[MeasureTemplateDefaultDataPoint]:
        return root.cache.measure_template_default_data_points.by_measure_template(root.pk)

    @staticmethod
    def resolve_measure(root: MeasureTemplate, info: GQLInfo, framework_config_id: str) -> Measure | None:
        try:
            fwc_id = int(framework_config_id)
        except Exception:
            raise GraphQLError("Invalid ID", nodes=info.field_nodes) from None
        fwc = root.cache.framework_configs.get(fwc_id)
        if fwc is None:
            return None
        fwc.cache.measure_datapoints.full_populate()
        measures = fwc.cache.measures.by_measure_template(root.pk)
        if len(measures) == 0:
            return None
        return measures[0]


class SectionType(DjangoNode):
    class Meta:
        model = Section
        fields = public_fields(Section)

    parent = graphene.Field(lambda: SectionType)
    children = graphene.List(graphene.NonNull(lambda: SectionType), required=True)
    descendants = graphene.List(graphene.NonNull(lambda: SectionType), required=True)
    measure_templates = graphene.List(graphene.NonNull(MeasureTemplateType), required=True)

    @staticmethod
    def resolve_measure_templates(root: Section, info: GQLInfo) -> list[MeasureTemplate]:
        return root.cache.fw_cache.measure_templates.by_section(root.pk)

    @staticmethod
    def resolve_parent(root: Section, info: GQLInfo) -> Section | None:
        return root.cache.get_parent()

    @staticmethod
    def resolve_children(root: Section, info: GQLInfo) -> Iterable[Section]:
        def is_child(obj: Section) -> bool:
            return obj.cache.parent_id == root.pk
        objs = root.cache.fw_cache.sections.get_list(is_child)
        objs = sorted(objs, key=lambda s: s.path)
        return objs

    @staticmethod
    def resolve_descendants(root: Section, info: GQLInfo) -> Iterable[Section]:
        def is_descendant(obj: Section) -> bool:
            if obj.pk is root.pk:
                return False
            if not obj.path.startswith(root.path):
                return False
            if obj.depth < root.depth:
                return False
            return True
        objs = root.cache.fw_cache.sections.get_list(is_descendant)
        objs = sorted(objs, key=lambda s: s.path)
        return objs


@register_strawberry_type
@sb.experimental.pydantic.type(model=MinMaxDefaultIntModel)
class MinMaxDefaultIntType:
    min: strawberry.auto
    max: strawberry.auto
    default: strawberry.auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=FrameworkDefaults, fields=['target_year', 'baseline_year'])
class FrameworkDefaultsType:
    target_year: strawberry.auto
    baseline_year: strawberry.auto


class FrameworkType(DjangoNode):
    class Meta(DjangoNodeMeta):
        model = Framework
        fields = public_fields(Framework)

    sections = graphene.List(graphene.NonNull(SectionType), required=True)
    section = graphene.Field(SectionType, identifier=graphene.ID(required=True))
    measure_template = graphene.Field(MeasureTemplateType, id=graphene.ID(required=True))
    configs = graphene.List(graphene.NonNull(lambda: FrameworkConfigType), required=True)
    config = graphene.Field(lambda: FrameworkConfigType, id=graphene.ID(required=True), required=False)
    defaults = graphene.Field(FrameworkDefaultsType, required=True)

    @staticmethod
    def resolve_sections(root: Framework, info: GQLInfo) -> list[Section]:
        return root.cache.sections.get_list()

    @staticmethod
    def resolve_section(root: Framework, info: GQLInfo, identifier: str) -> Section | None:
        return root.cache.sections.first(query_pk_or_uuid_or_identifier(identifier))

    @staticmethod
    def resolve_measure_template(root: Framework, info: GQLInfo, id: str) -> MeasureTemplate | None:
        return root.cache.measure_templates.first(query_pk_or_uuid(id))

    @staticmethod
    def resolve_configs(root: Framework, info: GQLInfo) -> list[FrameworkConfig]:
        return root.cache.framework_configs.get_list()

    @staticmethod
    def resolve_config(root: Framework, info: GQLInfo, id: str) -> FrameworkConfig | None:
        fwc = root.cache.framework_configs.first(get_fwc_q(id))
        if fwc is None:
            return None
        fwc.ensure_gql_action_allowed(info, 'view')
        return fwc


class PlaceHolderDataPoint(graphene.ObjectType):
    value = graphene.Float()
    year = graphene.Int()


class MeasureType(DjangoNode):
    measure_template = graphene.Field(MeasureTemplateType, required=True)
    placeholder_data_points = graphene.List(PlaceHolderDataPoint)

    class Meta:
        model = Measure
        fields = public_fields(Measure)

    @staticmethod
    def resolve_measure_template(root: Measure, info: GQLInfo) -> MeasureTemplate:
        mt = root.cache.fw_cache.measure_templates.get(root.measure_template_id)
        assert mt is not None
        return mt

    @staticmethod
    def resolve_data_points(root: Measure, info: GQLInfo) -> list[MeasureDataPoint]:
        return root.cache.measure_datapoints.by_measure(root.pk)

    @staticmethod
    def resolve_placeholder_data_points(root: Measure, info: GQLInfo) -> list[PlaceHolderDataPoint]:
        measure_template_uuid = str(root.measure_template.uuid)
        fwc = root.cache.framework_config
        if hasattr(root.cache, 'instance'):
            instance = root.cache.instance
        else:
            instance = fwc.instance_config.get_instance()
            root.cache.instance = instance
        context = instance.context
        node_dimension_selection = fwc.measure_template_uuid_to_node_dimension_selection.get(measure_template_uuid)
        if node_dimension_selection is None:
            return []
        node_id = node_dimension_selection.node_id
        node = context.get_node(node_id)
        with context.get_default_scenario().override():
            try:
                df = node.get_output_pl()
            except NodeComputationError as e:
                sentry_sdk.capture_exception(e)
                return []
        df = df.filter((pl.col(YEAR_COLUMN) > fwc.baseline_year) & (pl.col(YEAR_COLUMN) <= timezone.now().year))
        dimensions = node_dimension_selection.dimensions
        if dimensions:
            df = df.filter(**dimensions)
        result = []
        for d in df.select(pl.col('Year'), pl.col('Value')).to_dicts():
            year = d['Year']
            value = d['Value']
            result.append(PlaceHolderDataPoint(year=year, value=value))
        return result

class FrameworkConfigType(DjangoNode):
    measures = graphene.List(graphene.NonNull(MeasureType), required=True)
    view_url = graphene.String(description=_("Public URL for instance dashboard"), required=False)
    results_download_url = graphene.String(description=_("URL for downloading a results file"))
    instance = graphene.Field('nodes.schema.InstanceType', required=False)
    organization_slug = graphene.String(required=False)
    organization_identifier = graphene.String(required=False)

    class Meta(DjangoNodeMeta):
        model = FrameworkConfig
        fields = public_fields(FrameworkConfig)

    @staticmethod
    def resolve_framework(root: FrameworkConfig, info: GQLInfo) -> Framework:
        return root.cache.fw_cache.framework

    @staticmethod
    def resolve_measures(root: FrameworkConfig, info: GQLInfo) -> list[Measure]:
        root.cache.fw_cache.measure_templates.full_populate()
        return root.cache.measures.get_list()

    @staticmethod
    def resolve_view_url(root: FrameworkConfig, info: GQLInfo) -> str | None:
        fw = root.cache.fw_cache.framework
        if not fw.public_base_fqdn:
            return None
        ic = root.cache.fw_cache.instance_configs.get(root.instance_config_id)
        assert ic is not None
        return 'https://%s.%s' % (ic.identifier, fw.public_base_fqdn)

    @staticmethod
    def resolve_results_download_url(root: FrameworkConfig, info: GQLInfo) -> str:
        req = info.context
        path = reverse('framework_config_results_download', kwargs=dict(fwc_id=root.pk, token=root.token))
        return req.build_absolute_uri(path)

    @staticmethod
    def resolve_instance(root: FrameworkConfig, info: GQLInfo) -> Instance:
        return root.instance_config.get_instance(node_refs=True)

    @staticmethod
    def resolve_organization_slug(root: FrameworkConfig, info: GQLInfo) -> str | None:
        if info.context.user.is_superuser:
            return root.organization_slug
        return None

    @staticmethod
    def resolve_organization_identifier(root: FrameworkConfig, info: GQLInfo) -> str | None:
        if info.context.user.is_superuser:
            return root.organization_identifier
        return None

    @staticmethod
    def resolve_target_year(root: FrameworkConfig, info: GQLInfo) -> int | None:
        if root.target_year is not None:
            return root.target_year
        return root.cache.fw_cache.framework.defaults.target_year.default


class MeasureDataPointType(DjangoNode):
    class Meta(DjangoNodeMeta):
        model = MeasureDataPoint
        fields = public_fields(MeasureDataPoint)


def get_fwc_q(fwc_id: str) -> Q:
    if fwc_id.isdigit():
        q = Q(id=fwc_id)
    else:
        try:
            uuid = UUID(fwc_id)
            q = Q(uuid=uuid)
        except ValueError:
            q = Q(instance_config__identifier=fwc_id)
    return q


def get_fwc(info: GQLInfo, fwc_id: str) -> FrameworkConfig:
    fwc = FrameworkConfig.objects.get_queryset().filter(get_fwc_q(fwc_id)).first()
    if fwc is None:
        raise GraphQLError("Framework config '%s' not found" % fwc_id, nodes=info.field_nodes)
    fw = info.context.cache.for_framework(fwc.framework)
    fwc_cached = fw.cache.framework_configs.get(fwc.pk) if fw is not None else None
    if fwc_cached is None:
        raise GraphQLError("Framework config '%s' not accessible" % fwc_id, nodes=info.field_nodes)
    return fwc_cached


class Query(graphene.ObjectType):
    frameworks = graphene.List(graphene.NonNull(FrameworkType))
    framework = graphene.Field(FrameworkType, identifier=graphene.ID(required=True))

    def resolve_frameworks(self, info: GQLInfo) -> list[Framework]:
        return info.context.cache.frameworks.get_list()

    def resolve_framework(self, info: GQLInfo, identifier: str) -> Framework | None:
        fw = Framework.objects.get_queryset().filter(query_pk_or_uuid_or_identifier(identifier)).first()
        if fw is None:
            return None
        return info.context.cache.for_framework(fw)



class FrameworkConfigInput(graphene.InputObjectType):
    framework_id = graphene.ID(required=True)
    instance_identifier = graphene.ID(required=True, description=_("Identifier for the model instance. Needs to be unique."))
    name = graphene.String(
        required=True,
        description=_('Name for the framework configuration instance. Typically the name of the organization.'),
    )
    baseline_year = graphene.Int(required=True)
    target_year = graphene.Int(
        required=False,
        description="Target year for model.",
    )
    uuid = graphene.UUID(
        required=False, description=_('UUID for the new framework config. If not set, will be generated automatically.'),
        default_value=None,
    )
    organization_name = graphene.String(
        required=False,
        description=_(
            "Name of the organization. If not set, it will be determined through the user's credentials, if possible.",
        ),
        default_value=None,
    )


class CreateFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        framework_id = graphene.ID(required=True)
        instance_identifier = graphene.ID(required=True, description=_("Identifier for the model instance. Needs to be unique."))
        name = graphene.String(
            required=True,
            description=_('Name for the framework configuration instance. Typically the name of the organization.'),
        )
        baseline_year = graphene.Int(required=True)
        uuid = graphene.UUID(
            required=False, description=_('UUID for the new framework config. If not set, will be generated automatically.'),
            default_value=None,
        )
        organization_name = graphene.String(
            required=False,
            description=_(
                "Name of the organization. If not set, it will be determined through the user's credentials, if possible.",
            ),
            default_value=None,
        )

    ok = graphene.Boolean(required=True)
    framework_config = graphene.Field(FrameworkConfigType, description=_("The created framework config instance"))

    @classmethod
    def _get_fw(cls, info: GQLInfo, framework_id: str) -> Framework:
        if framework_id.isdigit():
            q = Q(id=int(framework_id))
        else:
            q = Q(identifier=framework_id)
        framework = info.context.cache.frameworks.first(q)
        if framework is None:
            raise GraphQLError("Framework '%s' not found" % framework_id, info.field_nodes)
        return framework

    @classmethod
    def _create_fwc(
        cls,
        info: GQLInfo,
        framework: Framework,
        instance_identifier: str,
        name: str,
        baseline_year: int,
        target_year: int | None = None,
        uuid: str | None = None,
    ) -> FrameworkConfig:
        id_field = cast(CharField, InstanceConfig._meta.get_field('identifier'))
        try:
            id_field.run_validators(instance_identifier)
        except ValidationError:
            raise GraphQLError("Invalid instance identifier", nodes=info.field_nodes) from None

        if InstanceConfig.objects.filter(identifier=instance_identifier).exists():
            raise GraphQLError("Instance with identifier '%s' already exists" % instance_identifier, nodes=info.field_nodes)

        if framework.configs.filter(organization_name__iexact=name).exists():
            raise GraphQLError("Framework config with organization name '%s' already exists" % name, nodes=info.field_nodes)

        if not uuid:
            uuid = str(uuid4())

        fc = FrameworkConfig.create_instance(
            framework=framework,
            instance_identifier=instance_identifier,
            org_name=name,
            baseline_year=baseline_year,
            target_year=target_year,
            uuid=uuid,
            user=info.context.user,
        )
        fc_cached = framework.cache.framework_configs.get(fc.pk)
        assert fc_cached is not None
        return fc_cached

    @classmethod
    def create_framework_config(cls, info: GQLInfo, config_input: FrameworkConfigInput) -> CreateFrameworkConfigMutation:
        framework = cls._get_fw(info, str(config_input.framework_id))
        pp = FrameworkConfig.permission_policy()
        if not pp.gql_action_allowed(info, 'add', context=framework):
            raise GraphQLError("Permission denied", nodes=info.field_nodes)
        fc = cls._create_fwc(
            info=info,
            framework=framework,
            instance_identifier=str(config_input.instance_identifier),
            name=cast(str, config_input.name),
            baseline_year=cast(int, config_input.baseline_year),
            target_year=cast(int | None, config_input.target_year),
            uuid=cast(str | None, config_input.uuid),
        )
        return CreateFrameworkConfigMutation(ok=True, framework_config=fc)

    @staticmethod
    def mutate(
        root, info: GQLInfo, framework_id: str, instance_identifier: str, name: str, baseline_year: int, uuid: str | None = None,
    ) -> CreateFrameworkConfigMutation:
        config = FrameworkConfigInput(
            framework_id=framework_id,
            instance_identifier=instance_identifier,
            name=name,
            baseline_year=baseline_year,
            uuid=uuid,
        )
        return CreateFrameworkConfigMutation.create_framework_config(info, config)

class UpdateFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        organization_name = graphene.String(required=False)
        organization_slug = graphene.String(required=False)
        organization_identifier = graphene.String(required=False)
        baseline_year = graphene.Int(
            required=False,
            description=(
                "New baseline year. Data point years will also be updated for measures that "
                "have exactly one data point which points to the previous baseline year."
            ),
        )
        target_year = graphene.Int(
            required=False,
            description="New target year for model.",
        )

    ok = graphene.Boolean()
    framework_config = graphene.Field(FrameworkConfigType)

    @staticmethod
    @transaction.atomic
    def mutate(
        root, info: GQLInfo, id: str, organization_name: str | None = None, organization_slug: str | None = None,
        organization_identifier: str | None = None, baseline_year: int | None = None, target_year: int | None = 0,
    ) -> UpdateFrameworkConfigMutation:
        fwc = get_fwc(info, id)
        fwc.ensure_gql_action_allowed(info, 'change')

        if organization_name is not None:
            fwc.organization_name = organization_name

        if organization_slug is not None or organization_identifier is not None:
            if not info.context.user.is_superuser:
                raise GraphQLError("Only superusers can set organization slug or identifier", nodes=info.field_nodes)
            if organization_slug is not None:
                fwc.organization_slug = organization_slug
            if organization_identifier is not None:
                fwc.organization_identifier = organization_identifier

        if baseline_year is not None:
            old_baseline_year = fwc.baseline_year
            fwc.baseline_year = baseline_year

            # Update datapoint years for measures with a single datapoint
            measures = fwc.measures.all()
            for measure in measures:
                datapoints = list(measure.data_points.all())
                if len(datapoints) == 1 and datapoints[0].year == old_baseline_year:
                    datapoint = datapoints[0]
                    datapoint.year = baseline_year
                    datapoint.save()

        # We use 0 to indicate that the target year was not supplied. `None` will
        # be interpreted as clearing the target year and using the default for the
        # framework.
        if target_year != 0:
            fwc.target_year = target_year

        fwc.notify_change(user=info.context.user)
        fwc.save()

        return UpdateFrameworkConfigMutation(ok=True, framework_config=fwc)


class DeleteFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True, description="ID (or UUID) of the framework config to be deleted")

    ok = graphene.Boolean()

    @staticmethod
    @transaction.atomic
    def mutate(root, info: GQLInfo, id: str) -> DeleteFrameworkConfigMutation:
        fwc = get_fwc(info, id)
        fwc.ensure_gql_action_allowed(info, 'delete')
        fwc.instance_config.delete()
        return DeleteFrameworkConfigMutation(ok=True)


class UpdateMeasureDataPoint(graphene.Mutation):
    class Arguments:
        framework_instance_id = graphene.ID(required=True, description=_("ID of the organization-specific framework instance"))
        measure_template_id = graphene.ID(required=True, description=_("ID of the measure template within a framework"))
        value = graphene.Float(required=False, description=_("Value for the data point (set to null to remove)"))
        year = graphene.Int(
            description=_(
                "Year of the data point. If not given, defaults to the baseline year for the framework instance."
            ),
            required=False,
        )
        internal_notes = graphene.String(description=_("Internal notes for the measure instance"), required=False)

    ok = graphene.Boolean()
    measure_data_point = graphene.Field(MeasureDataPointType, required=False)

    @transaction.atomic
    @staticmethod
    def mutate(
        root,
        info: GQLInfo,
        framework_instance_id: str,
        measure_template_id: str,
        value: float | None = None,
        year: int | None = None,
        internal_notes: str | None = None,
    ) -> UpdateMeasureDataPoint:
        fwc = get_fwc(info, framework_instance_id)
        fwc.ensure_gql_action_allowed(info, 'change')
        fw = fwc.framework
        measure_template = fw.measure_templates().filter(query_pk_or_uuid(measure_template_id)).first()
        if measure_template is None:
            raise GraphQLError("Measure template not found", nodes=info.field_nodes)

        if year is None:
            year = fwc.baseline_year

        measure = Measure.objects.filter(
            framework_config=fwc, measure_template=measure_template,
        ).first()

        if measure is None:
            measure = Measure(framework_config=fwc, measure_template=measure_template)
        if internal_notes is not None:
            measure.internal_notes = internal_notes
        measure.save()

        dp = measure.data_points.filter(year=year).first()
        if dp is None:
            dp = MeasureDataPoint(measure=measure, year=year)
        dp.value = value
        dp.save()

        fwc.notify_change(info.context.user, save=True)

        return UpdateMeasureDataPoint(ok=True, measure_data_point=dp)


class MeasureDataPointInput(graphene.InputObjectType):
    value = graphene.Float(required=False, description=_("Value for the data point (set to null to remove)"))
    year = graphene.Int(
        description=_("Year of the data point. If not given, defaults to the baseline year for the framework instance."),
        required=False,
    )


class MeasureInput(graphene.InputObjectType):
    measure_template_id = graphene.ID(required=True, description=_("ID (or UUID) of the measure template within a framework"))
    internal_notes = graphene.String(description=_("Internal notes for the measure instance"), required=False)
    data_points = graphene.List(graphene.NonNull(MeasureDataPointInput), required=False)


class UpdateMeasureDataPoints(graphene.Mutation):
    class Arguments:
        framework_config_id = graphene.ID(required=True)
        measures = graphene.List(graphene.NonNull(MeasureInput), required=True)

    ok = graphene.Boolean()
    created_data_points = graphene.List(MeasureDataPointType)
    updated_data_points = graphene.List(MeasureDataPointType)
    deleted_data_point_count = graphene.Int(required=True)

    @staticmethod
    @transaction.atomic
    def mutate(  # noqa: C901, PLR0912, PLR0915
        root, info: GQLInfo, framework_config_id: str, measures: list[dict[str, Any]],
    ) -> UpdateMeasureDataPoints:
        fwc = get_fwc(info, framework_config_id)
        fwc.ensure_gql_action_allowed(info, 'change')

        # Extract all measure template IDs
        mt_ids: set[str] = set()
        mt_uuids: set[str] = set()
        for m in measures:
            mt_id = m['measure_template_id']
            if mt_id.isnumeric():
                mt_ids.add(mt_id)
            elif UUID_PATTERN.match(mt_id):
                mt_uuids.add(mt_id)
            else:
                raise GraphQLError("Invalid ID: %s" % mt_id, nodes=info.field_nodes)

        # Fetch all referenced measure templates in a single query
        mt_qs = MeasureTemplate.objects.filter(Q(id__in=mt_ids) | Q(uuid__in=mt_uuids))
        mt_by_id = {str(mt.pk): mt for mt in mt_qs}
        mt_by_uuid = {str(mt.uuid): mt for mt in mt_qs}

        # Check if all referenced measure templates were found
        missing_ids = mt_ids - set(mt_by_id.keys())
        if missing_ids:
            msg = f"Measure templates not found: {', '.join(missing_ids)}"
            raise GraphQLError(msg)
        missing_uuids = mt_uuids - set(mt_by_uuid.keys())
        if missing_uuids:
            msg = f"Measure templates not found: {', '.join(missing_uuids)}"
            raise GraphQLError(msg)

        # Fetch all existing measures for this framework config and measure templates
        existing_measures = fwc.measures.filter(measure_template__in=mt_qs)
        m_by_mtid: dict[str, Measure] = {str(m.measure_template_id): m for m in existing_measures}

        created_data_points = []
        updated_data_points = []
        deleted_data_points = 0

        for m_in in measures:
            mt_id = m_in['measure_template_id']
            measure_template = mt_by_id.get(mt_id)
            if measure_template is None:
                measure_template = mt_by_uuid[mt_id]
            mt_id = str(measure_template.pk)

            measure = m_by_mtid.get(str(measure_template.pk))

            if measure is None:
                measure = Measure(
                    framework_config=fwc,
                    measure_template=measure_template,
                    internal_notes=m_in.get('internal_notes', ''),
                )
                m_by_mtid[mt_id] = measure
                measure.save()
            elif 'internal_notes' in m_in:
                measure.internal_notes = m_in.get('internal_notes', '')
                measure.save()

            dps_in: list[dict] = m_in.get('data_points', [])
            if not dps_in:
                continue
            for dp_input in dps_in:
                year = dp_input.get('year', fwc.baseline_year)
                value = dp_input['value']
                mdp = measure.data_points.filter(year=year).first()
                if mdp is None:
                    if value is None:
                        continue
                    mdp = MeasureDataPoint(measure=measure, year=year, value=value)
                    mdp.save()
                    created_data_points.append(mdp)
                elif mdp.value != value:
                    updated_data_points.append(mdp)
                    mdp.value = value
                    mdp.save()

        fwc.notify_change(info.context.user, save=True)

        return UpdateMeasureDataPoints(
            ok=True,
            created_data_points=created_data_points,
            updated_data_points=updated_data_points,
            deleted_data_point_count=deleted_data_points,
        )


class LowHigh(graphene.Enum):
    LOW = 0
    HIGH = 1


class NZCCityEssentialData(graphene.InputObjectType):
    population = graphene.Int(required=True, description="Population of the city")
    temperature = LowHigh(required=True, description="Average yearly temperature (low or high)")
    renewable_mix = LowHigh(required=True, description="Share of renewables in energy production (low or high)")


class CreateNZCFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        config_input = FrameworkConfigInput(required=True)
        nzc_data = NZCCityEssentialData(required=True)

    ok = graphene.Boolean(required=True)
    framework_config = graphene.Field(FrameworkConfigType, description=_("The created framework config instance"))

    @staticmethod
    def mutate(
        root,
        info: GQLInfo,
        config_input: FrameworkConfigInput,
        nzc_data: NZCCityEssentialData,
    ) -> CreateFrameworkConfigMutation:
        from .nzc import NZCPlaceholderInput, get_nzc_default_values

        def lowhigh_to_str(val: int) -> Literal['high', 'low']:
            if val == LowHigh.HIGH:
                return 'high'
            return 'low'

        ret = CreateFrameworkConfigMutation.create_framework_config(info, config_input)
        fwc = cast(FrameworkConfig, ret.framework_config)
        instance = fwc.instance_config.get_instance()
        dvc_repo = instance.context.dataset_repo
        data = cast(dict, nzc_data)
        defaults = get_nzc_default_values(dvc_repo, NZCPlaceholderInput(
            population=data['population'],
            renewmix=lowhigh_to_str(data['renewable_mix']),
            temperature=lowhigh_to_str(data['temperature']),
        ))
        fwc.create_measure_defaults(defaults)
        return ret


class Mutations(graphene.ObjectType):
    create_framework_config = CreateFrameworkConfigMutation.Field()
    create_nzc_framework_config = CreateNZCFrameworkConfigMutation.Field()
    update_framework_config = UpdateFrameworkConfigMutation.Field()
    delete_framework_config = DeleteFrameworkConfigMutation.Field()
    update_measure_data_point = UpdateMeasureDataPoint.Field()
    update_measure_data_points = UpdateMeasureDataPoints.Field()
