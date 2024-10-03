from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

import graphene
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import CharField, Q
from django.db.models.query import QuerySet
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from graphql import GraphQLError

from kausal_common.graphene import DjangoNode, DjangoNodeMeta
from kausal_common.models.general import public_fields
from kausal_common.models.uuid import UUID_PATTERN, query_pk_or_uuid

from paths.graphql_types import resolve_unit

from nodes.models import InstanceConfig

from .models import (
    Framework,
    FrameworkConfig,
    FrameworkConfigQuerySet,
    Measure,
    MeasureDataPoint,
    MeasureTemplate,
    MeasureTemplateDefaultDataPoint,
    Section,
)

if TYPE_CHECKING:
    from django.db.models.query import QuerySet

    from kausal_common.graphene import GQLInfo
    from kausal_common.models.types import QS

    from frameworks.models import SectionQuerySet
    from nodes.instance import Instance
    from nodes.units import Unit


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
    def resolve_default_data_points(root: MeasureTemplate, info: GQLInfo) -> QS[MeasureTemplateDefaultDataPoint]:
        return root.default_data_points.all()

    @staticmethod
    def resolve_measure(root: MeasureTemplate, info: GQLInfo, framework_config_id: str) -> Measure | None:
        try:
            fwc_id = int(framework_config_id)
        except Exception:
            raise GraphQLError("Invalid ID", nodes=info.field_nodes) from None
        return root.measures.filter(framework_config=fwc_id).first()


class SectionType(DjangoNode):
    class Meta:
        model = Section
        fields = public_fields(Section)

    parent = graphene.Field(lambda: SectionType)
    children = graphene.List(graphene.NonNull(lambda: SectionType), required=True)
    descendants = graphene.List(graphene.NonNull(lambda: SectionType), required=True)
    measure_templates = graphene.List(graphene.NonNull(MeasureTemplateType), required=True)

    @staticmethod
    def resolve_measure_templates(root: Section, info: GQLInfo) -> QS[MeasureTemplate]:
        return root.measure_templates.all()

    @staticmethod
    def resolve_parent(root: Section, info: GQLInfo) -> Section | None:
        return root.get_parent()

    @staticmethod
    def resolve_children(root: Section, info: GQLInfo) -> SectionQuerySet:
        return root.get_children()

    @staticmethod
    def resolve_descendants(root: Section, info: GQLInfo) -> SectionQuerySet:
        return root.get_descendants()


class FrameworkType(DjangoNode):
    class Meta(DjangoNodeMeta):
        model = Framework
        fields = public_fields(Framework)

    sections = graphene.List(graphene.NonNull(SectionType), required=True)
    section = graphene.Field(SectionType, identifier=graphene.ID(required=True))
    measure_template = graphene.Field(MeasureTemplateType, id=graphene.ID(required=True))
    configs = graphene.List(graphene.NonNull(lambda: FrameworkConfigType), required=True)
    config = graphene.Field(lambda: FrameworkConfigType, id=graphene.ID(required=True), required=False)

    @staticmethod
    def resolve_sections(root: Framework, info: GQLInfo) -> QuerySet[Section, Section]:
        return root.sections.all()

    @staticmethod
    def resolve_section(root: Framework, info: GQLInfo, identifier: str) -> Section | None:
        return root.sections.filter(identifier=identifier).first()

    @staticmethod
    def resolve_measure_template(root: Framework, info: GQLInfo, id: str) -> MeasureTemplate | None:  # noqa: A002
        return MeasureTemplate.objects.filter(section__framework=root, id=id).first()

    @staticmethod
    def resolve_configs(root: Framework, info: GQLInfo) -> FrameworkConfigQuerySet:
        return root.configs.get_queryset().viewable_by(info.context.user)

    @staticmethod
    def resolve_config(root: Framework, info: GQLInfo, id: str) -> FrameworkConfig | None:  # noqa: A002
        fwc = root.configs.get_queryset().filter(get_fwc_q(id)).first()
        if fwc is None:
            return None
        fwc.ensure_gql_action_allowed(info, 'view')
        return fwc


class MeasureType(DjangoNode):
    measure_template = graphene.Field(MeasureTemplateType, required=True)

    class Meta:
        model = Measure
        fields = public_fields(Measure)


class FrameworkConfigType(DjangoNode):
    measures = graphene.List(graphene.NonNull(MeasureType), required=True)
    view_url = graphene.String(description=_("Public URL for instance dashboard"), required=False)
    results_download_url = graphene.String(description=_("URL for downloading a results file"))
    instance = graphene.Field('nodes.schema.InstanceType', required=False)

    class Meta(DjangoNodeMeta):
        model = FrameworkConfig
        fields = public_fields(FrameworkConfig)

    @staticmethod
    def resolve_measures(root: FrameworkConfig, info: GQLInfo) -> QuerySet[Measure, Measure]:
        return root.measures.get_queryset()

    @staticmethod
    def resolve_view_url(root: FrameworkConfig, info: GQLInfo) -> str | None:
        fw = root.framework
        if not fw.public_base_fqdn:
            return None
        return 'https://%s.%s' % (root.instance_config.identifier, fw.public_base_fqdn)

    @staticmethod
    def resolve_results_download_url(root: FrameworkConfig, info: GQLInfo) -> str:
        req = info.context
        path = reverse('framework_config_results_download', kwargs=dict(fwc_id=root.pk, token=root.token))
        return req.build_absolute_uri(path)

    @staticmethod
    def resolve_instance(root: FrameworkConfig, info: GQLInfo) -> Instance:
        return root.instance_config.get_instance(node_refs=True)


class MeasureDataPointType(DjangoNode):
    class Meta(DjangoNodeMeta):
        model = MeasureDataPoint
        fields = public_fields(MeasureDataPoint)


class Query(graphene.ObjectType):
    frameworks = graphene.List(graphene.NonNull(FrameworkType))
    framework = graphene.Field(FrameworkType, identifier=graphene.ID(required=True))

    def resolve_frameworks(self, info: GQLInfo):
        return Framework.objects.all()

    def resolve_framework(self, info: GQLInfo, identifier: str):
        return Framework.objects.get(identifier=identifier)


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


def get_fwc(info: GQLInfo, fwc_id: str, qs: FrameworkConfigQuerySet | None = None) -> FrameworkConfig:
    if qs is None:
        qs = FrameworkConfig.objects.get_queryset()
    fwc = qs.filter(get_fwc_q(fwc_id)).first()
    if fwc is None:
        raise GraphQLError("Framework config '%s' not found" % fwc_id, nodes=info.field_nodes)
    return fwc


class FrameworkConfigInput(graphene.InputObjectType):
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
    framework_config = graphene.Field(FrameworkConfigType, description=_("The created framework config instance."))

    @classmethod
    def _get_fw(cls, info: GQLInfo, framework_id: str) -> Framework:
        if framework_id.isdigit():
            q = Q(id=int(framework_id))
        else:
            q = Q(identifier=framework_id)
        framework = Framework.objects.filter(q).first()
        if framework is None:
            raise GraphQLError("Framework '%s' not found" % framework_id, info.field_nodes)
        return framework

    @classmethod
    def _create_fwc(  # noqa: PLR0913
        cls,
        info: GQLInfo,
        framework: Framework,
        instance_identifier: str,
        name: str,
        baseline_year: int,
        uuid: str | None = None,
    ) -> FrameworkConfig:
        id_field = cast(CharField, InstanceConfig._meta.get_field('identifier'))
        try:
            id_field.run_validators(instance_identifier)
        except ValidationError:
            raise GraphQLError("Invalid instance identifier", nodes=info.field_nodes) from None

        if InstanceConfig.objects.filter(identifier=instance_identifier).exists():
            raise GraphQLError("Instance with identifier '%s' already exists" % instance_identifier, nodes=info.field_nodes)

        if not uuid:
            uuid = str(uuid4())

        fc = FrameworkConfig.create_instance(
            framework=framework,
            instance_identifier=instance_identifier,
            org_name=name,
            baseline_year=baseline_year,
            uuid=uuid,
            user=info.context.user,
        )
        return fc

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
            uuid=cast(str | None, config_input.uuid),
        )
        return CreateFrameworkConfigMutation(ok=True, framework_config=fc)

    @staticmethod
    def mutate(  # noqa: PLR0913
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
        baseline_year = graphene.Int(
            required=False,
            description=(
                "New baseline year. Data point years will also be updated for measures that "
                "have exactly one data point which points to the previous baseline year."
            ),
        )

    ok = graphene.Boolean()
    framework_config = graphene.Field(FrameworkConfigType)

    @staticmethod
    @transaction.atomic
    def mutate(
        root, info: GQLInfo, id: str, organization_name: str | None = None, baseline_year: int | None = None,  # noqa: A002
    ) -> UpdateFrameworkConfigMutation:
        fwc = get_fwc(info, id)
        fwc.ensure_gql_action_allowed(info, 'change')

        if organization_name is not None:
            fwc.organization_name = organization_name

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

        fwc.notify_change(user=info.context.user)
        fwc.save()

        return UpdateFrameworkConfigMutation(ok=True, framework_config=fwc)


class DeleteFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True, description="ID (or UUID) of the framework config to be deleted")

    ok = graphene.Boolean()

    @staticmethod
    @transaction.atomic
    def mutate(root, info: GQLInfo, id: str) -> DeleteFrameworkConfigMutation:  # noqa: A002
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
            description=_("Year of the data point. If not given, defaults to the baseline year for the framework instance"),
            required=False,
        )
        internal_notes = graphene.String(description=_("Internal notes for the measure instance"), required=False)

    ok = graphene.Boolean()
    measure_data_point = graphene.Field(MeasureDataPointType, required=False)

    @transaction.atomic
    @staticmethod
    def mutate(  # noqa: PLR0913
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
        description=_("Year of the data point. If not given, defaults to the baseline year for the framework instance"),
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
    framework_config = graphene.Field(FrameworkConfigType, description=_("The created framework config instance."))

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
