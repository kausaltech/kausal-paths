from __future__ import annotations

from django.db import transaction
import graphene
from graphql import GraphQLError

from kausal_common.graphene import DjangoNode, GQLInfo
from kausal_common.models.general import public_fields
from paths.graphql_types import resolve_unit
from django.utils.translation import gettext_lazy as _

from .models import Framework, FrameworkConfig, Measure, MeasureDataPoint, MeasureTemplate, MeasureTemplateDefaultDataPoint, Section


class MeasureTemplateDefaultDataPointType(DjangoNode):
    class Meta:
        model = MeasureTemplateDefaultDataPoint
        fields = public_fields(MeasureTemplateDefaultDataPoint)


class MeasureTemplateType(DjangoNode):
    default_data_points = graphene.List(graphene.NonNull(MeasureTemplateDefaultDataPointType), required=True)
    measure = graphene.Field(lambda: MeasureType, framework_config_id=graphene.ID(required=True), required=False)

    class Meta:
        model = MeasureTemplate
        fields = public_fields(MeasureTemplate)

    @staticmethod
    def resolve_unit(root: MeasureTemplate, info: GQLInfo):
        return resolve_unit(root.unit, info)

    @staticmethod
    def resolve_default_data_points(root: MeasureTemplate, info: GQLInfo):
        return root.default_data_points.all()

    @staticmethod
    def resolve_measure(root: MeasureTemplate, info: GQLInfo, framework_config_id: str):
        try:
            fwc_id = int(framework_config_id)
        except Exception:
            raise GraphQLError("Invalid ID", nodes=info.field_nodes)
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
    def resolve_measure_templates(root: Section, info: GQLInfo):
        return root.measure_templates.all()

    @staticmethod
    def resolve_parent(root: Section, info: GQLInfo):
        return root.get_parent()

    @staticmethod
    def resolve_children(root: Section, info: GQLInfo):
        return root.get_children()

    @staticmethod
    def resolve_descendants(root: Section, info: GQLInfo):
        return root.get_descendants()


class FrameworkType(DjangoNode):
    class Meta:
        model = Framework
        fields = public_fields(Framework)

    sections = graphene.List(graphene.NonNull(SectionType), required=True)
    section = graphene.Field(SectionType, identifier=graphene.ID(required=True))
    measure_template = graphene.Field(MeasureTemplateType, id=graphene.ID(required=True))
    configs = graphene.List(graphene.NonNull(lambda: FrameworkConfigType), required=True)
    config = graphene.Field(lambda: FrameworkConfigType, id=graphene.ID(required=True), required=False)

    @staticmethod
    def resolve_sections(root: Framework, info: GQLInfo):
        return root.sections.all()

    @staticmethod
    def resolve_section(root: Framework, info: GQLInfo, identifier: str):
        return root.sections.filter(identifier=identifier).first()

    @staticmethod
    def resolve_measure_template(root: Framework, info: GQLInfo, id: str):
        return MeasureTemplate.objects.filter(section__framework=root, id=id).first()

    @staticmethod
    def resolve_configs(root: Framework, info: GQLInfo):
        # FIXME: Permission filtering
        return root.configs.all()

    @staticmethod
    def resolve_config(root: Framework, info: GQLInfo, id: str):
        # FIXME: Permission filtering
        return root.configs.filter(id=id).first()


class MeasureType(DjangoNode):
    measure_template = graphene.Field(MeasureTemplateType, required=True)

    class Meta:
        model = Measure
        fields = public_fields(Measure)


class FrameworkConfigType(DjangoNode):
    measures = graphene.List(graphene.NonNull(MeasureType), required=True)

    class Meta:
        model = FrameworkConfig
        fields = public_fields(FrameworkConfig)


class MeasureDataPointType(DjangoNode):
    class Meta:
        model = MeasureDataPoint
        fields = public_fields(MeasureDataPoint)


class Query(graphene.ObjectType):
    frameworks = graphene.List(graphene.NonNull(FrameworkType))
    framework = graphene.Field(FrameworkType, identifier=graphene.ID(required=True))

    def resolve_frameworks(self, info: GQLInfo):
        return Framework.objects.all()

    def resolve_framework(self, info: GQLInfo, identifier: str):
        return Framework.objects.get(identifier=identifier)


class CreateFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        framework_id = graphene.ID(required=True)
        name = graphene.String(required=True, description=_("Name for the framework configuration instance. Typically the name of the organization."))
        baseline_year = graphene.Int(required=True)

    ok = graphene.Boolean()
    framework_config = graphene.Field(FrameworkConfigType, description=_("The created framework config instance."))

    def mutate(self, info: GQLInfo, framework_id: str, name: str, baseline_year: int):
        framework = Framework.objects.filter(identifier=framework_id).first()
        if framework is None:
            try:
                fw_id = int(framework_id)
            except Exception:
                raise GraphQLError("Invalid ID", nodes=info.field_nodes)
            framework = Framework.objects.filter(id=fw_id).first()
        if framework is None:
            raise GraphQLError("Framework '%s' not found" % framework_id, info.field_nodes)

        fc = FrameworkConfig.create_instance(framework, name, baseline_year)
        return dict(ok=True, framework_config=fc)


class DeleteFrameworkConfigMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    ok = graphene.Boolean()

    def mutate(self, info: GQLInfo, id: str):
        # FIXME: Permission checking
        fwc = FrameworkConfig.objects.filter(id=id).first()
        if fwc is None:
            raise GraphQLError("FrameworkConfig '%s' not found" % id, info.field_nodes)
        fwc.delete()

        return dict(ok=True)


class UpdateMeasureDataPoint(graphene.Mutation):
    class Arguments:
        framework_instance_id = graphene.ID(required=True, description=_("ID of the organization-specific framework instance"))
        measure_template_id = graphene.ID(required=True, description=_("ID of the measure template within a framework"))
        value = graphene.Float(required=True)
        year = graphene.Int(
            description=_("Year of the data point. If not given, defaults to the baseline year for the framework instance"),
            required=False,
        )
        internal_notes = graphene.String(description=_("Internal notes for the measure instance"), required=False)

    ok = graphene.Boolean()
    measure_data_point = graphene.Field(MeasureDataPointType)

    @classmethod
    @transaction.atomic
    def mutate(cls, root, info: GQLInfo, framework_instance_id: str, measure_template_id: str, value: float, year: int | None = None, internal_notes: str | None = None):
        try:
            framework_config = FrameworkConfig.objects.get(id=framework_instance_id)
        except FrameworkConfig.DoesNotExist:
            raise GraphQLError("Framework instance not found", nodes=info.field_nodes)

        try:
            measure_template = MeasureTemplate.objects.get(id=measure_template_id)
        except MeasureTemplate.DoesNotExist:
            raise GraphQLError("Measure template not found", nodes=info.field_nodes)

        if year is None:
            year = framework_config.baseline_year

        measure = Measure.objects.filter(
            framework_config=framework_config, measure_template=measure_template
        ).first()
        if measure is None:
            measure = Measure(framework_config=framework_config, measure_template=measure_template)
        if internal_notes is not None:
            measure.internal_notes = internal_notes
        measure.save()

        dp = measure.data_points.filter(year=year).first()
        if dp is None:
            dp = MeasureDataPoint(measure=measure, year=year)
        dp.value = value
        dp.save()

        return UpdateMeasureDataPoint(ok=True, measure_data_point=dp)


class Mutations(graphene.ObjectType):
    create_framework_config = CreateFrameworkConfigMutation.Field()
    delete_framework_config = DeleteFrameworkConfigMutation.Field()
    update_measure_data_point = UpdateMeasureDataPoint.Field()
