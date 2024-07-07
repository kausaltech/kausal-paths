from __future__ import annotations

import graphene

from kausal_common.graphene import DjangoNode, GQLInfo
from kausal_common.models.general import public_fields
from paths.graphql_types import resolve_unit

from .models import Framework, MeasureTemplate, MeasureTemplateDefaultDataPoint, Section


class MeasureTemplateDefaultDataPointType(DjangoNode):
    class Meta:
        model = MeasureTemplateDefaultDataPoint
        fields = public_fields(MeasureTemplateDefaultDataPoint)


class MeasureTemplateType(DjangoNode):
    default_data_points = graphene.List(graphene.NonNull(MeasureTemplateDefaultDataPointType), required=True)

    class Meta:
        model = MeasureTemplate
        fields = public_fields(MeasureTemplate)

    @staticmethod
    def resolve_unit(root: MeasureTemplate, info: GQLInfo):
        return resolve_unit(root.unit, info)


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

    @staticmethod
    def resolve_sections(root: Framework, info: GQLInfo):
        return root.sections.all()

    @staticmethod
    def resolve_section(root: Framework, info: GQLInfo, identifier: str):
        return root.sections.filter(identifier=identifier).first()

    @staticmethod
    def resolve_measure_template(root: Framework, info: GQLInfo, id: str):
        return MeasureTemplate.objects.filter(section__framework=root, id=id).first()


class Query(graphene.ObjectType):
    frameworks = graphene.List(graphene.NonNull(FrameworkType))
    framework = graphene.Field(FrameworkType, identifier=graphene.ID(required=True))

    def resolve_frameworks(self, info: GQLInfo):
        return Framework.objects.all()

    def resolve_framework(self, info: GQLInfo, identifier: str):
        return Framework.objects.get(identifier=identifier)


schema = graphene.Schema(query=Query)
