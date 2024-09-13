from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast
from uuid import UUID  # noqa: TCH003

from django.contrib.postgres.expressions import ArraySubquery
from django.db.models.expressions import F, OuterRef
from django.db.models.functions import JSONObject
from pydantic import Field

from kausal_common.models.django_pydantic import DjangoAdapter, DjangoDiffModel, JSONAdapter, TypedAdapter

from frameworks.models import Framework, MeasureTemplate, MeasureTemplateDefaultDataPoint, Section

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from diffsync import DiffSyncModel


class SectionModel(DjangoDiffModel[Section]):
    _model = Section
    _modelname = 'section'
    _identifiers = ('uuid',)
    _attributes = ('parent', 'identifier', 'name', 'description', 'available_years')
    _children = {
        'measure_template': 'measure_templates',
        'section': 'children',
    }

    parent: UUID | None
    children: list[UUID] = Field(default_factory=list)
    measure_templates: list[str] = Field(default_factory=list)

    @classmethod
    def get_queryset(cls, fw: Framework) -> QuerySet[Section, dict[str, Any]]:
        if fw.root_section is None:
            qs_base = cls._model.objects.get_queryset().none()
        else:
            qs_base = fw.root_section.get_descendants().order_by('path')
        sections = (
            qs_base
            .annotate_parent_field('parent', 'uuid', min_depth=2)
            .values(*cls._django_fields.plain_fields, 'parent', 'depth', _instance_pk=F('pk'))
        )
        return sections


class MeasureTemplateModel(DjangoDiffModel[MeasureTemplate]):
    _model = MeasureTemplate
    _identifiers = ('uuid',)
    _modelname = 'measure_template'
    _attributes = (
        'name', 'unit', 'priority', 'min_value', 'max_value', 'time_series_max', 'default_value_source',
        'default_data_points', 'section',
    )

    class DefaultDataPoint(TypedDict):
        year: int
        value: float

    section: UUID
    default_data_points: list[DefaultDataPoint] = Field(default_factory=list)
    _section_pk: int | None = None

    @classmethod
    def get_queryset(cls, fw: Framework) -> QuerySet[MeasureTemplate, dict[str, Any]]:
        mt_fields = cls._django_fields.field_names - {'section'}
        ddps = MeasureTemplateDefaultDataPoint.objects.filter(
            template_id=OuterRef('pk'),
        ).annotate(data=JSONObject(
            year=F('year'), value=F('value'),
        )).values_list('data')
        mt_objs = (
            MeasureTemplate.objects.filter(section__framework=fw)
                .values(*mt_fields)
                .annotate(_instance_pk=F('pk'))
                .annotate(section=F('section__uuid'))
                .annotate(_section_pk=F('section_id'))
                .annotate(default_data_points=ArraySubquery(ddps))
        )
        return mt_objs

    @classmethod
    def get_create_kwargs(cls, adapter: DjangoAdapter, ids: dict, attrs: dict) -> dict:
        kwargs = super().get_create_kwargs(adapter, ids, attrs)
        sec = cast(SectionModel, adapter.get(SectionModel, str(kwargs.pop('section'))))
        assert sec._instance_pk is not None
        kwargs['section_id'] = sec._instance_pk
        return kwargs

    @classmethod
    def create_related(cls, _adapter: DjangoAdapter, _ids: dict, attrs: dict, instance: MeasureTemplate, /) -> None:
        ddps = attrs.get('default_data_points')
        if not ddps:
            return
        ddp_objs = [MeasureTemplateDefaultDataPoint(template=instance, **ddp) for ddp in ddps]
        MeasureTemplateDefaultDataPoint.objects.bulk_create(ddp_objs)


class FrameworkModel(DjangoDiffModel[Framework]):
    _model = Framework
    _modelname = 'framework'
    _identifiers = ('identifier',)
    _attributes = ('name', 'description', 'public_base_fqdn', 'result_excel_url', 'result_excel_node_ids')
    _children = {'section': 'sections'}

    sections: list[str] = Field(default_factory=list)
    identifier: str


class FrameworkAdapter(TypedAdapter):
    framework = FrameworkModel
    section = SectionModel
    measure_template = MeasureTemplateModel
    top_level = ['framework']

    def add_child(self, parent: DiffSyncModel, child: DiffSyncModel):
        self.add(child)
        parent.add_child(child)


class FrameworkDjangoAdapter(FrameworkAdapter, DjangoAdapter):
    def load_sections(self, fw: Framework, fw_model: FrameworkModel):
        if fw.root_section is None:
            return
        sections = self.section.get_queryset(fw)
        for sec in sections:
            sec_model = self.section.from_django(sec)
            if sec_model.parent:
                parent_sec = self.get(self.section, str(sec_model.parent))
                self.add_child(parent_sec, sec_model)
            else:
                self.add_child(fw_model, sec_model)

    def load_measure_templates(self, fw: Framework):
        mt_objs = self.measure_template.get_queryset(fw)
        for mt in mt_objs:
            section = self.get(SectionModel, dict(uuid=mt['section']))
            mt_model = self.measure_template.from_django(mt)
            self.add_child(section, mt_model)

    def load(self, framework_id: str | None = None) -> None:
        qs = Framework.objects.select_for_update()
        if framework_id:
            qs = qs.filter(identifier=framework_id)
        for fw_obj in qs:
            fw = FrameworkModel.from_django(fw_obj)
            self.add(fw)
            self.load_sections(fw_obj, fw)
            self.load_measure_templates(fw_obj)


class FrameworkJSONAdapter(JSONAdapter, FrameworkAdapter):
    def load(self) -> None:
        self.load_json()
        data = self.data
        fw_model = self.framework(**data['framework'])
        self.add(fw_model)
        for sec in data['sections']:
            measure_templates = sec.pop('measure_templates', [])
            sec_model = self.section(**sec)
            if sec_model.parent:
                parent_sec = self.get(self.section, str(sec_model.parent))
                self.add_child(parent_sec, sec_model)
            else:
                self.add_child(fw_model, sec_model)
            for mt in measure_templates:
                mt_model = self.measure_template(section=sec_model.uuid, **mt)  # type: ignore[attr-defined]
                self.add_child(sec_model, mt_model)
        self.update(fw_model)
