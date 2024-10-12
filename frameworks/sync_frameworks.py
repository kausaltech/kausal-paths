from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast
from uuid import UUID, uuid4

from django.contrib.postgres.expressions import ArraySubquery
from django.db.models.expressions import F, OuterRef
from django.db.models.functions import JSONObject
from pydantic import Field, PrivateAttr

from diffsync.diff import Diff
from diffsync.enum import DiffSyncFlags
from diffsync.store.local import LocalStore
from loguru import logger

from kausal_common.models.django_pydantic import DjangoAdapter, DjangoDiffModel, JSONAdapter, TypedAdapter

from frameworks.models import (
    Framework,
    MeasureTemplate,
    MeasureTemplateDefaultDataPoint,
    Section,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.db.models import QuerySet

    from diffsync import Adapter


class SectionModel(DjangoDiffModel[Section]):
    _model = Section
    _modelname = 'section'
    _identifiers = ('uuid',)
    _attributes = ('parent', 'identifier', 'name', 'description', 'available_years', 'framework')
    _parent_key = 'parent'
    _children = {
        'measure_template': 'measure_templates',
        'section': 'children',
    }

    framework: str
    parent: UUID | None
    children: list[UUID] = Field(default_factory=list)
    measure_templates: list[str] = Field(default_factory=list)

    @classmethod
    def get_queryset(cls, fw: Framework) -> QuerySet[Section, dict[str, Any]]:
        if fw.root_section is None:
            qs_base = cls._model.objects.get_queryset().none()
        else:
            qs_base = fw.root_section.get_descendants().order_by('path')
        plain_fields = list(cls._django_fields.plain_fields.keys())
        plain_fields.remove('framework')
        sections = qs_base.annotate_parent_field('parent', 'uuid', min_depth=2).values(
            *plain_fields, 'parent', _instance_pk=F('pk'),
        )
        return sections

    @classmethod
    def get_mpnode_root_instance(cls, instance: Section) -> Section | None:
        assert instance.framework.root_section is not None
        return instance.framework.root_section

    @classmethod
    def get_create_kwargs(cls, adapter: DjangoAdapter, ids: dict, attrs: dict) -> dict:
        kwargs = super().get_create_kwargs(adapter, ids, attrs)
        fw_id = kwargs.pop('framework')
        fw = adapter.get(FrameworkModel, fw_id)
        fw_obj = fw.get_django_instance()
        kwargs['framework'] = fw_obj
        return kwargs


class MeasureTemplateModel(DjangoDiffModel[MeasureTemplate]):
    _model = MeasureTemplate
    _identifiers = ('uuid',)
    _modelname = 'measure_template'
    _attributes = (
        'name',
        'unit',
        'priority',
        'min_value',
        'max_value',
        'time_series_max',
        'default_value_source',
        'default_data_points',
        'section',
    )
    _parent_key = 'section'
    _parent_type = 'section'

    class DefaultDataPoint(TypedDict):
        year: int
        value: float

    section: UUID
    default_data_points: list[DefaultDataPoint] = Field(default_factory=list)
    _section_pk: int | None = PrivateAttr(default=None)

    @classmethod
    def get_queryset(cls, fw: Framework) -> QuerySet[MeasureTemplate, dict[str, Any]]:
        mt_fields = cls._django_fields.field_names - {'section'}
        ddps = (
            MeasureTemplateDefaultDataPoint.objects.filter(
                template_id=OuterRef('pk'),
            )
            .annotate(
                data=JSONObject(
                    year=F('year'),
                    value=F('value'),
                ),
            )
            .values_list('data')
        )
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

    def get_excludes(self) -> set[str]:
        excludes = super().get_excludes()
        excludes.add('_section_pk')
        return excludes


class FrameworkModel(DjangoDiffModel[Framework]):
    _model = Framework
    _modelname = 'framework'
    _identifiers = ('uuid',)
    _attributes = ('identifier', 'name', 'description', 'public_base_fqdn', 'result_excel_url', 'result_excel_node_ids')
    _children = {'section': 'sections'}

    sections: list[str] = Field(default_factory=list)
    identifier: str
    uuid: UUID = Field(default_factory=uuid4)

    @classmethod
    def create_related(cls, adapter: DjangoAdapter, ids: dict, attrs: dict, instance: Framework, /) -> None:  # noqa: ARG003
        assert instance.root_section is None
        instance.create_root_section()


class FrameworkAdapter(TypedAdapter):
    framework = FrameworkModel
    section = SectionModel
    measure_template = MeasureTemplateModel
    top_level = ['framework']


class FrameworkDjangoAdapter(DjangoAdapter, FrameworkAdapter):
    def load_sections(self, fw: Framework, fw_model: FrameworkModel):
        if fw.root_section is None:
            return
        sections = self.section.get_queryset(fw)
        fw_uid = fw_model.get_unique_id()
        for sec in sections:
            sec['framework'] = fw_uid
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

    def _change_fw_uuid(self, fw: FrameworkModel, new_uuid: UUID) -> None:
        store = self.store
        assert isinstance(store, LocalStore)
        if 'uuid' in fw._identifiers:
            fw_data = store._data['framework']
            fw_data[str(new_uuid)] = fw_data.pop(str(fw.uuid))

        fw.uuid = new_uuid
        for sec in self.get_all(SectionModel):
            sec.framework = str(new_uuid)
        for sec in fw.get_children(SectionModel):
            sec._parent_id = fw.get_unique_id()
        obj = fw._instance
        assert obj is not None
        obj.uuid = new_uuid  # pyright: ignore
        obj.save(update_fields=['uuid'])

    def diff_from(
        self,
        source: Adapter,
        diff_class: type[Diff] = Diff,
        flags: DiffSyncFlags = DiffSyncFlags.NONE,
        callback: Callable[[str, int, int], None] | None = None,
    ) -> Diff:
        if True:
            my_fws = {fw.identifier: fw for fw in self.get_all(FrameworkModel)}
            assert isinstance(source, TypedAdapter)
            src_fws = {fw.identifier: fw for fw in source.get_all(FrameworkModel)}
            for fw_id, fw in my_fws.items():
                src_fw = src_fws.get(fw_id)
                if src_fw is None:
                    continue
                if str(fw.uuid) != str(src_fw.uuid):
                    logger.warning("Changing Framework %s UUID from %s to %s" % (fw.identifier, fw.uuid, src_fw.uuid))
                    self._change_fw_uuid(fw, src_fw.uuid)
        return super().diff_from(source, diff_class, flags, callback)


class FrameworkJSONAdapter(JSONAdapter, FrameworkAdapter):
    def load_legacy(self, data: dict):
        fw_model = self.framework(**data['framework'])
        self.add(fw_model)
        for sec in data['sections']:
            measure_templates = sec.pop('measure_templates', [])
            sec_model = self.section(framework=str(fw_model.uuid), **sec)
            if sec_model.parent:
                parent_sec = self.get(self.section, str(sec_model.parent))
                self.add_child(parent_sec, sec_model)
            else:
                self.add_child(fw_model, sec_model)
            for mt in measure_templates:
                mt_model = self.measure_template(section=sec_model.uuid, **mt)  # type: ignore[attr-defined]
                self.add_child(sec_model, mt_model)
        self.update(fw_model)

    def load(self) -> None:
        data = self.load_json()
        assert isinstance(data, dict)
        if 'sections' in data:
            self.load_legacy(data)
            return

        assert len(data['framework']) == 1
        fw_model = self.framework.model_validate(data['framework'][0])
        self.add(fw_model)
        for sec_data in data['section']:
            sec_model = self.section.model_validate(sec_data)
            if sec_model.parent:
                parent_sec = self.get(self.section, str(sec_model.parent))
                self.add_child(parent_sec, sec_model)
            else:
                self.add_child(fw_model, sec_model)

        for mt_data in data['measure_template']:
            mt_model = self.measure_template.model_validate(mt_data)
            sec_model = self.get(self.section, str(mt_model.section))
            self.add_child(sec_model, mt_model)
