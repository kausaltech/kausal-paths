from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any, TypedDict, cast
from uuid import UUID, uuid4

from django.contrib.postgres.expressions import ArraySubquery
from django.db.models import Q
from django.db.models.expressions import F, OuterRef, Value
from django.db.models.functions import JSONObject
from django.db.models.functions.text import Concat
from pydantic import Field

from kausal_common.models.django_pydantic import DjangoAdapter, DjangoDiffModel, JSONAdapter, TypedAdapter

from frameworks.models import (
    Framework,
    FrameworkConfig,
    Measure,
    MeasureDataPoint,
    MeasureTemplate,
)
from frameworks.sync_frameworks import FrameworkModel

if TYPE_CHECKING:
    from pathlib import Path

    from django.db.models import QuerySet


class MeasureDataPointModel(DjangoDiffModel[MeasureDataPoint]):
    _model = MeasureDataPoint
    _modelname = 'measure_datapoint'
    _identifiers = ('measure', 'year')
    _attributes = ('value', 'default_value')
    _parent_key = 'measure'
    _parent_type = 'measure'

    measure: str

    @classmethod
    def get_queryset(cls, fwc_or_fw: FrameworkConfig | Framework) -> QuerySet[MeasureDataPoint, dict[str, Any]]:
        if isinstance(fwc_or_fw, FrameworkConfig):
            q = Q(measure__framework_config=fwc_or_fw)
        else:
            q = Q(measure__framework_config__framework=fwc_or_fw)

        mdp_fields = cls._django_fields.field_names - {'measure'}
        measure_f = Concat(F('measure__measure_template__uuid'), Value('__'), F('measure__framework_config__uuid'))
        mdp_objs = (
            MeasureDataPoint.objects.filter(q).values(*mdp_fields).annotate(_instance_pk=F('pk')).annotate(measure=measure_f)
        )
        return mdp_objs


class MeasureModel(DjangoDiffModel[Measure]):
    _model = Measure
    _modelname = 'measure'
    _identifiers = ('framework_config', 'measure_template')
    _attributes = ('unit', 'internal_notes', 'data_points')
    _parent_key = 'framework_config'
    _parent_type = 'framework_config'
    _allow_related_model_deletion = True

    class DataPoint(TypedDict):
        year: int
        value: float | None
        default_value: float | None

    measure_template: UUID
    framework_config: UUID
    data_points: list[DataPoint] = Field(default_factory=list)

    @classmethod
    def get_queryset(cls, fwc_or_fw: FrameworkConfig | Framework) -> QuerySet[Measure, dict[str, Any]]:
        if isinstance(fwc_or_fw, FrameworkConfig):
            q = Q(framework_config=fwc_or_fw)
        else:
            q = Q(framework_config__framework=fwc_or_fw)

        dps = (
            MeasureDataPoint.objects.filter(
                measure_id=OuterRef('pk'),
            )
            .annotate(
                data=JSONObject(
                    year=F('year'),
                    value=F('value'),
                    default_value=F('default_value'),
                )
            )
            .values_list('data')
        )
        m_fields = cls._django_fields.field_names - {'measure_template', 'framework_config'}
        m_objs = (
            Measure.objects.filter(q)
            .values(*m_fields)
            .annotate(_instance_pk=F('pk'))
            .annotate(_template_order=F('measure_template__order'))
            .annotate(_section_order=F('measure_template__section__path'))
            .annotate(measure_template=F('measure_template__uuid'))
            .annotate(framework_config=F('framework_config__uuid'))
            .annotate(data_points=ArraySubquery(dps))
            .order_by('_section_order', '_template_order')
        )
        return m_objs

    def get_excludes(self) -> set[str]:
        excludes = super().get_excludes()
        excludes.update(('_template_order', '_section_order'))
        return excludes

    @classmethod
    def get_create_kwargs(cls, adapter: DjangoAdapter, ids: dict, attrs: dict) -> dict:
        kwargs = super().get_create_kwargs(adapter, ids, attrs)
        fwc_model = cast('FrameworkConfigModel', adapter.get(FrameworkConfigModel, str(kwargs.pop('framework_config'))))
        kwargs['framework_config'] = fwc_model._instance
        mt_uuid: UUID = kwargs.pop('measure_template')
        assert isinstance(adapter, FrameworkConfigDjangoAdapter)
        kwargs['measure_template'] = adapter.templates_by_uuid[mt_uuid]
        return kwargs

    @classmethod
    def create_related(cls, _adapter: DjangoAdapter, _ids: dict, attrs: dict, instance: Measure, /) -> None:
        dps = attrs.get('data_points')
        if not dps:
            return
        dp_objs = [MeasureDataPoint(measure=instance, **dp) for dp in dps]
        MeasureDataPoint.objects.bulk_create(dp_objs)

    def update_related(self, obj: Measure, attrs: dict) -> None:
        data_points = attrs.get('data_points')
        if data_points is None:
            return
        obj.data_points.all().delete()
        assert isinstance(self.adapter, DjangoAdapter)
        self.create_related(self.adapter, {}, attrs, obj)


class FrameworkConfigModel(DjangoDiffModel[FrameworkConfig]):
    _model = FrameworkConfig
    _modelname = 'framework_config'
    _identifiers = ('uuid',)
    _attributes = (
        'framework',
        'organization_name',
        'organization_identifier',
        'organization_slug',
        'baseline_year',
        'created_at',
        'instance_identifier',
    )
    _children = {'measure': 'measures'}

    framework: UUID
    instance_identifier: str
    created_at: datetime | None = None
    measures: list[str] = Field(default_factory=list)
    uuid: UUID = Field(default_factory=uuid4)

    @classmethod
    def get_queryset(cls, fw: Framework) -> QuerySet[FrameworkConfig, dict[str, Any]]:
        qs = fw.configs.get_queryset()
        fields = cls._django_fields.field_names - {'framework'}
        fwc_objs = (
            qs.values(*fields)
            .annotate(_instance_pk=F('pk'))
            .annotate(framework=F('framework__uuid'))
            .annotate(instance_identifier=F('instance_config__identifier'))
            .order_by('created_at')
        )
        return fwc_objs

    @classmethod
    def get_create_kwargs(cls, adapter: DjangoAdapter, ids: dict, attrs: dict) -> dict:
        kwargs = super().get_create_kwargs(adapter, ids, attrs)
        fw_model = cast('FrameworkModel', adapter.get(FrameworkModel, str(kwargs.pop('framework'))))
        assert fw_model._instance is not None
        kwargs['framework'] = fw_model._instance
        return kwargs

    @classmethod
    def create_django_instance(cls, _adapter: DjangoAdapter, create_kwargs: dict) -> FrameworkConfig:
        fw = create_kwargs.pop('framework')
        org_name = create_kwargs.pop('organization_name')
        baseline_year = create_kwargs.pop('baseline_year')
        uuid = create_kwargs.pop('uuid')
        instance_identifier = create_kwargs.pop('instance_identifier')
        fwc = FrameworkConfig.create_instance(
            fw,
            instance_identifier=instance_identifier,
            org_name=org_name,
            baseline_year=baseline_year,
            uuid=uuid,
        )
        for key, val in create_kwargs.items():
            setattr(fwc, key, val)
        fwc.save()
        return fwc


class FrameworkConfigAdapter(TypedAdapter):
    framework_config = FrameworkConfigModel
    measure = MeasureModel
    framework = FrameworkModel
    top_level = ['framework_config']


class FrameworkConfigDjangoAdapter(FrameworkConfigAdapter, DjangoAdapter):
    templates_by_uuid: dict[UUID, MeasureTemplate]

    def __init__(self, framework_id: str, /, **kwargs) -> None:
        self.framework_id = framework_id
        self.templates_by_uuid = {}
        super().__init__(**kwargs)

    def load_measures(self, fw: Framework):
        qs = self.measure.get_queryset(fw)

        for m_data in qs:
            fwc_model = self.get(FrameworkConfigModel, str(m_data['framework_config']))
            m_model = MeasureModel.from_django(m_data)
            self.add_child(fwc_model, m_model)

    def load(self) -> None:
        fw = Framework.objects.select_for_update().filter(identifier=self.framework_id).first()
        if fw is None:
            raise Exception("Framework '%s' not found" % self.framework_id)
        self.templates_by_uuid = {mt.uuid: mt for mt in fw.measure_templates()}
        fw_model = self.framework.from_django(fw)
        self.add(fw_model)
        for fwc_data in self.framework_config.get_queryset(fw):
            fwc = FrameworkConfigModel.from_django(fwc_data)
            self.add(fwc)
        self.load_measures(fw)


class FrameworkConfigJSONAdapter(FrameworkConfigAdapter, JSONAdapter):
    def __init__(self, framework_id: str, path: Path, /, **kwargs) -> None:
        self.framework_id = framework_id
        super().__init__(path, **kwargs)

    def load(self) -> None:
        data = self.load_json()
        assert isinstance(data, dict)
        for fwc_data in data['framework_config']:
            fwc_model = self.framework_config(**fwc_data)
            self.add(fwc_model)

        for m_data in data['measure']:
            m_model = self.measure(**m_data)
            fwc_model = self.get(self.framework_config, str(m_model.framework_config))
            self.add_child(fwc_model, m_model)
