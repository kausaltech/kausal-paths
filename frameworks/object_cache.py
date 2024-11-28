from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from paths.context import ModelObjectCache, ObjectCacheGroup

from frameworks.models import (
    Framework,
    FrameworkConfig,
    FrameworkQuerySet,
    Measure,
    MeasureDataPoint,
    MeasureQuerySet,
    MeasureTemplate,
    MeasureTemplateDefaultDataPoint,
    MeasureTemplateDefaultDataPointQuerySet,
    MeasureTemplateQuerySet,
    Section,
    SectionQuerySet,
)

if TYPE_CHECKING:
    from kausal_common.users import UserOrAnon

    class CFramework(Framework):
        cache: FrameworkSpecificCache

    class CSection(Section):
        fw_cache: FrameworkSpecificCache
        parent_id: int | None
        _cached_parent_obj: CSection | None

    class CMeasureTemplate(MeasureTemplate):
        fw_cache: FrameworkSpecificCache

    class CFrameworkConfig(FrameworkConfig):
        cache: FrameworkConfigSpecificCache

    class CMeasure(Measure):
        fwc_cache: FrameworkConfigSpecificCache


class SectionCache(ModelObjectCache[Section, 'CSection', SectionQuerySet, 'CFramework']):
    @property
    def model(self):
        return Section

    def populate(self, qs: SectionQuerySet) -> list[CSection]:
        qs = qs.annotate_parent_field('parent_id', 'id')
        obj_list = self._as_list(qs)
        obj_by_id: dict[int, CSection] = {}
        for obj in obj_list:
            obj_by_id[obj.pk] = obj
            obj.fw_cache = self.parent.cache

        for obj in obj_list:
            parent_id = obj.parent_id
            if parent_id is not None:
                parent = obj_by_id.get(parent_id, self._by_id.get(parent_id))
            else:
                parent = None
            obj._cached_parent_obj = parent
        return obj_list


class MeasureTemplateCache(ModelObjectCache[MeasureTemplate, 'CMeasureTemplate', MeasureTemplateQuerySet, 'CFramework']):
    @property
    def model(self):
        return MeasureTemplate

    def __post_init__(self):
        super().__post_init__()
        self._groups['section'] = ObjectCacheGroup(self, lambda obj: obj.section_id)

    def by_section(self, section_id: int) -> list[CMeasureTemplate]:
        return self.get_list_by_group('section', section_id)

    def add_obj(self, obj: CMeasureTemplate) -> None:
        obj.fw_cache = self.parent.cache
        super().add_obj(obj)


@dataclass
class MeasureTemplateDefaultDataPointCache(ModelObjectCache[
    MeasureTemplateDefaultDataPoint,
    MeasureTemplateDefaultDataPoint,
    MeasureTemplateDefaultDataPointQuerySet,
    'CFramework',
]):
    @property
    def model(self):
        return MeasureTemplateDefaultDataPoint

    def __post_init__(self):
        self._groups['measure_template'] = ObjectCacheGroup(self, lambda obj: obj.template_id)

    def by_measure_template(self, measure_template_id: int) -> list[MeasureTemplateDefaultDataPoint]:
        return self.get_list_by_group('measure_template', measure_template_id)


@dataclass
class FrameworkSpecificCache:
    framework: CFramework
    user: UserOrAnon | None
    sections: SectionCache = field(init=False)
    measure_templates: MeasureTemplateCache = field(init=False)
    measure_template_default_data_points: MeasureTemplateDefaultDataPointCache = field(init=False)

    def __post_init__(self):
        self.sections = SectionCache(self.framework, self.user)
        self.measure_templates = MeasureTemplateCache(self.framework, self.user)
        self.measure_template_default_data_points = MeasureTemplateDefaultDataPointCache(self.framework, self.user)


class MeasureCache(ModelObjectCache[Measure, 'CMeasure', MeasureQuerySet, 'CFramework']):
    @property
    def model(self):
        return Measure

    def __post_init__(self):
        super().__post_init__()
        self._groups['section'] = ObjectCacheGroup(self, lambda obj: obj.section_id)

    def by_section(self, section_id: int) -> list[CMeasureTemplate]:
        return self.get_list_by_group('section', section_id)

    def add_obj(self, obj: CMeasureTemplate) -> None:
        obj.fw_cache = self.parent.cache
        super().add_obj(obj)


@dataclass
class FrameworkConfigSpecificCache:
    framework_config: CFrameworkConfig
    user: UserOrAnon | None
    measures: MeasureCache = field(init=False)
    measure_datapoints: MeasureDataPointCache = field(init=False)

    def __post_init__(self):
        self.measures = MeasureCache(self.framework_config, self.user)
        self.measure_datapoints = MeasureDataPointCache(self.framework_config, self.user)


class FrameworkCache(ModelObjectCache[Framework, 'CFramework', FrameworkQuerySet, None]):
    @property
    def model(self):
        return Framework

    def populate(self, qs: FrameworkQuerySet) -> list[CFramework]:
        obj_list = self._as_list(qs)
        for obj in obj_list:
            obj.cache = FrameworkSpecificCache(obj, self.user)
        return obj_list
