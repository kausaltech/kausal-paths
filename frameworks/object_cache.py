from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kausal_common.models.object_cache import ModelObjectCache, ObjectCacheGroup

from frameworks.models import (
    Framework,
    FrameworkConfig,
    FrameworkConfigQuerySet,
    FrameworkQuerySet,
    Measure,
    MeasureDataPoint,
    MeasureDataPointQuerySet,
    MeasureQuerySet,
    MeasureTemplate,
    MeasureTemplateDefaultDataPoint,
    MeasureTemplateDefaultDataPointQuerySet,
    MeasureTemplateQuerySet,
    Section,
    SectionQuerySet,
)
from nodes.models import InstanceConfig, InstanceConfigQuerySet

if TYPE_CHECKING:
    from kausal_common.users import UserOrAnon


@dataclass
class SectionCacheData:
    fw_cache: FrameworkSpecificCache
    parent_id: int | None = field(init=False)

    def get_parent(self) -> Section | None:
        if self.parent_id is None:
            return None
        return self.fw_cache.sections.get(self.parent_id)


class SectionCache(ModelObjectCache[Section, SectionQuerySet, Framework]):
    @property
    def model(self):
        return Section

    def filter_by_parent(self, qs: SectionQuerySet) -> SectionQuerySet:
        return qs.filter(framework=self.parent)

    def populate(self, qs: SectionQuerySet) -> list[Section]:
        qs = qs.annotate_parent_field('parent_id', 'id')
        obj_list = self._as_list(qs)
        obj_by_id: dict[int, Section] = {}
        for obj in obj_list:
            obj_by_id[obj.pk] = obj
            if not obj.has_cache():
                obj.cache = SectionCacheData(self.parent.cache)
                obj.cache.parent_id = getattr(obj, 'parent_id')  # noqa: B009

        return obj_list


class MeasureTemplateCache(ModelObjectCache[MeasureTemplate, MeasureTemplateQuerySet, Framework]):
    @property
    def model(self):
        return MeasureTemplate

    def filter_by_parent(self, qs: MeasureTemplateQuerySet) -> MeasureTemplateQuerySet:
        return qs.filter(section__framework=self.parent)

    def __post_init__(self):
        super().__post_init__()
        self._groups['section'] = ObjectCacheGroup(self, lambda obj: obj.section_id)

    def by_section(self, section_id: int) -> list[MeasureTemplate]:
        return self.get_list_by_group('section', section_id)

    def add_obj(self, obj: MeasureTemplate) -> None:
        obj.cache = self.parent.cache
        super().add_obj(obj)


@dataclass
class MeasureTemplateDefaultDataPointCache(ModelObjectCache[
    MeasureTemplateDefaultDataPoint,
    MeasureTemplateDefaultDataPointQuerySet,
    Framework,
]):
    @property
    def model(self):
        return MeasureTemplateDefaultDataPoint

    def __post_init__(self):
        self._groups['measure_template'] = ObjectCacheGroup(self, lambda obj: obj.template_id)

    def filter_by_parent(self, qs: MeasureTemplateDefaultDataPointQuerySet) -> MeasureTemplateDefaultDataPointQuerySet:
        return qs.filter(template__section__framework=self.parent)

    def by_measure_template(self, measure_template_id: int) -> list[MeasureTemplateDefaultDataPoint]:
        return self.get_list_by_group('measure_template', measure_template_id)


@dataclass
class MeasureDataPointCache(ModelObjectCache[MeasureDataPoint, MeasureDataPointQuerySet, FrameworkConfig]):
    @property
    def model(self):
        return MeasureDataPoint

    def __post_init__(self):
        self._groups['measure'] = ObjectCacheGroup(self, lambda obj: obj.measure_id)

    def by_measure(self, measure_id: int) -> list[MeasureDataPoint]:
        return self.get_list_by_group('measure', measure_id)


class MeasureCache(ModelObjectCache[Measure, MeasureQuerySet, FrameworkConfig]):
    @property
    def model(self):
        return Measure

    def filter_by_parent(self, qs: MeasureQuerySet) -> MeasureQuerySet:
        return qs.filter(framework_config=self.parent)

    def __post_init__(self):
        super().__post_init__()
        self._groups['framework_config'] = ObjectCacheGroup(self, lambda obj: obj.framework_config_id)
        self._groups['measure_template'] = ObjectCacheGroup(self, lambda obj: obj.measure_template_id)

    def by_framework_config(self, framework_config_id: int) -> list[Measure]:
        return self.get_list_by_group('framework_config', framework_config_id)

    def by_measure_template(self, measure_template_id: int) -> list[Measure]:
        return self.get_list_by_group('measure_template', measure_template_id)

    def add_obj(self, obj: Measure) -> None:
        obj.cache = self.parent.cache
        super().add_obj(obj)


@dataclass
class FrameworkConfigCacheData:
    fw_cache: FrameworkSpecificCache
    framework_config: FrameworkConfig
    user: UserOrAnon | None
    measures: MeasureCache = field(init=False)
    measure_datapoints: MeasureDataPointCache = field(init=False)

    def __post_init__(self):
        self.measures = MeasureCache(self.framework_config, self.user)
        self.measure_datapoints = MeasureDataPointCache(self.framework_config, self.user)


class FrameworkConfigCache(ModelObjectCache[FrameworkConfig, FrameworkConfigQuerySet, Framework]):
    @property
    def model(self):
        return FrameworkConfig

    def add_obj(self, obj: FrameworkConfig) -> None:
        obj.cache = FrameworkConfigCacheData(self.parent.cache, obj, self.user)
        super().add_obj(obj)


class InstanceConfigCache(ModelObjectCache[InstanceConfig, InstanceConfigQuerySet, Framework]):
    @property
    def model(self):
        return InstanceConfig

    def add_obj(self, obj: InstanceConfig) -> None:
        from nodes.object_cache import InstanceConfigCacheData
        obj.cache = InstanceConfigCacheData()
        super().add_obj(obj)


@dataclass
class FrameworkSpecificCache:
    framework: Framework
    user: UserOrAnon | None
    sections: SectionCache = field(init=False)
    measure_templates: MeasureTemplateCache = field(init=False)
    measure_template_default_data_points: MeasureTemplateDefaultDataPointCache = field(init=False)
    framework_configs: FrameworkConfigCache = field(init=False)
    instance_configs: InstanceConfigCache = field(init=False)

    def __post_init__(self):
        self.sections = SectionCache(self.framework, self.user)
        self.measure_templates = MeasureTemplateCache(self.framework, self.user)
        self.measure_template_default_data_points = MeasureTemplateDefaultDataPointCache(self.framework, self.user)
        self.framework_configs = FrameworkConfigCache(self.framework, self.user)
        self.instance_configs = InstanceConfigCache(self.framework, self.user)


class FrameworkCache(ModelObjectCache[Framework, FrameworkQuerySet, None]):
    @property
    def model(self):
        return Framework

    def populate(self, qs: FrameworkQuerySet) -> list[Framework]:
        obj_list = self._as_list(qs)
        for obj in obj_list:
            obj.cache = FrameworkSpecificCache(obj, self.user)
        return obj_list
