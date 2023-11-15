from __future__ import annotations
from functools import cached_property

from nodes.models import InstanceConfig, NodeConfig



class InstanceSpecificCache:
    ic: 'InstanceConfig'
    nc_by_id: dict[int, NodeConfig]

    def __init__(self, ic: 'InstanceConfig'):
        self.ic = ic
        self.nc_by_id = {}

    @classmethod
    def fetch_by_identifier(cls, identifier: str) -> InstanceSpecificCache:
        ic = InstanceConfig.objects.get(identifier=identifier)
        return cls(ic)


class PathsObjectCache:
    instance_caches: dict[str, InstanceSpecificCache]
    admin_instance_cache: InstanceSpecificCache | None

    def __init__(self):
        self.instance_caches = {}
        self.admin_instance_cache = None

    def for_instance_identifier(self, identifier: str) -> InstanceSpecificCache:
        instance_cache = self.instance_caches.get(identifier)
        if instance_cache is None:
            instance_cache = InstanceSpecificCache.fetch_by_identifier(identifier)
            self.instance_caches[identifier] = instance_cache
        return instance_cache

    def for_ic(self, ic: InstanceConfig) -> InstanceSpecificCache:
        return self.for_instance_identifier(ic.identifier)
