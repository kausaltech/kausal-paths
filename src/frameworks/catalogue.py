"""Effective catalogue visibility; a framework definition is never copied into each dependent instance."""

from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.db.models import Q

from kausal_common.datasets.models import DatasetSchemaScope, DimensionScope

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from nodes.models import InstanceConfig


def catalogue_scope_q(instance: InstanceConfig) -> Q:
    query = Q(scope_content_type=ContentType.objects.get_for_model(instance), scope_id=instance.pk)
    if instance.has_framework_config():
        framework = instance.framework_config.framework
        query |= Q(scope_content_type=ContentType.objects.get_for_model(framework), scope_id=framework.pk)
    return query


def dimension_scopes(instance: InstanceConfig) -> QuerySet[DimensionScope]:
    return DimensionScope.objects.filter(catalogue_scope_q(instance))


def schema_scopes(instance: InstanceConfig) -> QuerySet[DatasetSchemaScope]:
    return DatasetSchemaScope.objects.filter(catalogue_scope_q(instance))
