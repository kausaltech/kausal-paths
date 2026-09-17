from typing import TYPE_CHECKING

from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest

import pytest

from paths.schema_context import PathsGraphQLContext

if TYPE_CHECKING:
    from pytest_django.fixtures import DjangoAssertNumQueries

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


def test_instance_metric_attributes(instance_config: InstanceConfig, django_assert_num_queries: DjangoAssertNumQueries) -> None:
    request = HttpRequest()
    request.user = AnonymousUser()
    context = PathsGraphQLContext(request=request, response=None)
    with django_assert_num_queries(0):
        assert context.get_metric_attributes() == {}
        context.instance_config = instance_config
        assert context.get_metric_attributes() == {
            'instance.id': instance_config.identifier,
            'instance.uuid': str(instance_config.uuid),
        }
