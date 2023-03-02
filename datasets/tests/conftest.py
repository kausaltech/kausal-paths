import json

import pytest
from django.urls import reverse
from rest_framework.test import APIClient


class JSONAPIClient(APIClient):
    default_format = 'json'

    def request(self, **kwargs):
        if 'HTTP_ACCEPT' not in kwargs:
            kwargs['HTTP_ACCEPT'] = 'application/json'
        resp = super().request(**kwargs)
        resp.json_data = json.loads(resp.content)
        return resp


@pytest.fixture
def api_client():
    client = JSONAPIClient()
    return client


@pytest.fixture
def dataset_list_url(instance_config):
    return reverse('instance-datasets-list', kwargs={'instance_pk': instance_config.pk})
