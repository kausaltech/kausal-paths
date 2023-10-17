import pytest
from django.urls import reverse
from rest_framework.exceptions import ErrorDetail

from datasets.models import Dataset

pytestmark = pytest.mark.django_db


def assert_unauthorized_with_no_data(response):
    assert response.status_code == 403
    assert set(response.data.keys()) == {'detail'}
    assert type(response.data['detail']) == ErrorDetail


def test_dataset_unauthorized_without_authenticated_user(api_client, dataset, dataset_list_url):
    response = api_client.get(dataset_list_url)
    assert_unauthorized_with_no_data(response)


"""
def test_dataset_list_authorized(api_client, user, instance_config_factory, dataset_factory):
    instance1 = instance_config_factory()
    instance2 = instance_config_factory()
    count = 3
    for _ in range(0, count):
        dataset_factory(instance=instance1)
    dataset_factory(instance=instance2)
    api_client.force_login(user)
    response = api_client.get(
        reverse('instance-datasets-list',
                kwargs={'instance_pk': instance1.pk})
    )
    assert response.status_code == 200
    assert len(response.data) == count
"""

def generate_dataset_metadata():
    return {
        'identifier': 'test-dataset-identifier',
        'name': 'Test Dataset',
        'years': [2023, 2024, 2025],
        'dimension_selections': [],  # TODO
        'metrics': []  # TODO
    }


FIELDS = ('identifier', 'name', 'years')


def assert_dataset_equals(dataset_model_instance, payload):
    ds = dataset_model_instance
    p = payload
    for field in FIELDS:
        assert p[field] == getattr(ds, field)


def test_dataset_metadata_create(api_client, admin_user, dataset_list_url, dataset_factory):
    api_client.force_login(admin_user)
    metadata = generate_dataset_metadata()
    response = api_client.post(
        dataset_list_url,
        metadata
    )
    assert response.status_code == 201
    assert_dataset_equals(Dataset.objects.get(pk=response.data['id']), metadata)


def test_dataset_metadata_update(api_client, admin_user, dataset, dataset_list_url, dataset_factory):
    api_client.force_login(admin_user)
    metadata = generate_dataset_metadata()
    pks = {'instance_pk': dataset.instance.pk, 'pk': dataset.pk}
    response = api_client.put(
        reverse('instance-datasets-detail', kwargs=pks),
        metadata
    )
    assert response.status_code == 200
    assert_dataset_equals(Dataset.objects.get(pk=pks['pk']), metadata)


def test_dataset_metadata_create_unauthorized(api_client, user, dataset_list_url, dataset_factory):
    api_client.force_login(user)
    metadata = generate_dataset_metadata()
    response = api_client.post(
        dataset_list_url,
        metadata
    )
    assert_unauthorized_with_no_data(response)


def test_dataset_metadata_update_unauthorized(api_client, user, dataset, dataset_list_url, dataset_factory):
    api_client.force_login(user)
    metadata = generate_dataset_metadata()
    pks = {'instance_pk': dataset.instance.pk, 'pk': dataset.pk}
    response = api_client.put(
        reverse('instance-datasets-detail', kwargs=pks),
        metadata
    )
    assert_unauthorized_with_no_data(response)
