from django.core.exceptions import PermissionDenied
from django.urls import reverse

import pytest

from kausal_common.datasets.wagtail_hooks import DataSourceViewSet
from kausal_common.testing.utils import parse_table

from paths.context import RealmContext, realm_context

from .fixtures import *


@pytest.mark.django_db
class TestDataSourceAdminAuthorization:
    """Test authorization for data source admin views."""

    @pytest.mark.parametrize(('user_key', 'expected_data_sources'), [
        ('superuser', ['data_source1', 'data_source1_alternative']),
        ('admin_user', ['data_source1', 'data_source1_alternative']),
        ('super_admin_user', ['data_source1', 'data_source1_alternative']),
        ('reviewer_user', ['data_source1', 'data_source1_alternative']),
        ('viewer_user', ['data_source1', 'data_source1_alternative']),
        ('schema1_viewer', ['data_source1', 'data_source1_alternative']),
        ('schema1_editor', ['data_source1', 'data_source1_alternative']),
        ('schema1_admin', ['data_source1', 'data_source1_alternative']),
        ('schema1_viewer_group_user', ['data_source1', 'data_source1_alternative']),
        ('schema1_editor_group_user', ['data_source1', 'data_source1_alternative']),
        ('schema1_admin_group_user', ['data_source1', 'data_source1_alternative']),
        ('regular_user', []),
    ])
    def test_data_source_index_view(self, client, dataset_test_data, user_key, expected_data_sources, get_in_admin_context):
        """Test access to data source index view."""
        data = dataset_test_data
        user = data[user_key]

        view = DataSourceViewSet().index_view
        url = reverse('datasets_datasource:list')
        if expected_data_sources:
            response = get_in_admin_context(user, view, url, data['instance1'])
        else:
            with pytest.raises(PermissionDenied):
                response = get_in_admin_context(user, view, url, data['instance1'])
            return

        ctx = RealmContext(realm=data['instance1'], user=user)
        with realm_context.activate(ctx):
            response.render()

        assert response.status_code == 200
        content = response.content.decode('utf-8')
        for ds_key in expected_data_sources:
            ds_name = data[ds_key].name
            assert ds_name in content
        # data_source2 should not be visible to instance1 users
        assert data['data_source2'].name not in content

    @pytest.mark.parametrize(*parse_table("""
user_key                   data_source_key                access_allowed

superuser                  data_source1                   +
superuser                  data_source1_alternative       +
superuser                  data_source2                   +
admin_user                 data_source1                   +
admin_user                 data_source1_alternative       +
admin_user                 data_source2                   -
super_admin_user           data_source1                   +
super_admin_user           data_source1_alternative       +
super_admin_user           data_source2                   -
reviewer_user              data_source1                   -
reviewer_user              data_source1_alternative       -
reviewer_user              data_source2                   -
viewer_user                data_source1                   -
viewer_user                data_source1_alternative       -
viewer_user                data_source2                   -
schema1_viewer             data_source1                   -
schema1_viewer             data_source1_alternative       -
schema1_viewer             data_source2                   -
schema1_editor             data_source1                   -
schema1_editor             data_source1_alternative       -
schema1_editor             data_source2                   -
schema1_admin              data_source1                   -
schema1_admin              data_source1_alternative       -
schema1_admin              data_source2                   -
schema1_viewer_group_user  data_source1                   -
schema1_viewer_group_user  data_source1_alternative       -
schema1_viewer_group_user  data_source2                   -
schema1_editor_group_user  data_source1                   -
schema1_editor_group_user  data_source1_alternative       -
schema1_editor_group_user  data_source2                   -
schema1_admin_group_user   data_source1                   -
schema1_admin_group_user   data_source1_alternative       -
schema1_admin_group_user   data_source2                   -
regular_user               data_source1                   -
regular_user               data_source1_alternative       -
regular_user               data_source2                   -
"""))
    def test_data_source_edit_view(
        self,
        client,
        dataset_test_data,
        user_key,
        data_source_key,
        access_allowed,
        get_in_admin_context
    ):
        """Test access to data source edit view."""
        data = dataset_test_data
        user = data[user_key]
        data_source = data[data_source_key]

        view = DataSourceViewSet().edit_view
        url = reverse('datasets_datasource:edit', args=[data_source.id])
        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': data_source.id})
            assert response.status_code == 200
            return
        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': data_source.id})

    @pytest.mark.parametrize(*parse_table("""
user_key                   data_source_key                access_allowed

superuser                  data_source1_alternative       +
superuser                  data_source2                   +
admin_user                 data_source1_alternative       +
admin_user                 data_source2                   -
super_admin_user           data_source1_alternative       +
super_admin_user           data_source2                   -
reviewer_user              data_source1_alternative       -
reviewer_user              data_source2                   -
viewer_user                data_source1_alternative       -
viewer_user                data_source2                   -
schema1_viewer             data_source1_alternative       -
schema1_viewer             data_source2                   -
schema1_editor             data_source1_alternative       -
schema1_editor             data_source2                   -
schema1_admin              data_source1_alternative       -
schema1_admin              data_source2                   -
schema1_viewer_group_user  data_source1_alternative       -
schema1_viewer_group_user  data_source2                   -
schema1_editor_group_user  data_source1_alternative       -
schema1_editor_group_user  data_source2                   -
schema1_admin_group_user   data_source1_alternative       -
schema1_admin_group_user   data_source2                   -
regular_user               data_source1_alternative       -
regular_user               data_source2                   -
"""))
    def test_data_source_delete_view(
        self, client, dataset_test_data, user_key, data_source_key, access_allowed, get_in_admin_context
    ):
        """Test access to data source delete view."""
        data = dataset_test_data
        user = data[user_key]
        data_source = data[data_source_key]

        view = DataSourceViewSet().delete_view
        url = reverse('datasets_datasource:delete', args=[data_source.id])
        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': data_source.id})
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': data_source.id})

    @pytest.mark.parametrize(*parse_table("""
user_key          access_allowed

superuser         +
admin_user        +
super_admin_user  +
reviewer_user     -
viewer_user       -
regular_user      -
"""))
    def test_data_source_create_view(self, client, dataset_test_data, user_key, access_allowed, get_in_admin_context):
        """Test access to data source create view."""
        data = dataset_test_data
        user = data[user_key]

        view = DataSourceViewSet().add_view
        url = reverse('datasets_datasource:add')

        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'])
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'])
