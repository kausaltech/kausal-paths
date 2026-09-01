from __future__ import annotations

import pytest

pytestmark = pytest.mark.django_db


# FIXME: Doesn't work because `AdminMiddleware` doesn't call `set_admin_instance()` because the existing
# `InstanceConfig` has no `site`.
# @pytest.mark.parametrize(('user__is_staff', 'user__is_superuser'), [(True, False), (True, True)])
# def test_no_access_for_non_staff_user(user, client):
#     # Prevent login if `is_staff` is false even for superusers.
#     client.force_login(user)
#     response = client.get(reverse('wagtailadmin_home'), follow=True)
#     assertContains(response, 'You do not have permission to access the admin')


# TODO: Tests that log in successfully
