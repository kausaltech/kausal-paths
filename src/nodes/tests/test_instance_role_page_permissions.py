"""
Instance role groups and the Wagtail page permissions they need.

The instance Admin and Super Admin roles grant every Wagtail page permission on the instance's
page tree. Wagtail only consults `GroupPagePermission` rows for that, so the role groups need rows
on the instance root page -- and on each translated root page, because a translation is a sibling
subtree at depth 2, not a descendant of the primary-language root, and so does not inherit them.

Instance routing used to go through a Wagtail `Site`, and the roles found the root page via
`site.root_page`. Once instances stored their root page directly, the roles found no site and
silently stopped writing page permissions, so a group created after that change had none and its
members were refused on every page in the admin.
"""

from typing import TYPE_CHECKING

from wagtail.models import PAGE_PERMISSION_TYPES, GroupPagePermission, Locale, Page

import pytest

from nodes.tests.factories import InstanceConfigFactory
from pages.models import InstanceRootPage, StaticPage
from users.models import User
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.contrib.auth.models import Group

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

ALL_PAGE_PERMS = {codename for codename, *_ in PAGE_PERMISSION_TYPES}


def _page_perm_codenames(group: Group | None, page: Page) -> set[str]:
    assert group is not None
    perms = GroupPagePermission.objects.filter(group=group, page=page)
    return set(perms.values_list('permission__codename', flat=True))


@pytest.fixture
def translated_instance() -> tuple[InstanceConfig, Page, Page, Page]:
    """Build an instance with a root page, a child page and a Spanish translation of the root."""
    ic = InstanceConfigFactory.create(
        identifier='role-perms',
        name='Role perms instance',
        primary_language='en',
        other_languages=['es-US'],
    )
    english, _ = Locale.objects.get_or_create(language_code='en')
    spanish, _ = Locale.objects.get_or_create(language_code='es-US')

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    root = wagtail_root.add_child(instance=InstanceRootPage(locale=english, title='Role perms', slug='role-perms', body='[]'))
    ic.root_page = root
    ic.save(update_fields=['root_page'])
    child = root.add_child(instance=StaticPage(locale=english, title='Emissions', slug='emissions'))

    spanish_root = root.copy_for_translation(spanish)
    spanish_root.save_revision().publish()
    return ic, root, child, spanish_root


def test_admin_groups_get_all_page_permissions_on_root_and_translations(
    translated_instance: tuple[InstanceConfig, Page, Page, Page],
) -> None:
    ic, root, _child, spanish_root = translated_instance

    ic.create_or_update_instance_groups()
    ic.refresh_from_db()

    for group in (ic.admin_group, ic.super_admin_group):
        assert _page_perm_codenames(group, root) == ALL_PAGE_PERMS
        assert _page_perm_codenames(group, spanish_root) == ALL_PAGE_PERMS


def test_read_only_groups_get_no_page_permissions(
    translated_instance: tuple[InstanceConfig, Page, Page, Page],
) -> None:
    ic, _root, _child, _spanish_root = translated_instance

    ic.create_or_update_instance_groups()
    ic.refresh_from_db()

    for group in (ic.viewer_group, ic.reviewer_group):
        assert group is not None
        assert not GroupPagePermission.objects.filter(group=group).exists()


def test_updating_instance_groups_is_idempotent_and_replaces_stale_rows(
    translated_instance: tuple[InstanceConfig, Page, Page, Page],
) -> None:
    ic, root, _child, spanish_root = translated_instance

    ic.create_or_update_instance_groups()
    ic.refresh_from_db()
    admin_group = ic.admin_group
    assert admin_group is not None
    # Leave a group with a partial set, as an older version of the roles might have.
    GroupPagePermission.objects.filter(group=admin_group, page=root, permission__codename='publish_page').delete()
    row_count = GroupPagePermission.objects.count() + 1

    ic.create_or_update_instance_groups()
    ic.create_or_update_instance_groups()

    assert _page_perm_codenames(admin_group, root) == ALL_PAGE_PERMS
    assert _page_perm_codenames(admin_group, spanish_root) == ALL_PAGE_PERMS
    assert GroupPagePermission.objects.count() == row_count


def test_super_admin_can_edit_instance_pages(
    translated_instance: tuple[InstanceConfig, Page, Page, Page],
) -> None:
    ic, _root, child, spanish_root = translated_instance
    user = UserFactory.create(is_staff=False, is_superuser=False)

    ic.permission_policy().super_admin_role.assign_user(ic, user)

    user = User.objects.get(pk=user.pk)
    assert child.permissions_for_user(user).can_edit()
    assert spanish_root.permissions_for_user(user).can_edit()


def test_default_content_grants_admin_page_permissions(instance_config: InstanceConfig) -> None:
    """A new instance gets its root page before its groups are granted permissions on it."""
    ic = instance_config
    assert ic.root_page is None

    ic.create_default_content()
    ic.refresh_from_db()

    assert ic.root_page is not None
    assert _page_perm_codenames(ic.admin_group, ic.root_page) == ALL_PAGE_PERMS
    assert _page_perm_codenames(ic.super_admin_group, ic.root_page) == ALL_PAGE_PERMS
