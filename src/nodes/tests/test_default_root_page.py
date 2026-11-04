"""
Tests for finding an instance's existing root page rather than building a second one.

The lookup used to be ``home_pages.get(slug=self.identifier)``, so once the slug and the
identifier had drifted apart it found nothing and its caller created a duplicate root page
beside the real one. That orphan is half of what made both page-routing incidents hard to
unpick: ``augsburg-bisko`` ended up with a duplicate in July 2026, and a stale page was left
holding the slug ``longmont`` needed in September 2026.
"""

from __future__ import annotations

from wagtail.models import Page

import pytest

from nodes.models import InstanceConfig
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def _root_page(ic: InstanceConfig, slug: str) -> Page:
    from pages.models import InstanceRootPage

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    page = wagtail_root.add_child(instance=InstanceRootPage(title=ic.name, slug=slug, url_path=''))
    InstanceConfig.objects.filter(pk=ic.pk).update(root_page=page)
    ic.refresh_from_db()
    return page


def test_the_root_page_is_found_by_fk_when_the_slug_has_drifted() -> None:
    """A rename leaves the slug behind but never moves the ``root_page`` link."""
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    found = ic._find_existing_root_page(wagtail_root.get_children())

    assert found is not None
    assert found.pk == root.pk


def test_the_slug_fallback_still_works_without_a_root_page_link() -> None:
    """Instances predating the FK being populated have only the slug to go on."""
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont')
    InstanceConfig.objects.filter(pk=ic.pk).update(root_page=None)
    ic.refresh_from_db()

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    found = ic._find_existing_root_page(wagtail_root.get_children())

    assert found is not None
    assert found.pk == root.pk


def test_a_sibling_instances_root_page_is_not_claimed() -> None:
    """
    The lookup must not reach outside the instance it is called on.

    With the slug fallback in play, an instance with no ``root_page`` and no page of its own
    must come back empty rather than adopting a neighbour's.
    """
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    other = InstanceConfigFactory.create(identifier='longmont-2021', name='Longmont 2021')
    _root_page(other, slug='longmont-2021')

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    assert ic._find_existing_root_page(wagtail_root.get_children()) is None


def test_creating_the_instance_root_page_twice_does_not_duplicate_it() -> None:
    """
    ``_create_instance_root_page`` is reached on every sync, not only at creation.

    It used to ``add_child`` unconditionally, so a drifted slug meant a fresh orphan on each
    call.
    """
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')

    returned = ic._create_instance_root_page()

    assert returned.pk == root.pk
    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    # Scoped to this instance's own page: the Wagtail default "Welcome" page is a sibling.
    assert wagtail_root.get_children().filter(title=ic.name).count() == 1
