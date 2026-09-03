"""Tests for the ``repair_instance_page_slugs`` command."""

from __future__ import annotations

import re
from io import StringIO

from django.core.management import call_command
from django.core.management.base import CommandError
from wagtail.models import Locale, Page

import pytest

from nodes.models import InstanceConfig
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def _root_page(ic: InstanceConfig, slug: str) -> Page:
    """Give the instance a root page under the Wagtail root, as ``_create_default_pages`` would."""
    from pages.models import InstanceRootPage

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    page = wagtail_root.add_child(instance=InstanceRootPage(title=ic.name, slug=slug, url_path=''))
    page.add_child(instance=InstanceRootPage(title='Actions', slug='actions'))
    InstanceConfig.objects.filter(pk=ic.pk).update(root_page=page)
    ic.refresh_from_db()
    return page


def _exposed_then_looked_up(ic: InstanceConfig, page: Page) -> Page | None:
    """
    Replay what the two resolvers do, which is where the 404 came from.

    ``PathsPage.resolve_url_path`` strips ``^/<instance-id>(-[0-9]+)?/`` and
    ``Query.resolve_page`` prepends the root's ``url_path`` to what it is handed.
    """
    exposed = re.sub('^/%s(-[0-9]+)?/' % re.escape(ic.identifier), '/', page.url_path)
    if len(exposed) > 1:
        exposed = exposed.rstrip('/')
    root = ic.root_page
    assert root is not None
    return Page.objects.filter(url_path=root.url_path.rstrip('/') + exposed + '/').first()


def test_mismatched_root_slug_makes_subpages_unresolvable() -> None:
    """The bug itself: the front page survives an identifier/slug mismatch, the children do not."""
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')  # as a rename leaves it
    child = root.get_children().get(slug='actions')

    assert _exposed_then_looked_up(ic, child) is None

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())

    ic.refresh_from_db()
    child.refresh_from_db()
    assert _exposed_then_looked_up(ic, child) is not None


def test_swapping_two_instances_needs_the_temporary_slug_pass() -> None:
    """
    The case a one-pass rename cannot do.

    Root pages are siblings, and Wagtail requires the slug to be unique among siblings, so
    the instance being retired still holds the slug the new one needs.
    """
    new = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    old = InstanceConfigFactory.create(identifier='longmont-2021', name='Longmont 2021')
    _root_page(new, slug='longmont-dev')
    _root_page(old, slug='longmont')  # occupies the slug `new` has to take

    call_command('repair_instance_page_slugs', 'longmont', 'longmont-2021', '--apply', stdout=StringIO())

    for ic in (new, old):
        ic.refresh_from_db()
        root = ic.root_page
        assert root is not None
        assert root.slug == ic.identifier
        assert root.url_path == f'/{ic.identifier}/'
        child = root.get_children().get(slug='actions')
        assert child.url_path == f'/{ic.identifier}/actions/'
        assert _exposed_then_looked_up(ic, child) is not None


def test_translated_root_page_gets_a_numeric_suffix() -> None:
    """``resolve_url_path`` strips ``<id>-<n>`` too, so the translation has to follow the rename."""
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')
    locale, _ = Locale.objects.get_or_create(language_code='es-US')
    translation = root.copy_for_translation(locale)
    translation.slug = 'longmont-dev-1'
    translation.save()

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())

    translation.refresh_from_db()
    assert translation.slug == 'longmont-1'
    # The suffix is only required to be digits; the regex accepts any number.
    assert re.match(r'^/longmont-[0-9]+/$', translation.url_path)


def test_plan_writes_nothing_and_is_idempotent() -> None:
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')

    out = StringIO()
    call_command('repair_instance_page_slugs', 'longmont', stdout=out)
    assert 'Plan only' in out.getvalue()
    root.refresh_from_db()
    assert root.slug == 'longmont-dev'

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())
    out = StringIO()
    call_command('repair_instance_page_slugs', 'longmont', stdout=out)
    assert 'Nothing to do' in out.getvalue()


def test_unknown_instance_is_refused() -> None:
    with pytest.raises(CommandError, match='No instance'):
        call_command('repair_instance_page_slugs', 'no-such-instance', stdout=StringIO())
