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


def _child(parent: Page, slug: str) -> Page:
    """Add a plain child page, the way the default page creation does."""
    from pages.models import InstanceRootPage

    return parent.add_child(instance=InstanceRootPage(title=slug.title(), slug=slug))


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


def test_disconnected_descendant_is_repaired_in_post_rename_terms() -> None:
    """
    A disconnected child's target has to be worked out after the rename, not before.

    ``Page.save()``'s cascade rewrites only paths sharing the old prefix, so a child Wagtail
    left orphaned survives the rename untouched. Computing its corrected path from the
    parent's *pre-rename* ``url_path`` writes the old prefix straight back, and the page
    stays unreachable behind a path that now looks plausible.
    """
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')
    child = root.get_children().get(slug='actions')
    Page.objects.filter(pk=child.pk).update(url_path='/orphaned/actions/')

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())

    child.refresh_from_db()
    assert child.url_path == '/longmont/actions/'
    assert _exposed_then_looked_up(ic, child) is not None


def test_disconnected_descendant_in_a_translated_tree_is_repaired() -> None:
    """Translated trees are served by the same two resolvers, so they are scanned the same way."""
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')
    locale, _ = Locale.objects.get_or_create(language_code='es-US')
    translation = root.copy_for_translation(locale)
    translation.slug = 'longmont-dev-1'
    translation.save()
    es_child = _child(translation, 'actions')
    Page.objects.filter(pk=es_child.pk).update(url_path='/orphaned-es/actions/')

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())

    es_child.refresh_from_db()
    assert es_child.url_path == '/longmont-1/actions/'


def test_revisions_are_rewritten_so_a_later_save_cannot_revert_the_rename() -> None:
    """
    A stale revision is one *Save draft* away from putting the old slug back.

    The Wagtail edit form populates from the latest revision, so the repair has to reach
    revisions and not only the live rows.
    """
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')
    child = root.get_children().get(slug='actions')
    root.save_revision()
    child.specific.save_revision()

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())

    root.refresh_from_db()
    child.refresh_from_db()
    assert root.latest_revision is not None
    assert root.latest_revision.content['slug'] == 'longmont'
    assert root.latest_revision.content['url_path'] == '/longmont/'
    assert child.latest_revision is not None
    assert child.latest_revision.content['url_path'] == '/longmont/actions/'


def test_stale_revisions_are_repaired_when_the_slugs_are_already_correct() -> None:
    """
    The state a rename done by hand leaves behind: live rows right, revisions not.

    The command has to find work here, otherwise it reports 'nothing to do' over an
    instance that is one admin save away from breaking again.
    """
    ic = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    root = _root_page(ic, slug='longmont-dev')
    root.save_revision()
    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())

    # Re-stale the revision the way an out-of-band rename would have left it.
    root.refresh_from_db()
    revision = root.latest_revision
    assert revision is not None
    revision.content['slug'] = 'longmont-dev'
    revision.content['url_path'] = '/longmont-dev/'
    revision.save(update_fields=['content'])

    out = StringIO()
    call_command('repair_instance_page_slugs', 'longmont', stdout=out)
    assert 'would put back the old slug' in out.getvalue()

    call_command('repair_instance_page_slugs', 'longmont', '--apply', stdout=StringIO())
    revision.refresh_from_db()
    assert revision.content['slug'] == 'longmont'
    assert revision.content['url_path'] == '/longmont/'


def test_swap_leaves_no_revision_holding_a_siblings_slug() -> None:
    """
    Longmont's real failure mode, one step worse than a revert.

    After the September 2026 rename, ``longmont-2021``'s Spanish root still had a revision
    saying ``longmont-1`` -- by then the live slug of ``longmont``'s Spanish root. Publishing
    it would have put two live siblings on one ``url_path``, and ``resolve_page`` returns
    whichever the queryset happens to yield first.
    """
    new = InstanceConfigFactory.create(identifier='longmont', name='Longmont')
    old = InstanceConfigFactory.create(identifier='longmont-2021', name='Longmont 2021')
    new_root = _root_page(new, slug='longmont-dev')
    old_root = _root_page(old, slug='longmont')
    locale, _ = Locale.objects.get_or_create(language_code='es-US')
    for root, slug in ((new_root, 'longmont-dev-1'), (old_root, 'longmont-1')):
        translation = root.copy_for_translation(locale)
        translation.slug = slug
        translation.save()
        translation.save_revision()

    call_command('repair_instance_page_slugs', 'longmont', 'longmont-2021', '--apply', stdout=StringIO())

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    tops = list(wagtail_root.get_children())
    assert len({page.slug for page in tops}) == len(tops), 'two root pages share a slug'
    for page in Page.objects.filter(depth__gte=2):
        for revision in page.revisions.all():
            assert revision.content['slug'] == page.slug, (
                f'page {page.pk} has a revision that would revert its slug to {revision.content["slug"]!r}'
            )
            assert revision.content['url_path'] == page.url_path
