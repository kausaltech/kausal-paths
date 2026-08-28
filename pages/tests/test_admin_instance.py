"""
Resolving which instance a page belongs to, including a translated one.

Wagtail keeps one page object per locale, tied by `translation_key`, so an instance's page tree is
several sibling subtrees at depth 2 — one per language — and `InstanceConfig.root_page` points at the
primary language's row only. Asking "is this row the instance's root page?" therefore answers no for
every translated page, and the admin form that asked it that way raised `DoesNotExist` on all of them.

kausal-watch has the same tree shape and resolves it by page type plus `translation_key`
(`pages/models.py`, `AplansPage.plan`). This mirrors that.
"""

from __future__ import annotations

from wagtail.models import Locale, Page

import pytest

from nodes.models import InstanceConfig
from nodes.tests.factories import InstanceConfigFactory
from pages.models import InstanceRootPage, StaticPage, instance_config_for_page

pytestmark = pytest.mark.django_db


@pytest.fixture
def translated_instance():
    """Build an instance whose root page and one child both have a Spanish sibling."""
    config = InstanceConfigFactory.create(
        identifier='translated',
        name='A translated instance',
        primary_language='en',
        other_languages=['es-US'],
    )
    english, _ = Locale.objects.get_or_create(language_code='en')
    spanish, _ = Locale.objects.get_or_create(language_code='es-US')

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    root = wagtail_root.add_child(
        instance=InstanceRootPage(locale=english, title='An instance', slug='translated-instance', body='[]')
    )
    config.root_page = root
    config.save(update_fields=['root_page'])
    child = root.add_child(instance=StaticPage(locale=english, title='Emissions', slug='emissions'))

    spanish_root = root.copy_for_translation(spanish)
    spanish_root.save_revision().publish()
    spanish_child = child.copy_for_translation(spanish)
    spanish_child.save_revision().publish()
    return config, root, child, spanish_root, spanish_child


def test_the_primary_language_root_resolves(translated_instance):
    config, root, _child, _es_root, _es_child = translated_instance
    assert instance_config_for_page(root) == config


def test_a_primary_language_child_resolves(translated_instance):
    config, _root, child, _es_root, _es_child = translated_instance
    assert instance_config_for_page(child) == config


def test_a_translated_root_resolves_to_the_same_instance(translated_instance):
    # `InstanceConfig.root_page` names the English row, so this is the case that used to raise.
    config, _root, _child, es_root, _es_child = translated_instance
    assert instance_config_for_page(es_root) == config


def test_a_translated_child_resolves_to_the_same_instance(translated_instance):
    config, _root, _child, _es_root, es_child = translated_instance
    assert instance_config_for_page(es_child) == config


def test_a_page_outside_any_instance_tree_resolves_to_nothing():
    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    english, _ = Locale.objects.get_or_create(language_code='en')
    stray = wagtail_root.add_child(instance=StaticPage(locale=english, title='Stray', slug='stray-page'))

    assert instance_config_for_page(stray) is None


def test_the_form_resolves_a_translated_page(translated_instance):
    """The form property is what the admin actually calls, so it gets its own test."""
    from pages.models import PathsAdminPageForm

    config, _root, _child, _es_root, es_child = translated_instance
    form = PathsAdminPageForm.__new__(PathsAdminPageForm)
    form.instance = es_child
    assert form.admin_instance == config


def test_an_instance_config_with_no_root_page_is_not_matched(translated_instance):
    # A `filter(...).first()` on a null `root_page` would otherwise match an unrelated instance.
    config, root, _child, _es_root, _es_child = translated_instance
    InstanceConfigFactory.create(identifier='rootless', name='Rootless', primary_language='en')
    assert InstanceConfig.objects.filter(root_page__isnull=True).exists()
    assert instance_config_for_page(root) == config


@pytest.fixture
def instance_rooted_on_an_outcome_page():
    """
    Build an instance whose root page is *not* an `InstanceRootPage`.

    Which is how a real instance was found to be set up: its `root_page` is an `OutcomePage`. Watch can
    assume the subtree root is a `PlanRootPage` because `create_default_pages` makes one; Paths puts no
    such constraint on `InstanceConfig.root_page`, so resolution must not key on the page's type.
    """
    config = InstanceConfigFactory.create(
        identifier='outcome-rooted',
        name='Rooted on an outcome page',
        primary_language='en',
        other_languages=['es-US'],
    )
    english, _ = Locale.objects.get_or_create(language_code='en')
    spanish, _ = Locale.objects.get_or_create(language_code='es-US')
    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    root = wagtail_root.add_child(instance=StaticPage(locale=english, title='Emissions', slug='outcome-rooted-instance'))
    config.root_page = root
    config.save(update_fields=['root_page'])
    spanish_root = root.copy_for_translation(spanish)
    spanish_root.save_revision().publish()
    return config, root, spanish_root


def test_a_root_page_that_is_not_an_instance_root_page_still_resolves(instance_rooted_on_an_outcome_page):
    config, root, _spanish_root = instance_rooted_on_an_outcome_page
    assert instance_config_for_page(root) == config


def test_its_translation_resolves_too(instance_rooted_on_an_outcome_page):
    config, _root, spanish_root = instance_rooted_on_an_outcome_page
    assert instance_config_for_page(spanish_root) == config
