"""
End-to-end check that the page API's two directions agree over GraphQL.

``pages`` hands out ``urlPath`` and ``page(path:)`` takes it back. Those are resolved by
different code in different modules, and the bug they caused twice was precisely that they
disagreed -- so the test that matters is the round trip through the real schema, with a root
page slug that does not match the instance identifier.
"""

from __future__ import annotations

from wagtail.models import Page

import pytest

from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

PAGES_QUERY = """
    query {
      pages {
        title
        urlPath
      }
    }
"""

PAGE_QUERY = """
    query($path: String!) {
      page(path: $path) {
        title
        urlPath
      }
    }
"""


def _give_root_page(ic: InstanceConfig, slug: str) -> Page:
    """Attach a two-level page tree to the instance, rooted on a page with the given slug."""
    from pages.models import InstanceRootPage

    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    root = wagtail_root.add_child(instance=InstanceRootPage(title='Emissions', slug=slug, url_path=''))
    root.add_child(instance=InstanceRootPage(title='Actions', slug='actions'))
    InstanceConfig.objects.filter(pk=ic.pk).update(root_page=root)
    ic.refresh_from_db()
    return root


@pytest.mark.parametrize('slug_suffix', ['', '-dev', '-2021'])
def test_every_page_the_api_exposes_can_be_looked_back_up(
    graphql_client_query_data, instance_config: InstanceConfig, slug_suffix: str
) -> None:
    """
    The contract the front end relies on, over the real schema.

    Two of these already worked before the prefix was unified: ``''`` because the slug
    matched the identifier, and ``'-2021'`` because ``-[0-9]+`` is exactly what the old
    regex's optional numeric-suffix branch stripped. They are kept as no-regression cases.

    ``'-dev'`` is the one that broke -- a non-numeric suffix the regex did not match -- and
    it 404ed every child while the front page kept resolving.
    """
    _give_root_page(instance_config, slug=f'{instance_config.identifier}{slug_suffix}')

    exposed = graphql_client_query_data(PAGES_QUERY)['pages']
    assert {page['title'] for page in exposed} == {'Emissions', 'Actions'}

    for page in exposed:
        looked_up = graphql_client_query_data(PAGE_QUERY, variables={'path': page['urlPath']})['page']
        assert looked_up is not None, f'{page["title"]!r} was handed out at {page["urlPath"]!r} but does not resolve'
        assert looked_up['title'] == page['title']


def test_the_exposed_paths_are_relative_to_the_instance(graphql_client_query_data, instance_config: InstanceConfig) -> None:
    """
    The slug must not leak into what the API hands out.

    This is the observable symptom the front end reported: a nav link to
    ``/longmont-dev`` instead of ``/``.
    """
    _give_root_page(instance_config, slug=f'{instance_config.identifier}-dev')

    exposed = graphql_client_query_data(PAGES_QUERY)['pages']
    by_title = {page['title']: page['urlPath'] for page in exposed}
    assert by_title == {'Emissions': '/', 'Actions': '/actions'}


def test_a_path_that_does_not_exist_still_resolves_to_null(graphql_client_query_data, instance_config: InstanceConfig) -> None:
    """The fix must not turn a genuine miss into a match."""
    _give_root_page(instance_config, slug=f'{instance_config.identifier}-dev')

    assert graphql_client_query_data(PAGE_QUERY, variables={'path': '/nope'})['page'] is None
    # The absolute Wagtail path is not a valid input and must not resolve either.
    absolute = graphql_client_query_data(PAGE_QUERY, variables={'path': f'/{instance_config.identifier}-dev/actions'})
    assert absolute['page'] is None
