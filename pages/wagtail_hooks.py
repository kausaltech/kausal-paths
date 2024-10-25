from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.db.models import Q
from wagtail import hooks
from wagtail.snippets.models import register_snippet

from pages.sitecontent import InstanceSiteContentViewSet

if TYPE_CHECKING:
    from wagtail.models import Page
    from wagtail.query import PageQuerySet

    from paths.types import PathsAdminRequest


@hooks.register('construct_explorer_page_queryset')
def filter_pages_to_admin_instance(
    parent_page: Any, pages: PageQuerySet[Page], request: PathsAdminRequest
) -> PageQuerySet[Page]:
    ic = request.admin_instance
    assert ic.site is not None

    q = Q()
    for page in ic.site.root_page.get_translations(inclusive=True):
        q |= pages.descendant_of_q(page, inclusive=True)
    return pages.filter(q)


@hooks.register('construct_page_chooser_queryset')
def filter_page_chooser_pages(pages: PageQuerySet[Page], request: PathsAdminRequest):
    return filter_pages_to_admin_instance(None, pages, request)


register_snippet(InstanceSiteContentViewSet)
