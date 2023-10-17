from typing import Any
from wagtail.models import Page
from wagtail.query import PageQuerySet
from paths.types import PathsAdminRequest

from wagtail import hooks


@hooks.register('construct_explorer_page_queryset')
def filter_pages_to_admin_instance(
    parent_page: Any, pages: PageQuerySet[Page], request: PathsAdminRequest
) -> PageQuerySet[Page]:
    ic = request.admin_instance
    assert ic.site is not None
    instance_pages = ic.site.root_page.get_descendants(inclusive=True)
    return pages.filter(id__in=instance_pages)


@hooks.register('construct_page_chooser_queryset')
def filter_page_chooser_pages(pages: PageQuerySet[Page], request: PathsAdminRequest):
    return filter_pages_to_admin_instance(None, pages, request)
