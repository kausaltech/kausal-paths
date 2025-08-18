from wagtail.rich_text import RichText

from factory import Sequence
from factory.django import DjangoModelFactory

from orgs.models import Organization


class OrganizationFactory(DjangoModelFactory[Organization]):
    class Meta:
        model = Organization

    name = Sequence(lambda i: f"Organization {i}")
    abbreviation = Sequence(lambda i: f'org{i}')
    description = RichText("<p>Description</p>")
    url = 'https://example.org'

    @classmethod
    def _create(cls, model_class, *args, **kwargs) -> Organization:  # noqa: ARG003
        parent = kwargs.pop('parent', None)
        node = Organization(**kwargs)
        if parent:
            return parent.add_child(instance=node)
        return Organization.add_root(instance=node)
