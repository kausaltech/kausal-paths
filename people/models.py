from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from django.db.models import Q
from django.utils.translation import gettext_lazy as _
from modeltrans.manager import MultilingualQuerySet

from kausal_common.models.types import MLModelManager
from kausal_common.people.models import BasePerson

from orgs.models import Organization
from users.models import User

from wagtail.search import index

if TYPE_CHECKING:
    from django.db import models

    from paths.types import PathsAdminRequest

    from nodes.models import InstanceConfig


class PersonQuerySet(MultilingualQuerySet['Person']):
    def available_for_instance(self, instance: InstanceConfig):
        related = Organization.objects.filter(id=instance.organization_id)
        # TODO: Replace with the following if / when we add `related_organizations` to InstanceConfig
        # related = Organization.objects.filter(id=instance.organization_id) | instance.related_organizations.all()
        q = Q(pk__in=[])  # always false; Q() doesn't cut it; https://stackoverflow.com/a/39001190/14595546
        for org in related:
            q |= Q(organization__path__startswith=org.path)
        return self.filter(q)


if TYPE_CHECKING:
    _PersonManager = models.Manager.from_queryset(PersonQuerySet)
    class PersonManager(MLModelManager['Person', PersonQuerySet], _PersonManager): ...  # pyright: ignore
    del _PersonManager
else:
    PersonManager = MLModelManager.from_queryset(PersonQuerySet)


class Person(BasePerson):
    objects: ClassVar[PersonManager] = PersonManager()  # pyright: ignore

    search_fields = BasePerson.search_fields + [
        index.SearchField('first_name'),
        index.SearchField('last_name'),
        index.SearchField('email'),
        index.SearchField('title'),
        index.FilterField('path'),
    ]
    class Meta:
        verbose_name = _('Person')
        verbose_name_plural = _('People')

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    def download_avatar(self):
        # Since this is a base implementation, we'll return None
        # Subclasses can override this to implement actual avatar downloading
        return None

    @override
    def get_avatar_url(self, request: PathsAdminRequest, size: str | None = None) -> str | None:
        # Return the URL of the person's image if it exists
        if self.image:
            return self.image.url
        return None

    def create_corresponding_user(self):
        # Get or create a user based on the person's email
        if not self.email:
            return None

        user, created = User.objects.get_or_create(
            email__iexact=self.email,
            defaults={
                'email': self.email,
                'first_name': self.first_name,
                'last_name': self.last_name,
            }
        )
        return user

    def visible_for_user(self, user, **kwargs) -> bool:
        # By default, make the person visible to all authenticated users
        # and to the person themselves
        if not user.is_authenticated:
            return False

        # Person is always visible to themselves
        if user == self.user:
            return True

        # Person is visible to users in the same organization
        if user.person and user.person.organization == self.organization:
            return True

        return False
