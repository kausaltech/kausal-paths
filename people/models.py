from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel
from modeltrans.manager import MultilingualQuerySet
from wagtail.search import index

from kausal_common.datasets.models import DatasetSchema
from kausal_common.models.types import FK, M2M, MLModelManager, ModelManager
from kausal_common.people.models import BasePerson, create_permission_membership_models

from paths.types import PathsModel, PathsQuerySet

from orgs.models import Organization
from users.models import User

if TYPE_CHECKING:
    from paths.types import PathsAdminRequest

    from nodes.models import InstanceConfig

    from .permissions import PersonGroupPermissionPolicy


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


class PersonGroupQuerySet(PathsQuerySet['PersonGroup']):
    pass


_PersonGroupManager = models.Manager.from_queryset(PersonGroupQuerySet)
class PersonGroupManager(ModelManager['PersonGroup', PersonGroupQuerySet], _PersonGroupManager):
    """Model manager for PersonGroup."""
del _PersonGroupManager


class PersonGroup(PathsModel, ClusterableModel):
    """
    Group of persons for various purposes such as assigning permissions on certain models or model instances.

    In contrast to Django groups, names don't have to be globally unique.
    """

    instance: FK[InstanceConfig] = models.ForeignKey(
        'nodes.InstanceConfig', on_delete=models.CASCADE, related_name='person_groups'
    )
    name = models.CharField(max_length=200)
    persons: M2M[Person, PersonGroupMember] = models.ManyToManyField(
        Person,
        through='PersonGroupMember',
        related_name='person_groups',
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['instance', 'name'],
                name='unique_person_group_name_per_instance',
            ),
        ]

    @classmethod
    def permission_policy(cls) -> PersonGroupPermissionPolicy:
        from .permissions import PersonGroupPermissionPolicy
        return PersonGroupPermissionPolicy()

    def __str__(self) -> str:
        return self.name


class PersonGroupMember(models.Model):
    group = ParentalKey(PersonGroup, on_delete=models.CASCADE, related_name='persons_edges')
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='groups_edges')

    class Meta:
        verbose_name = _('Group member')
        verbose_name_plural = _('Group members')

    def __str__(self) -> str:
        return f'{self.person} ∈ {self.group}'


# Create permission membership models here, in the `people` app, since they will be part of this app. If you call
# `create_permission_membership_models` in a different app, `shell_plus` will get confused.
DatasetSchemaGroupPermission, DatasetSchemaPersonPermission = create_permission_membership_models(DatasetSchema)
