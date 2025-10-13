from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING, ClassVar, override

from django.db import models
from django.db.models import Q
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel
from modeltrans.manager import MultilingualQuerySet
from wagtail.search import index

from loguru import logger

from kausal_common.datasets.models import DatasetSchema

# from wagtail.images.rect import Rect
from kausal_common.models.permissions import PermissionedModel
from kausal_common.models.types import FK, M2M, MLModelManager, ModelManager
from kausal_common.people.models import BasePerson, create_permission_membership_models

from paths.types import PathsModel, PathsQuerySet

from orgs.models import Organization
from users.models import User

if TYPE_CHECKING:
    from django.http import HttpRequest

    from kausal_common.models.permission_policy import ModelPermissionPolicy

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


class Person(PermissionedModel, BasePerson):
    objects: ClassVar[PersonManager] = PersonManager()  # pyright: ignore

    search_fields = BasePerson.search_fields + [
        index.SearchField('first_name'),
        index.SearchField('last_name'),
        index.SearchField('email'),
        index.SearchField('title'),
    ]
    class Meta:
        verbose_name = _('Person')
        verbose_name_plural = _('People')

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    def __rich_repr__(self):
        yield 'first_name', self.first_name
        yield 'last_name', self.last_name
        yield 'email', self.email


    def download_avatar(self):
        # Since this is a base implementation, we'll return None
        # Subclasses can override this to implement actual avatar downloading
        return None

    @override
    def get_avatar_url(self, request: HttpRequest | None = None, size: str | None = None) -> str | None:
        if not self.image:
            return None
        try:
            with self.image.open():
                pass
        except FileNotFoundError:
            logger.info('Avatar file for %s not found' % self)
            return None
        return self.image.url

    def avatar(self, request: HttpRequest | None = None) -> str:
        avatar_url = self.get_avatar_url(request, size='50x50')
        if not avatar_url:
            return ''
        return format_html('<span class="avatar"><img src="{}" /></span>', avatar_url)

    @override
    def create_corresponding_user(self):
        user = self.get_corresponding_user()
        email = self.email.lower()
        if user:
            created = False
            email_changed = user.email.lower() != email
            if email_changed:
                # If we change the email address to that of an existing deactivated user, we need to deactivate the
                # user with the old email address (done after this returns because it returns a user different from
                # `self.user`) and re-activate the user with the new email address (done further down in this method).
                with contextlib.suppress(User.DoesNotExist):
                    user = User.objects.get(email__iexact=email, is_active=False)
        else:
            user = User(
                email=email,
                uuid=uuid.uuid4(),
            )
            created = True
            email_changed = False

        if not created and not user.is_active:
            # Probably the user has been deactivated because the person has been deleted. Reactivate it.
            user.is_active = True
            reactivated = True
        else:
            reactivated = False

        set_password = created or reactivated or email_changed
        if set_password:
            # client = self.get_client_for_email_domain() # TODO: Add this if we implement clients
            # if client is not None and client.auth_backend:
            #     user.set_unusable_password()
            # else:
            user.set_password(str(uuid.uuid4()))

        user.first_name = self.first_name
        user.last_name = self.last_name
        user.email = email
        user.is_staff = True
        user.save()
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

    @classmethod
    def permission_policy(cls) -> ModelPermissionPolicy:
        from .permission_policy import PersonPermissionPolicy
        return PersonPermissionPolicy()


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
