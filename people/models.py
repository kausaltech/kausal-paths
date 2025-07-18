from __future__ import annotations

# import re
from typing import TYPE_CHECKING, ClassVar, override

from django.db.models import Q
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from modeltrans.manager import MultilingualQuerySet
from wagtail.search import index

# from easy_thumbnails.files import get_thumbnailer
from loguru import logger

# from wagtail.images.rect import Rect
from kausal_common.models.types import MLModelManager
from kausal_common.people.models import BasePerson

from orgs.models import Organization
from users.models import User

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
        # from kausal_common.model_images import determine_image_dim
        # Return the URL of the person's image if it exists
        if not self.image:
            return None

        try:
            with self.image.open():
                pass
        except FileNotFoundError:
            logger.info('Avatar file for %s not found' % self)
            return None

        # if size is None:
        url = self.image.url
        # else:
        #     m = re.match(r'(\d+)?(x(\d+))?', size)
        #     if not m:
        #         raise ValueError('Invalid size argument (should be "<width>x<height>")')
        #     width, _, height = m.groups()

        #     dim = determine_image_dim(self.image, width, height)

        #     tn_args: dict = {
        #         'size': dim,
        #     }
        #     if self.image_cropping:
        #         tn_args['focal_point'] = Rect(*[int(x) for x in self.image_cropping.split(',')])
        #         tn_args['crop'] = 30

        #     out_image = get_thumbnailer(self.image).get_thumbnail(tn_args)
        #     if out_image is None:
        #         return None
        #     url = out_image.url

        # if request:
        #     url = request.build_absolute_uri(url)
        return url


    def avatar(self, request: PathsAdminRequest | None = None) -> str:
        avatar_url = self.get_avatar_url(request, size='50x50')
        if not avatar_url:
            return ''
        return format_html('<span class="avatar"><img src="{}" /></span>', avatar_url)


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
