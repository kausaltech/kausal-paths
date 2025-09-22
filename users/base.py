from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar, TypeVar
from uuid import UUID, uuid4

from django.contrib.auth.models import AbstractUser as DjangoAbstractUser, UserManager as DjangoUserManager
from django.db import models

from social_django.models import UserSocialAuth

if TYPE_CHECKING:
    from django.db.models.fields.related_descriptors import RelatedManager  # pyright: ignore
    from django.db.models.manager import RelatedManager  # type: ignore  # noqa


def uuid_to_username(uuid: UUID | str):
    """
    Convert UUID to username.

    >>> uuid_to_username('00fbac99-0bab-5e66-8e84-2e567ea4d1f6')
    'u-ad52zgilvnpgnduefzlh5jgr6y'

    >>> uuid_to_username(UUID('00fbac99-0bab-5e66-8e84-2e567ea4d1f6'))
    'u-ad52zgilvnpgnduefzlh5jgr6y'
    """

    uuid_data: bytes
    if isinstance(uuid, UUID):
        uuid_data = uuid.bytes
    else:
        uuid_data = UUID(uuid).bytes
    b32coded = base64.b32encode(uuid_data)
    return 'u-' + b32coded.decode('ascii').replace('=', '').lower()


def username_to_uuid(username: str):
    """
    Convert username to UUID.

    >>> username_to_uuid('u-ad52zgilvnpgnduefzlh5jgr6y')
    UUID('00fbac99-0bab-5e66-8e84-2e567ea4d1f6')
    """
    if not username.startswith('u-') or len(username) != 28:
        raise ValueError('Not an UUID based username: %r' % (username,))
    decoded = base64.b32decode(username[2:].upper() + '======')
    return UUID(bytes=decoded)


UMM = TypeVar('UMM', bound='AbstractUser')


class UserManager(DjangoUserManager[UMM]):
    def create_superuser(self, username=None, email=None, password=None, **extra_fields) -> UMM:
        uuid = uuid4()
        if not username:
            username = uuid_to_username(uuid)
        extra_fields['uuid'] = uuid
        return super().create_superuser(username, email, password, **extra_fields)


class AbstractUser(DjangoAbstractUser):
    uuid = models.UUIDField(unique=True)

    objects: ClassVar[UserManager] = UserManager()

    social_auth: RelatedManager[UserSocialAuth]

    def save(self, *args, **kwargs):
        self.clean()
        return super(AbstractUser, self).save(*args, **kwargs)

    def clean(self):
        self._make_sure_uuid_is_set()
        if not self.username:
            self.set_username_from_uuid()

    def _make_sure_uuid_is_set(self):
        if self.uuid is None:
            self.uuid = uuid4()

    def set_username_from_uuid(self):
        self._make_sure_uuid_is_set()
        self.username = uuid_to_username(self.uuid)

    def get_display_name(self):
        if self.first_name and self.last_name:
            return f'{self.first_name} {self.last_name}'.strip()
        return self.email

    def get_short_name(self):
        if self.first_name:
            return self.first_name
        return self.email

    def get_username(self):
        if not self.username or self.username.startswith('u-'):
            return self.email
        return self.username

    def __str__(self):
        if self.first_name and self.last_name:
            return '%s %s (%s)' % (self.last_name, self.first_name, self.email)
        return self.email

    class Meta:
        abstract = True
        ordering = ('id',)
