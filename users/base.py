from __future__ import annotations

import base64
from uuid import UUID, uuid4

from django.contrib.auth.models import AbstractUser as DjangoAbstractUser, UserManager as DjangoUserManager
from django.db import models

from social_django.models import UserSocialAuth


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


class UserManager(DjangoUserManager):
    def create_superuser(self, username=None, email=None, password=None, **extra_fields):
        uuid = uuid4()
        if not username:
            username = uuid_to_username(uuid)
        extra_fields['uuid'] = uuid
        return super().create_superuser(username, email, password, **extra_fields)

    def get_by_natural_key(self, uuid):
        return self.get(uuid=uuid)


class AbstractUser(DjangoAbstractUser):
    uuid = models.UUIDField(unique=True)

    objects = UserManager()

    social_auth: models.manager.RelatedManager[UserSocialAuth]

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
            return '{0} {1}'.format(self.first_name, self.last_name).strip()
        else:
            return self.email

    def get_short_name(self):
        if self.first_name:
            return self.first_name
        return self.email

    def get_username(self):
        if not self.username or self.username.startswith('u-'):
            return self.email
        return self.username

    def natural_key(self):
        return (str(self.uuid),)

    def __str__(self):
        if self.first_name and self.last_name:
            return '%s %s (%s)' % (self.last_name, self.first_name, self.email)
        else:
            return self.email

    class Meta:
        abstract = True
        ordering = ('id',)
