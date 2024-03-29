from factory.django import DjangoModelFactory
from factory import Sequence


class UserFactory(DjangoModelFactory):
    class Meta:
        model = 'users.User'
        django_get_or_create = ('username',)

    username = Sequence(lambda i: f'user{i}')
    email = Sequence(lambda i: f'email.i{i}@nonexistent.tech')
    is_staff = False
    is_superuser = False
