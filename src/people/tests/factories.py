from factory import Sequence, SubFactory
from factory.django import DjangoModelFactory

from orgs.tests.factories import OrganizationFactory
from people.models import Person


class PersonFactory(DjangoModelFactory[Person]):
    class Meta:
        model = Person

    first_name = 'John'
    last_name = 'Frum'
    email = Sequence(lambda i: f'person{i}@example.com')
    organization = SubFactory(OrganizationFactory)  # type: ignore[var-annotated]
    user: None  # will be created by Person.save() because it calls Person.create_corresponding_user()
