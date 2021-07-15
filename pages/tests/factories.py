from factory import Sequence
from factory.django import DjangoModelFactory

from pages.models import InstanceContent


class InstanceContentFactory(DjangoModelFactory):
    class Meta:
        model = InstanceContent

    identifier = Sequence(lambda i: f'instance-content{i}')
    lead_title = "lead title"
    lead_paragraph = "Lead paragraph"
