from django.db import models
from treebeard.models import Node as Node


class MP_Node(Node):  # type: ignore[django-manager-missing]
    steplen: int
    alphabet: str
    node_order_by: list[str]
    path: models.CharField
    depth: models.PositiveIntegerField
    numchild: models.PositiveBigIntegerField
    gap: int
