from __future__ import annotations

from django.db import transaction
from django.db.models.signals import m2m_changed, post_delete, post_save
from django.dispatch import receiver
from django.urls import reverse
from wagtail.admin.admin_url_finder import AdminURLFinder, register_admin_url_finder
from wagtail.models import ReferenceIndex

from . import (
    choosers,  # noqa: F401
    node_admin,
)
from .models import NodeConfig, NodeDataset

ReferenceIndex.register_model(NodeDataset)


@receiver(post_save, sender=NodeDataset)
def update_reference_index_on_save(sender, instance, **kwargs):
    with transaction.atomic():
        ReferenceIndex.create_or_update_for_object(instance)


@receiver(post_delete, sender=NodeDataset)
def update_reference_index_on_delete(sender, instance, **kwargs):
    ReferenceIndex.remove_for_object(instance)


@receiver(m2m_changed, sender=NodeConfig.datasets.through)
def update_reference_index_on_m2m_change(sender, instance, action, pk_set, **kwargs):
    if action == 'post_add':
        with transaction.atomic():
            for node_dataset in NodeDataset.objects.filter(node=instance, dataset_id__in=pk_set):
                ReferenceIndex.create_or_update_for_object(node_dataset)


class NodeDatasetAdminURLFinder(AdminURLFinder):
    """Custom URL finder that returns the Node's edit URL for NodeDataset objects."""

    def get_edit_url(self, instance):
        if isinstance(instance, NodeDataset):

            # Return the edit URL for the Node instead
            return reverse(node_admin.NodeViewSet().get_url_name('edit'), args=[instance.node.pk])
        return None


register_admin_url_finder(NodeDataset, NodeDatasetAdminURLFinder)
