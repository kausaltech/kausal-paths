from __future__ import annotations

from django.urls import reverse
from wagtail.admin.admin_url_finder import AdminURLFinder, register_admin_url_finder
from wagtail.models import ReferenceIndex

from . import (
    choosers,  # noqa: F401
    node_admin,
)
from .models import NodeDataset

# Register NodeDataset so Wagtail tracks references from it to Dataset
ReferenceIndex.register_model(NodeDataset)


class NodeDatasetAdminURLFinder(AdminURLFinder):
    """Custom URL finder that returns the Node's edit URL for NodeDataset objects."""

    def get_edit_url(self, instance):
        if isinstance(instance, NodeDataset):

            # Return the edit URL for the Node instead
            return reverse(node_admin.NodeViewSet().get_url_name('edit'), args=[instance.node.pk])
        return None


register_admin_url_finder(NodeDataset, NodeDatasetAdminURLFinder)
