from django.utils.translation import gettext_lazy as _
from wagtail import hooks

from generic_chooser.views import ModelChooserMixin, ModelChooserViewSet
from generic_chooser.widgets import AdminChooser
from nodes.models import NodeConfig
from paths.types import PathsAdminRequest


class NodeChooserMixin(ModelChooserMixin):
    order_by: str | list[str]
    request: PathsAdminRequest

    def get_unfiltered_object_list(self):
        instance = self.request.admin_instance
        objects = NodeConfig.objects.filter(instance=instance)
        if self.order_by:
            if isinstance(self.order_by, str):
                objects = objects.order_by(self.order_by)
            else:
                objects = objects.order_by(*self.order_by)
        return objects

    def user_can_create(self, user):
        return False


class NodeChooserViewSet(ModelChooserViewSet):
    chooser_mixin_class = NodeChooserMixin

    icon = 'circle-nodes'
    model = NodeConfig
    page_title = _("Choose a node")
    per_page = 30
    fields = ['name', 'identifier']


class NodeChooser(AdminChooser):
    choose_one_text = _('Choose a node')
    choose_another_text = _('Choose another node')
    model = NodeConfig
    choose_modal_url_name = 'node_chooser:choose'


@hooks.register('register_admin_viewset')
def register_action_chooser_viewset():
    return NodeChooserViewSet('node_chooser', url_prefix='node-chooser')
