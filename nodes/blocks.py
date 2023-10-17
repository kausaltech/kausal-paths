from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from wagtail.blocks import StructBlock, ChooserBlock


class NodeChooserBlock(ChooserBlock):
    class Meta:
        label = _('Node')

    @cached_property
    def target_model(self):
        from .models import NodeConfig
        return NodeConfig

    @cached_property
    def widget(self):
        from .choosers import NodeChooser
        return NodeChooser()

    def get_form_state(self, value):
        return self.widget.get_value_data(value)


class OutcomeBlock(StructBlock):
    outcome_node = NodeChooserBlock(label=_("Outcome node"))
