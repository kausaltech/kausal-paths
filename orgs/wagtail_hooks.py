# Parts adapted from https://posts-by.lb.ee/building-a-configurable-taxonomy-in-wagtail-django-94ca1080fb28
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from django.contrib.admin.utils import quote
from django.core.exceptions import ValidationError
from django.db.models import Q
from django.urls import URLPattern, path, reverse
from django.utils.translation import gettext_lazy as _, pgettext_lazy
from wagtail.admin.menu import AdminOnlyMenuItem
from wagtail.admin.panels import FieldPanel, InlinePanel, ObjectList, TabbedInterface
from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet
from wagtail.snippets.widgets import SnippetListingButton

from wagtailgeowidget import __version__ as wagtailgeowidget_version

from kausal_common.i18n.panels import TranslatedFieldPanel
from kausal_common.models.permission_policy import ModelPermissionPolicy, ObjectSpecificAction
from kausal_common.organizations.forms import NodeForm
from kausal_common.organizations.views import (
    OrganizationDeleteView,
    OrganizationEditView,
    OrganizationIndexView,
)

# from admin_site.utils import admin_req
# from admin_site.wagtail import CondensedInlinePanel
from kausal_common.people.chooser import PersonChooser

from paths.context import realm_context

from orgs.views import OrganizationCreateView
from users.models import User

from .models import Organization
from .views import CreateChildNodeView

if TYPE_CHECKING:
    from django.contrib.auth.models import AnonymousUser
    from wagtail.admin.panels.base import Panel
    from wagtail.core.models import Model

    from nodes.models import InstanceConfig


import logging

logger = logging.getLogger(__name__)

if int(wagtailgeowidget_version.split('.')[0]) >= 7:
    from wagtailgeowidget.panels import GoogleMapsPanel
else:
    from wagtailgeowidget.edit_handlers import GoogleMapsPanel


class CondensedInlinePanel[M: Model, RelatedM: Model](InlinePanel[M, RelatedM]):
    pass

class OrganizationPermissionPolicy(ModelPermissionPolicy):

    def user_has_permission(self, user: User | AnonymousUser, action: str) -> bool:
        assert isinstance(user, User)
        if user.is_superuser:
            return True
        # if action == 'view':
        #     return True
        # TODO: The following is the old logic, which we may reinstate when we
        # thought about how to handle permissions best.
        # person = user.get_corresponding_person()
        # return person and person.metadata_adminable_organizations.exists()
        # For now we allow general admins (for any plan) to create organizations.
        # if action == 'add':
        #     return user.is_general_admin_for_plan()
        # We cannot know if the user has other permissions to the instance
        # without knowing the instance. user_has_permission should be overridden
        # in relevant places to call user_has_permission_for_instance
        return False

    def user_has_permission_for_instance(self, user: User | AnonymousUser, action: str, instance: Organization) -> bool:
        assert isinstance(user, User)
        if user.is_superuser:
            return True

        # if action in ('change', 'delete'):
        #     return user.is_admin_for_instance(instance.instances)
        # if action == 'add':
        #     return user.is_admin_for_instance(instance.instances)


        return super().user_has_permission_for_instance(user, action, instance)

    def anon_has_perm(self, action: ObjectSpecificAction, obj: Any) -> bool:
        return False

    def construct_perm_q(self, user: User, action: ObjectSpecificAction) -> Q | None:
        if user.is_superuser:
            return Q()
        return Q(pk__in=[])

    def construct_perm_q_anon(self, action: ObjectSpecificAction) -> Q | None:
        return Q(pk__in=[])

    def user_can_create(self, user: User, context: Any) -> bool:
        return user.is_superuser

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: Any) -> bool:
        return user.is_superuser


class OrganizationForm(NodeForm):
    user: User

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('for_user')
        assert isinstance(user, User)
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_parent(self):
        parent = super().clean_parent()
        if self.instance._state.adding:
            return parent
        # If a user has edit access to an organization only because they can edit an ancestor, prevent them from losing
        # edit rights by moving it to a parent which they cannot edit (or make it a root). For now, only allow
        # superusers to set roots. (Only editable organizations are avaible as parent choices anyway.)
        if parent is None and not self.user.is_superuser:
            # On the other hand, allow direct metadata admins of a top level organizations to save the org when editing
            # if (
            #     self.instance.parent is None and
            #     OrganizationMetadataAdmin.objects
            #         .filter(person=self.user.person)
            #         .filter(organization=self.instance)
            #         .exists()
            # ):
            #     return parent
            # For now, allow for general plan admins
            # if self.instance.parent is None and self.user.is_general_admin_for_plan():
            #     return parent
            raise ValidationError(_("Creating organizations without a parent not allowed."), code='invalid_parent')
        return parent

    def save(self, *args, **kwargs):
        creating = self.instance._state.adding
        result = super().save(*args, **kwargs)
        if creating and self.instance.parent is None:
            # When creating a new root organization make sure the creator retains edit permissions
            self.instance.metadata_admins.add(self.user.person) # type: ignore[attr-defined]
        return result



class OrganizationViewSet(SnippetViewSet):
    model = Organization
    menu_label = _("Organizations")
    icon = 'kausal-organisations'
    menu_order = 301
    permission_policy = OrganizationPermissionPolicy(model)
    index_view_class = OrganizationIndexView
    add_view_class = OrganizationCreateView
    edit_view_class = OrganizationEditView
    delete_view_class = OrganizationDeleteView
    search_fields = ['name', 'abbreviation']
    list_display = ['name', 'parent','abbreviation']
    add_to_admin_menu = True
    menu_item_class = AdminOnlyMenuItem  # TODO: remove this line once permission policies are ready
    add_child_url_name = 'add_child'

    basic_panels = [
        TranslatedFieldPanel('name'),
        FieldPanel(
            # virtual field, needs to be specified in the form
            'parent', heading=pgettext_lazy('organization', 'Parent'),
        ),
        # FieldPanel('logo'),
        TranslatedFieldPanel('abbreviation'),
        FieldPanel('internal_abbreviation'),
        # Don't allow editing identifiers at this point
        # CondensedInlinePanel('identifiers', panels=[
        #     FieldPanel('namespace'),
        #     FieldPanel('identifier'),
        # ]),
        FieldPanel('description'),
        FieldPanel('url'),
        FieldPanel('email'),
        FieldPanel('primary_language', read_only=True),  # read-only for now because changes could cause trouble
        GoogleMapsPanel('location', permission='superuser'),
    ]

    permissions_panels: list[Panel] = [
        # CondensedInlinePanel(
        #     'organization_plan_admins',
        #     panels=[
        #         InvisiblePlanPanel('plan'),
        #         FieldPanel('person', widget=PersonChooser),
        #     ],
        #     heading=_("Plan admins"),
        #     help_text=_("People who can edit plan-specific content related to this organization"),
        # ),
        CondensedInlinePanel(
            'organization_metadata_admins',
            panels=[
                FieldPanel('person', widget=PersonChooser),
            ],
            heading=_("Metadata admins"),
            help_text=_("People who can edit data of this organization and suborganizations but no plan-specific "
                        "content"),
        ),
    ]

    @property
    def add_child_view(self):
        """Generate a class-based view to provide 'add child' functionality."""
        return self.construct_view(CreateChildNodeView, **self.get_add_view_kwargs())

    def get_urlpatterns(self) -> list[URLPattern]:
        urls =  super().get_urlpatterns()
        add_child_url = path(
            route=f'{self.add_child_url_name}/<str:parent_pk>/',
            view=self.add_child_view,
            name=self.add_child_url_name,
        )

        return urls + [
            add_child_url,
        ]

    def get_common_view_kwargs(self, **kwargs):
        return super().get_common_view_kwargs(
            add_child_url_name=self.get_url_name(self.add_child_url_name),
            **kwargs,
        )

    def get_edit_handler(self):
        tabs = [
            ObjectList(self.basic_panels, heading=_('Basic information')),
            ObjectList(self.permissions_panels, heading=_('Permissions')),
        ]
        return TabbedInterface(tabs, base_form_class=OrganizationForm).bind_to_model(self.model)

    def get_index_view_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        kwargs = super().get_index_view_kwargs(**kwargs)
        kwargs['view_set'] = self
        return kwargs

    def _get_edit_button(self, instance: Organization) -> SnippetListingButton:
        return SnippetListingButton(
            _("Edit"),
            url=reverse(self.get_url_name("edit"), args=(quote(instance.pk),)),
            icon_name="edit",
            attrs={
                "aria-label": _("Edit '%(title)s'") % {"title": str(instance)}
            },
            priority=10,
        )

    def _get_copy_button(self, instance: Organization) -> SnippetListingButton:
        return SnippetListingButton(
            _("Copy"),
            url=reverse(self.get_url_name("copy"), args=(quote(instance.pk),)),
            icon_name="copy",
            attrs={
                "aria-label": _("Copy '%(title)s'") % {"title": str(instance)}
            },
            priority=20,
        )

    def _get_delete_button(self, instance: Organization) -> SnippetListingButton:
        return SnippetListingButton(
            _("Delete"),
            url=reverse(self.get_url_name("delete"), args=(quote(instance.pk),)),
            icon_name="bin",
            attrs={
                "aria-label": _("Delete '%(title)s'") % {"title": str(instance)}
            },
            priority=30,
        )

    def _get_add_child_button(self, instance: Organization) -> SnippetListingButton:
        return SnippetListingButton(
            url=reverse(self.get_url_name(self.add_child_url_name), kwargs={'parent_pk': quote(instance.pk)}),
            label=_("Add suborganization"),
            icon_name='plus',
            attrs={'aria-label': _("Add suborganization")},
        )


    def get_index_view_buttons(self, user: User, instance: Organization, instance_config: InstanceConfig):
        """Get the buttons to show in the index view for an organization."""

        # The button definitions are done here to allow querying them through
        # GraphQL, as GraphQL has trouble fetching the buttons through
        # IndexView's get_list_more_buttons method where the definitions are
        # usually done. The GraphQL-queried buttons are used by the
        # JavaScript-implemented custom implementation of the index view.

        buttons = []

        # Basic buttons provided by Wagtail
        if self.permission_policy.user_has_permission_for_instance(user, "change", instance):
            buttons.append(self._get_edit_button(instance))
        if self.permission_policy.user_has_permission(user, "add"):
            buttons.append(self._get_copy_button(instance))
        if self.permission_policy.user_has_permission_for_instance(user, "delete", instance):
            buttons.append(self._get_delete_button(instance))

        # Show "add child" button
        # TODO: allow for organization metadata admins but without the huge
        # amount of db queries that iterating org.user_can_edit entails
        if user.user_is_admin_for_instance(instance):
            buttons.append(self._get_add_child_button(instance))

        return buttons

    def get_queryset(self, request):
        active_instance = realm_context.get().realm
        qs = Organization.objects.qs.available_for_instance(active_instance)
        return qs

# If kausal_watch_extensions is installed, an extended version of the view set is registered there
if not find_spec('kausal_paths_extensions'):
    register_snippet(OrganizationViewSet)
