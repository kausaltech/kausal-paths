from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.utils.translation import gettext_lazy as _
from wagtail.models import PAGE_PERMISSION_CODENAMES, GroupPagePermission

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django_stubs_ext import StrPromise

    from nodes.models import InstanceConfig


ALL_MODEL_PERMS = ('view', 'change', 'delete', 'add')


def _get_perm_objs(model, perms) -> list[Permission]:
    content_type = ContentType.objects.get_for_model(model)
    perms = ['%s_%s' % (x, model._meta.model_name) for x in perms]
    perm_objs = Permission.objects.filter(content_type=content_type, codename__in=perms)
    return list(perm_objs)


def _model_perms(app_label: str, models: str | Sequence[str], perms: Sequence[str]) -> tuple[str, list[str]]:
    if isinstance(models, str):
        models = [models]

    return (app_label, ['%s_%s' % (p, m) for p in perms for m in models])


def _join_perms(model_perms: list[tuple[str, Sequence[str]]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for app_label, perms in model_perms:
        out[app_label] = out.get(app_label, set()).union(set(perms))
    return out


class Role:
    id: str
    name: StrPromise
    model_perms: dict[str, set[str]] = {}
    page_perms: set[str] = set()
    group_name: str
    group: Group

    @cached_property
    def perms(self):
        perms = Permission.objects.filter(content_type__app_label__in=self.model_perms.keys()).select_related('content_type')
        by_app: dict[str, dict[str, Permission]] = {}
        for perm in perms:
            ct = perm.content_type
            app_perms = by_app.setdefault(ct.app_label, {})
            app_perms[perm.codename] = perm
        return by_app

    def get_group_name(self, instance: InstanceConfig) -> str:
        return '%s %s' % (instance.name, self.group_name)

    def _update_model_perms(self, group: Group, instance: InstanceConfig) -> None:
        old_perms = set(group.permissions.all())
        new_perms = set()
        for app_label, perms in self.model_perms.items():
            for p in list(perms):
                new_perms.add(self.perms[app_label][p])

        if old_perms != new_perms:
            instance.log.info('Setting new %s permissions' % self.group_name)
            group.permissions.set(new_perms)

    @transaction.atomic()
    def _update_page_perms(self, group: Group, instance: InstanceConfig) -> None:
        if instance.site is None:
            return

        filt = dict(
            content_type__app_label='wagtailcore',
            content_type__model='page',
        )
        root_page = instance.site.root_page
        grp_perms = root_page.group_permissions.filter(group=group)
        old_perms = set(grp_perms.values_list('permission', flat=True))
        new_perms = set(Permission.objects.filter(**filt, codename__in=self.page_perms))
        if old_perms != new_perms:
            instance.log.info('Setting new %s page permissions' % self.group_name)
            grp_perms.delete()
            objs = [
                GroupPagePermission(
                    group=group,
                    page=root_page,
                    permission=perm,
                )
                for perm in new_perms
            ]
            GroupPagePermission.objects.bulk_create(objs)

    def create_group(self, instance: InstanceConfig):
        name = self.get_group_name(instance)
        if instance.admin_group is not None:
            group = instance.admin_group
            if group.name != name:
                group.name = name
                group.save(update_fields=['name'])
        else:
            group, _ = Group.objects.get_or_create(name=name)

        self._update_model_perms(group, instance)
        self._update_page_perms(group, instance)

        return group

    def update_instance(self, instance: InstanceConfig):
        group = self.create_group(instance)
        if instance.admin_group != group:
            instance.admin_group = group


class AdminRole(Role):
    id = 'admin'
    name = _('General admin')
    group_name = 'General admins'

    model_perms = _join_perms(
        [
            _model_perms('wagtailadmin', 'admin', ('access',)),
            _model_perms('wagtailcore', 'collection', ALL_MODEL_PERMS),
            _model_perms('wagtailimages', 'image', ALL_MODEL_PERMS),
            _model_perms('nodes', ('instanceconfig', 'nodeconfig'), ('view', 'change')),
            _model_perms(
                'datasets',
                (
                    'dataset',
                    'datasetcomment',
                    'datasetdimension',
                    'datasetdimensionselectedcategory',
                    'datasetmetric',
                    'datasetsourcereference',
                    'dimension',
                    'dimensioncategory',
                ),
                ALL_MODEL_PERMS,
            ),
        ]
    )

    page_perms = set(PAGE_PERMISSION_CODENAMES)
