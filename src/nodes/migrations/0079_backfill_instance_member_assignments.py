"""Preserve existing municipal group memberships as member assignments."""

from django.db import migrations


def backfill(apps, schema_editor):
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    Assignment = apps.get_model('nodes', 'InstanceMemberAssignment')
    User = apps.get_model('users', 'User')
    for instance in InstanceConfig.objects.all().iterator():
        roles = {}
        for field, role in (
            ('viewer_group_id', 'viewer'),
            ('reviewer_group_id', 'reviewer'),
            ('editor_group_id', 'editor'),
            ('admin_group_id', 'admin'),
        ):
            group_id = getattr(instance, field)
            if group_id is None:
                continue
            for user_id in User.objects.filter(groups__id=group_id).values_list('pk', flat=True):
                roles[user_id] = role
        if instance.owned_by_id is not None:
            roles[instance.owned_by_id] = 'admin'
        Assignment.objects.bulk_create(
            [Assignment(instance_config_id=instance.pk, user_id=user_id, role=role) for user_id, role in roles.items()],
            ignore_conflicts=True,
        )


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0078_instanceconfig_editor_group_instanceinvitation_role_and_more'),
    ]

    operations = [migrations.RunPython(backfill, migrations.RunPython.noop)]
