# Drop the legacy NodeEdge / DatasetPort tables.
#
# The unified NodeInputPortBinding table has been authoritative since the
# 0068/0069 flip; both legacy tables have been empty since 0069 truncated
# them, so this deletes no data. Change-log entries recorded against the
# legacy content types keep resolving through the shared binding UUID
# (see nodes/graphql/types/change_history.py).

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0071_action_group_uuid'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='nodeedge',
            name='created_by',
        ),
        migrations.RemoveField(
            model_name='nodeedge',
            name='from_node',
        ),
        migrations.RemoveField(
            model_name='nodeedge',
            name='instance',
        ),
        migrations.RemoveField(
            model_name='nodeedge',
            name='last_modified_by',
        ),
        migrations.RemoveField(
            model_name='nodeedge',
            name='latest_revision',
        ),
        migrations.RemoveField(
            model_name='nodeedge',
            name='to_node',
        ),
        migrations.DeleteModel(
            name='DatasetPort',
        ),
        migrations.DeleteModel(
            name='NodeEdge',
        ),
    ]
