from django.db import migrations, models
from django.utils.translation import gettext_lazy as _


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0043_alter_datasetport_options_datasetport_dataset_index'),
    ]

    operations = [
        migrations.AddField(
            model_name='instanceconfig',
            name='is_locked',
            field=models.BooleanField(
                default=False,
                help_text=_('Whether end-user mutation surfaces should treat this instance as read-only.'),
            ),
        ),
    ]
