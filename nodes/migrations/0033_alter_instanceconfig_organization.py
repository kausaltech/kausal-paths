import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0032_create_instanceconfig_organizations'),
    ]

    operations = [
        migrations.AlterField(
            model_name='instanceconfig',
            name='organization',
            field=models.ForeignKey(help_text='The main organization for the instance', on_delete=django.db.models.deletion.PROTECT, related_name='instances', to='orgs.organization', verbose_name='organization'),
        ),
    ]
