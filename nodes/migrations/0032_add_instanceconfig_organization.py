import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0031_alter_nodeconfig_color'),
        ('orgs', '0002_organizationmetadataadmin_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='instanceconfig',
            name='organization',
            field=models.ForeignKey(help_text='The main organization for the instance', null=True, on_delete=django.db.models.deletion.PROTECT, related_name='instances', to='orgs.organization', verbose_name='organization'),

        ),
    ]
