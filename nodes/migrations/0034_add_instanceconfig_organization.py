import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0033_alter_instanceconfig_name'),
        ('orgs', '0002_organizationmetadataadmin_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='instanceconfig',
            name='organization',
            field=models.ForeignKey(null=True, blank=True, on_delete=django.db.models.deletion.PROTECT, to='orgs.organization'),
        ),
    ]
