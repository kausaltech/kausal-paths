# Generated by Django 3.2.16 on 2023-03-03 08:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0013_datasource'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasource',
            name='instance',
            field=models.ForeignKey(default=1, editable=False, on_delete=django.db.models.deletion.CASCADE, related_name='data_sources', to='nodes.instanceconfig'),
            preserve_default=False,
        ),
    ]
