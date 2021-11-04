# Generated by Django 3.2.4 on 2021-10-25 04:17

import django
from django.db import migrations, models
import wagtail.core.fields


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='nodeconfig',
            name='body',
            field=wagtail.core.fields.RichTextField(blank=True, null=True, verbose_name='Body'),
        ),
        migrations.AddField(
            model_name='nodeconfig',
            name='short_description',
            field=wagtail.core.fields.RichTextField(blank=True, null=True, verbose_name='Short description'),
        ),
        migrations.AddField(
            model_name='instanceconfig',
            name='lead_paragraph',
            field=wagtail.core.fields.RichTextField(blank=True, null=True, verbose_name='Lead paragraph'),
        ),
        migrations.AddField(
            model_name='instanceconfig',
            name='lead_title',
            field=models.CharField(blank=True, max_length=100, verbose_name='Lead title'),
        ),
        migrations.AlterField(
            model_name='nodeconfig',
            name='forecast_values',
            field=models.JSONField(editable=False, null=True),
        ),
        migrations.AlterField(
            model_name='nodeconfig',
            name='historical_values',
            field=models.JSONField(editable=False, null=True),
        ),
        migrations.AlterField(
            model_name='nodeconfig',
            name='instance',
            field=models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, related_name='nodes', to='nodes.instanceconfig'),
        ),
        migrations.AlterField(
            model_name='nodeconfig',
            name='params',
            field=models.JSONField(editable=False, null=True),
        ),
    ]
