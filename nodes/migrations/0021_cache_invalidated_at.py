# Generated by Django 4.1.12 on 2023-10-18 10:59

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0020_add_extra_script_urls'),
    ]

    operations = [
        migrations.AddField(
            model_name='instanceconfig',
            name='cache_invalidated_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
