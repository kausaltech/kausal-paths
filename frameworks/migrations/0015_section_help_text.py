# Generated by Django 5.2 on 2025-06-30 10:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frameworks', '0014_measuretemplate_include_in_progress_tracker'),
    ]

    operations = [
        migrations.AddField(
            model_name='section',
            name='help_text',
            field=models.TextField(blank=True, default=''),
        ),
    ]
