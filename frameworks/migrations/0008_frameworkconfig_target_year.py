# Generated by Django 5.1.4 on 2024-12-12 11:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frameworks', '0007_framework_viewer_group'),
    ]

    operations = [
        migrations.AddField(
            model_name='frameworkconfig',
            name='target_year',
            field=models.IntegerField(null=True),
        ),
    ]