# Generated by Django 3.2.16 on 2023-01-30 17:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0004_add_comment_review_metadata'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='dataset',
            options={'ordering': ('instance', 'name'), 'verbose_name': 'dataset', 'verbose_name_plural': 'datasets'},
        ),
    ]