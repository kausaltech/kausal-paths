# Generated by Django 3.2.16 on 2023-03-03 07:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0012_add_dataset_source_reference'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='datasetcomment',
            options={'ordering': ('dataset', 'cell_path', '-created_at'), 'verbose_name': 'comment', 'verbose_name_plural': 'comments'},
        ),
        migrations.AlterModelOptions(
            name='datasetsourcereference',
            options={'ordering': ('dataset', 'cell_path'), 'verbose_name': 'data source reference', 'verbose_name_plural': 'data source references'},
        ),
        migrations.AlterUniqueTogether(
            name='datasetsourcereference',
            unique_together={('dataset', 'cell_path')},
        ),
    ]