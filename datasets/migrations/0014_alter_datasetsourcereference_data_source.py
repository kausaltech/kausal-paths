# Generated by Django 3.2.16 on 2023-03-03 08:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0014_datasource_instance'),
        ('datasets', '0013_make_cell_source_references_unique'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datasetsourcereference',
            name='data_source',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='references', to='nodes.datasource'),
        ),
    ]