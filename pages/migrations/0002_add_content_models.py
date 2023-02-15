# Generated by Django 3.1.8 on 2021-06-20 07:25

from django.db import migrations, models
import django.db.models.deletion
import wagtail.fields


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='InstanceContent',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('identifier', models.CharField(max_length=100, unique=True, verbose_name='Instance identifier')),
                ('modified_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Instance',
                'verbose_name_plural': 'Instances',
            },
        ),
        migrations.CreateModel(
            name='NodeContent',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('node_id', models.CharField(max_length=100, unique=True, verbose_name='Node identifier')),
                ('short_description', wagtail.fields.RichTextField(blank=True, null=True, verbose_name='Short description')),
                ('body', wagtail.fields.RichTextField(blank=True, null=True, verbose_name='Body')),
                ('instance', models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, to='pages.instancecontent')),
            ],
            options={
                'verbose_name': 'Node',
                'verbose_name_plural': 'Nodes',
            },
        ),
    ]
