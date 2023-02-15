# Generated by Django 3.2.8 on 2021-10-29 12:31

from django.db import migrations, models
import django.db.models.deletion
import modelcluster.fields
import wagtail.fields


class Migration(migrations.Migration):

    dependencies = [
        ('wagtailcore', '0062_comment_models_and_pagesubscription'),
        ('nodes', '0004_add_fields'),
        ('pages', '0005_move_node_and_instance_content_to_nodes_app'),
    ]

    operations = [
        migrations.CreateModel(
            name='ActionListPage',
            fields=[
                ('page_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='wagtailcore.page')),
                ('i18n', models.JSONField(blank=True, null=True)),
                ('show_in_footer', models.BooleanField(default=False, help_text='Should the page be shown in the footer?', verbose_name='show in footer')),
                ('lead_title', models.CharField(blank=True, max_length=100, verbose_name='Lead title')),
                ('lead_paragraph', wagtail.fields.RichTextField(blank=True, null=True, verbose_name='Lead paragraph')),
            ],
            options={
                'verbose_name': 'Action list page',
                'verbose_name_plural': 'Action list pages',
            },
            bases=('wagtailcore.page',),
        ),
        migrations.CreateModel(
            name='OutcomePage',
            fields=[
                ('page_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='wagtailcore.page')),
                ('i18n', models.JSONField(blank=True, null=True)),
                ('show_in_footer', models.BooleanField(default=False, help_text='Should the page be shown in the footer?', verbose_name='show in footer')),
                ('lead_title', models.CharField(blank=True, max_length=100, verbose_name='Lead title')),
                ('lead_paragraph', wagtail.fields.RichTextField(blank=True, null=True, verbose_name='Lead paragraph')),
                ('outcome_node', modelcluster.fields.ParentalKey(on_delete=django.db.models.deletion.PROTECT, related_name='pages', to='nodes.nodeconfig')),
            ],
            options={
                'verbose_name': 'Outcome page',
                'verbose_name_plural': 'Outcome pages',
            },
            bases=('wagtailcore.page',),
        ),
        migrations.RemoveField(
            model_name='nodecontent',
            name='instance',
        ),
        migrations.DeleteModel(
            name='InstanceContent',
        ),
        migrations.DeleteModel(
            name='NodeContent',
        ),
    ]
