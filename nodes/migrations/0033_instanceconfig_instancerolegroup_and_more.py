import django.db.models.deletion
import modelcluster.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0017_datasetsourcereference_uuid'),
        ('nodes', '0032_alter_instanceconfig_organization'),
        ('orgs', '0002_organizationmetadataadmin_and_more'),
        ('people', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='InstanceRoleGroup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150, verbose_name='name')),
                ('instance', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='role_groups', to='nodes.instanceconfig')),
            ],
            options={
                'verbose_name': 'Role group',
                'verbose_name_plural': 'Role groups',
            },
        ),
        migrations.CreateModel(
            name='InstanceRoleGroupDataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('can_view', models.BooleanField(default=False, verbose_name='can view')),
                ('can_edit', models.BooleanField(default=False, verbose_name='can edit')),
                ('can_delete', models.BooleanField(default=False, verbose_name='can delete')),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='instance_role_groups_edges', to='datasets.dataset', verbose_name='dataset')),
                ('group', modelcluster.fields.ParentalKey(on_delete=django.db.models.deletion.CASCADE, related_name='datasets_edges', to='nodes.instancerolegroup', verbose_name='group')),
            ],
            options={
                'verbose_name': 'Instance role group dataset',
                'verbose_name_plural': 'Instance role group datasets',
            },
        ),
        migrations.AddField(
            model_name='instancerolegroup',
            name='datasets',
            field=models.ManyToManyField(related_name='instance_role_groups', through='nodes.InstanceRoleGroupDataset', to='datasets.dataset'),
        ),
        migrations.CreateModel(
            name='InstanceRoleGroupPerson',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('group', modelcluster.fields.ParentalKey(on_delete=django.db.models.deletion.CASCADE, related_name='persons_edges', to='nodes.instancerolegroup', verbose_name='group')),
                ('person', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='instance_role_groups_edges', to='people.person', verbose_name='person')),
            ],
            options={
                'verbose_name': 'Instance role group person',
                'verbose_name_plural': 'Instance role group persons',
            },
        ),
        migrations.AddField(
            model_name='instancerolegroup',
            name='persons',
            field=models.ManyToManyField(related_name='instance_role_groups', through='nodes.InstanceRoleGroupPerson', to='people.person'),
        ),
        migrations.AddConstraint(
            model_name='instancerolegroupdataset',
            constraint=models.UniqueConstraint(fields=('group', 'dataset'), name='unique_dataset_per_instance_role_group'),
        ),
        migrations.AddConstraint(
            model_name='instancerolegroupperson',
            constraint=models.UniqueConstraint(fields=('group', 'person'), name='unique_person_per_instance_role_group'),
        ),
        migrations.AddConstraint(
            model_name='instancerolegroup',
            constraint=models.UniqueConstraint(fields=('instance', 'name'), name='unique_role_group_name_per_instance'),
        ),
    ]
