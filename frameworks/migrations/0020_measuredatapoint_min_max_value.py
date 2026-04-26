from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frameworks', '0019_framework_allow_instance_creation_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='measuredatapoint',
            name='min_value',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='measuredatapoint',
            name='max_value',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
