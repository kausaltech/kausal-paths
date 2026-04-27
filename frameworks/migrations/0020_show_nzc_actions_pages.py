from django.db import migrations


def show_nzc_actions_pages(apps, schema_editor):
    FrameworkConfig = apps.get_model('frameworks', 'FrameworkConfig')
    ActionListPage = apps.get_model('pages', 'ActionListPage')
    Page = apps.get_model('wagtailcore', 'Page')

    page_ids: list[int] = []
    qs = FrameworkConfig.objects.filter(framework__identifier='nzc').select_related('instance_config__site__root_page')
    for fwc in qs:
        site = fwc.instance_config.site
        if site is None:
            continue
        root = site.root_page
        descendant_ids = ActionListPage.objects.filter(
            path__startswith=root.path,
            depth__gt=root.depth,
        ).values_list('pk', flat=True)
        page_ids.extend(descendant_ids)

    if not page_ids:
        return

    Page.objects.filter(pk__in=page_ids).update(show_in_menus=True)


class Migration(migrations.Migration):
    dependencies = [
        ('frameworks', '0019_framework_allow_instance_creation_and_more'),
        ('pages', '0017_alter_instancerootpage_body'),
        ('nodes', '0041_remove_nodeconfig_modified_at_and_more'),
    ]

    operations = [
        migrations.RunPython(show_nzc_actions_pages, migrations.RunPython.noop),
    ]
