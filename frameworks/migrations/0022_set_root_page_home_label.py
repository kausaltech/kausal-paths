from django.db import migrations


def set_root_page_home_label(apps, schema_editor):
    FrameworkConfig = apps.get_model('frameworks', 'FrameworkConfig')
    InstanceRootPage = apps.get_model('pages', 'InstanceRootPage')
    Page = apps.get_model('wagtailcore', 'Page')

    root_page_ids: list[int] = []
    qs = FrameworkConfig.objects.select_related('instance_config__site__root_page')
    for fwc in qs:
        ic = fwc.instance_config
        if ic is None or ic.site is None:
            continue
        root_page_ids.append(ic.site.root_page_id)

    if not root_page_ids:
        return

    Page.objects.filter(pk__in=root_page_ids).update(show_in_menus=True)
    InstanceRootPage.objects.filter(pk__in=root_page_ids, menu_label='').update(menu_label='Home')


class Migration(migrations.Migration):
    dependencies = [
        ('frameworks', '0021_show_nzc_actions_pages'),
        ('pages', '0018_add_menu_label'),
    ]

    operations = [
        migrations.RunPython(set_root_page_home_label, migrations.RunPython.noop),
    ]
