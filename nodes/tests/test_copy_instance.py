"""
Tests for the node-reference remapping used by the ``copy_instance`` command.

The main long-term correctness risk when copying an instance's Wagtail pages
is that copied content keeps pointing at the *source* instance's NodeConfig
rows — not just on the live page rows but inside copied revisions (drafts,
preview, rollback). These tests assert remapping reaches all of them.
"""

import json
from io import StringIO
from typing import Any

from django.core.management import call_command
from django.core.management.base import CommandError
from wagtail.blocks import CharBlock, ListBlock, StreamBlock, StructBlock
from wagtail.models import Page

import pytest

from nodes.blocks import NodeChooserBlock
from nodes.management.commands.copy_instance import (
    _read_yaml_identity,
    _remap_raw,
    find_source_node_refs,
    remap_page_live,
    remap_page_revisions,
    rewrite_include_paths,
    rewrite_instance_yaml,
)
from nodes.models import InstanceConfig, InstanceHostname
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# _read_yaml_identity: name resolution for --use-existing-yaml defaults
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, text):
    p = tmp_path / 'inst.yaml'
    p.write_text(text)
    return p


def test_read_yaml_identity_prefers_top_level_name(tmp_path):
    p = _write_yaml(tmp_path, 'id: x\nsite_url: https://x/\nname: Top\nname_de: Deutsch\n')
    assert _read_yaml_identity(p) == 'Top'


def test_read_yaml_identity_uses_default_language_when_no_bare_name(tmp_path):
    # default_language is a regional code, but the name suffix is the short form.
    p = _write_yaml(tmp_path, 'id: x\ndefault_language: de-CH\nname_en: English\nname_de: Deutsch\n')
    assert _read_yaml_identity(p) == 'Deutsch'


def test_read_yaml_identity_falls_back_to_first_name_field(tmp_path):
    p = _write_yaml(tmp_path, 'id: x\nname_fi: Suomi\nname_en: English\n')
    assert _read_yaml_identity(p) == 'Suomi'


# ---------------------------------------------------------------------------
# rewrite_instance_yaml: only instance identity changes, data paths are kept
# ---------------------------------------------------------------------------


def test_rewrite_instance_yaml_changes_only_identity():
    data = {
        'id': 'zuerich',
        'site_url': 'https://zuerich.example/',
        'name_en': 'Net Zero Cockpit',
        'name_de': 'Netto-Null-Cockpit',
        'default_path': 'zuerich',
        'emission_dataset': 'zuerich/emissions',
        'nodes': [{'id': 'zuerich/population', 'type': 'ch.zuerich.Foo'}],
    }
    out = rewrite_instance_yaml(data, dest_id='zuerich-copy')

    assert out['id'] == 'zuerich-copy'
    assert 'site_url' not in out
    assert out['name_en'] == 'Net Zero Cockpit (copy)'
    assert out['name_de'] == 'Netto-Null-Cockpit (copy)'
    # Dataset paths and code references are left untouched.
    assert out['default_path'] == 'zuerich'
    assert out['emission_dataset'] == 'zuerich/emissions'
    assert out['nodes'][0] == {'id': 'zuerich/population', 'type': 'ch.zuerich.Foo'}


def test_rewrite_instance_yaml_drops_site_url_when_none():
    data = {'id': 'a', 'site_url': 'https://a.example/'}
    out = rewrite_instance_yaml(data, dest_id='b')
    assert out['id'] == 'b'
    assert 'site_url' not in out


def test_rewrite_instance_yaml_explicit_name_overwrites_all_name_fields():
    data = {'id': 'a', 'name_en': 'Foo', 'name_de': 'Föö'}
    out = rewrite_instance_yaml(data, dest_id='b', name='Sandbox')
    assert out['name_en'] == 'Sandbox'
    assert out['name_de'] == 'Sandbox'


def test_rewrite_include_paths_repoints_only_instance_fragments():
    data = {
        'include': [
            {'file': 'zuerich/buildings.yaml', 'node_group': 'Buildings'},
            {'file': 'zuerich/waste.yaml'},
            {'file': 'common/shared.yaml'},  # not under <src_id>/ → shared, untouched
            {'node_group': 'NoFileKey'},  # no 'file' → skipped
        ],
    }
    copies = rewrite_include_paths(data, src_id='zuerich', dest_id='zuerich-copy')

    inc = data['include']
    assert inc[0]['file'] == 'zuerich-copy/buildings.yaml'
    assert inc[0]['node_group'] == 'Buildings'  # sibling keys preserved
    assert inc[1]['file'] == 'zuerich-copy/waste.yaml'
    assert inc[2]['file'] == 'common/shared.yaml'  # shared library fragment left shared
    assert copies == [
        ('zuerich/buildings.yaml', 'zuerich-copy/buildings.yaml'),
        ('zuerich/waste.yaml', 'zuerich-copy/waste.yaml'),
    ]


def test_rewrite_include_paths_no_includes():
    assert rewrite_include_paths({'id': 'x'}, src_id='x', dest_id='y') == []


# ---------------------------------------------------------------------------
# _remap_raw: pure StreamField-walk over Struct / Stream / List nesting
# ---------------------------------------------------------------------------


def test_remap_raw_walks_nested_struct_stream_list():
    """
    NodeChooser PKs are remapped at every nesting depth; unknown PKs are kept.

    This is the depth that real blocks reach (e.g. DashboardCardBlock nests a
    NodeChooserBlock inside a StructBlock inside the cards StreamBlock).
    """
    block = StreamBlock([
        (
            'card',
            StructBlock([
                ('title', CharBlock()),
                ('node', NodeChooserBlock()),
                ('extra_nodes', ListBlock(NodeChooserBlock())),
            ]),
        ),
    ])
    raw: list[Any] = [
        {
            'type': 'card',
            'id': 'c1',
            'value': {
                'title': 'keep me',
                'node': 10,
                'extra_nodes': [
                    {'type': 'item', 'id': 'a', 'value': 20},
                    {'type': 'item', 'id': 'b', 'value': 99},  # unknown -> unchanged
                ],
            },
        },
    ]
    node_map = {10: 110, 20: 120}

    out = _remap_raw(block, raw, node_map)

    card = out[0]['value']
    assert card['title'] == 'keep me'
    assert card['node'] == 110
    assert [it['value'] for it in card['extra_nodes']] == [120, 99]
    # Input is not mutated in place.
    assert raw[0]['value']['node'] == 10


# ---------------------------------------------------------------------------
# Fixtures: a source instance + node and a destination with a matching node
# ---------------------------------------------------------------------------


@pytest.fixture
def remap_setup():
    src_ic = InstanceConfigFactory.create(name='copy-src', config_source='database')
    dst_ic = InstanceConfigFactory.create(name='copy-dst', config_source='database')
    src_node = NodeConfigFactory.create(instance=src_ic, identifier='net_emissions')
    dst_node = NodeConfigFactory.create(instance=dst_ic, identifier='net_emissions')
    node_map = {src_node.pk: dst_node.pk}
    return src_node, dst_node, node_map


# ---------------------------------------------------------------------------
# OutcomePage.outcome_node FK — live row and every revision
# ---------------------------------------------------------------------------


def test_remap_outcome_page_fk_in_row_and_revisions(remap_setup):
    from pages.models import OutcomePage

    src_node, dst_node, node_map = remap_setup
    root = Page.get_first_root_node()
    assert root is not None

    page = root.add_child(
        instance=OutcomePage(title='Cockpit', slug='cockpit', outcome_node=src_node),
    )
    # A published revision, then an unpublished draft revision.
    page.save_revision().publish()
    page.title = 'Cockpit (draft)'
    page.save_revision()  # unpublished

    assert page.revisions.count() >= 2

    changed = remap_page_live(page, node_map)
    n_revs = remap_page_revisions(page, node_map)

    assert changed is True
    assert n_revs == page.revisions.count()

    page.refresh_from_db()
    assert page.outcome_node_id == dst_node.pk
    for rev in page.revisions.all():
        assert rev.content['outcome_node'] == dst_node.pk
        assert rev.content['outcome_node'] != src_node.pk


# ---------------------------------------------------------------------------
# StreamField node refs — live row and every revision
# ---------------------------------------------------------------------------


def test_remap_streamfield_in_row_and_revisions(remap_setup):
    from pages.models import InstanceRootPage

    src_node, dst_node, node_map = remap_setup
    root = Page.get_first_root_node()
    assert root is not None

    body_block = InstanceRootPage._meta.get_field('body').stream_block
    raw = [{'type': 'outcome', 'id': 'o1', 'value': {'outcome_node': src_node.pk}}]

    page = root.add_child(
        instance=InstanceRootPage(
            title='Home',
            slug='copytest-home',
            body=body_block.to_python(raw),
        ),
    )
    page.save_revision().publish()
    page.save_revision()  # second, unpublished revision

    changed = remap_page_live(page, node_map)
    n_revs = remap_page_revisions(page, node_map)

    assert changed is True
    assert n_revs == page.revisions.count()

    page.refresh_from_db()
    body_pks = [b.value['outcome_node'].pk for b in page.body if b.block_type == 'outcome']
    assert body_pks == [dst_node.pk]

    for rev in page.revisions.all():
        body = json.loads(rev.content['body'])
        pks = [b['value']['outcome_node'] for b in body if b['type'] == 'outcome']
        assert pks == [dst_node.pk]
        assert src_node.pk not in pks


def test_find_source_node_refs_flags_unmapped_and_clears_after_remap(remap_setup):
    from pages.models import OutcomePage

    src_node, _dst_node, node_map = remap_setup
    root = Page.get_first_root_node()
    assert root is not None

    page = root.add_child(
        instance=OutcomePage(title='Cockpit', slug='cockpit-dangling', outcome_node=src_node),
    )
    page.save_revision().publish()

    source_pks = {src_node.pk}
    # Before remapping, the copied-style page still references the source node.
    assert find_source_node_refs(page, source_pks)

    # An empty node_map (e.g. the node was not materialised) leaves it dangling.
    remap_page_live(page, {})
    remap_page_revisions(page, {})
    assert find_source_node_refs(page, source_pks)

    # With a proper map the reference is cleared on row and revisions.
    remap_page_live(page, node_map)
    remap_page_revisions(page, node_map)
    page.refresh_from_db()
    assert find_source_node_refs(page, source_pks) == []


def test_remap_is_noop_when_no_node_refs(remap_setup):
    """A page with no node references is left untouched (no spurious saves)."""
    from pages.models import StaticPage

    _src_node, _dst_node, node_map = remap_setup
    root = Page.get_first_root_node()
    assert root is not None

    page = root.add_child(instance=StaticPage(title='Help', slug='help'))
    page.save_revision().publish()

    assert remap_page_live(page, node_map) is False
    assert remap_page_revisions(page, node_map) == 0


# ---------------------------------------------------------------------------
# Command-level integration (db mode): the full orchestration end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def db_source():
    """Build a database-backed source instance with a node, root page, and site content."""
    from pages.models import InstanceSiteContent, OutcomePage

    ic = InstanceConfigFactory.create(identifier='copytest-src', name='copytest-src', config_source='database')
    node = NodeConfigFactory.create(instance=ic, identifier='net_emissions')

    root = Page.get_first_root_node()
    assert root is not None
    home = root.add_child(instance=OutcomePage(title='Home', slug='copytest-src', outcome_node=node))
    home.save_revision().publish()
    home.title = 'Home (draft)'
    home.save_revision()  # an unpublished draft revision too

    ic.root_page = home
    ic.save(update_fields=['root_page'])

    # Give the source non-empty intro content so we can assert it is (not) copied.
    sc = InstanceSiteContent.objects.get(instance=ic)
    intro_block = InstanceSiteContent._meta.get_field('intro_content').stream_block
    sc.intro_content = intro_block.to_python([{'type': 'paragraph', 'id': 'p1', 'value': '<p>Intro</p>'}])
    sc.save()

    return ic, node


def _run(*args):
    call_command('copy_instance', *args, stdout=StringIO(), stderr=StringIO())


def test_copy_instance_db_mode_end_to_end(db_source):
    from pages.models import InstanceSiteContent, OutcomePage

    ic_src, _src_node = db_source
    _run('copytest-src', 'copytest-dst', '--site-url', 'https://copytest-dst.example/')

    cp = InstanceConfig.objects.get(identifier='copytest-dst')
    assert cp.config_source == 'database'
    assert {n.identifier for n in cp.nodes.all()} == {n.identifier for n in ic_src.nodes.all()}

    # copy_of points the copy (and its nodes) back at the source.
    assert cp.copy_of_id == ic_src.pk
    src_by_id = {n.identifier: n.pk for n in ic_src.nodes.all()}
    for n in cp.nodes.all():
        assert n.copy_of_id == src_by_id[n.identifier]

    # Root page + InstanceHostname routing both created for the copy.
    assert cp.root_page is not None
    assert InstanceHostname.objects.filter(instance=cp, hostname='copytest-dst.example').exists()

    # Page tree copied and the outcome_node FK repointed to the copy's node.
    src_pks = set(ic_src.nodes.values_list('pk', flat=True))
    cp_pks = set(cp.nodes.values_list('pk', flat=True))
    home = cp.root_page.specific
    assert isinstance(home, OutcomePage)
    assert home.outcome_node_id in cp_pks
    assert home.outcome_node_id not in src_pks
    for rev in home.revisions.all():
        assert rev.content['outcome_node'] in cp_pks
        assert rev.content['outcome_node'] not in src_pks

    # Instance site content copied.
    cp_sc = InstanceSiteContent.objects.get(instance=cp)
    assert len(cp_sc.intro_content.raw_data) == 1


def test_copy_instance_dry_run_leaves_nothing(db_source):
    _run('copytest-src', 'copytest-dst', '--site-url', 'https://copytest-dst.example/', '--dry-run')

    assert not InstanceConfig.objects.filter(identifier='copytest-dst').exists()
    assert not InstanceHostname.objects.filter(hostname='copytest-dst.example').exists()


def test_copy_instance_no_pages_skips_all_wagtail_content(db_source):
    from pages.models import InstanceSiteContent

    _run('copytest-src', 'copytest-dst', '--site-url', 'https://copytest-dst.example/', '--no-pages')

    cp = InstanceConfig.objects.get(identifier='copytest-dst')
    assert cp.nodes.exists()  # model still copied
    assert cp.root_page is None  # no pages
    assert not InstanceHostname.objects.filter(hostname='copytest-dst.example').exists()
    # The signal-created InstanceSiteContent stays blank — source intro NOT copied.
    cp_sc = InstanceSiteContent.objects.get(instance=cp)
    assert len(cp_sc.intro_content.raw_data) == 0


def test_yaml_mode_imports_editor_edges_from_snapshot(db_source):
    """yaml-mode recreates the editor graph (NodeEdge) from the snapshot, repointed to the copy's nodes."""
    import uuid as uuidlib

    from nodes.instance_serialization import (
        export_instance,
        import_instance_edges_and_ports,
        import_instance_nodes,
    )
    from nodes.models import NodeEdge

    ic_src, dst_node = db_source
    src_node = NodeConfigFactory.create(instance=ic_src, identifier='energy_use')
    NodeEdge.objects.create(
        instance=ic_src,
        from_node=src_node,
        to_node=dst_node,
        from_port=uuidlib.uuid4(),
        to_port=uuidlib.uuid4(),
    )

    export = export_instance(ic_src)
    ic_copy = InstanceConfigFactory.create(identifier='copytest-dst', name='dst', config_source='yaml')
    nodes_by_id = import_instance_nodes(ic_copy, export)
    import_instance_edges_and_ports(ic_copy, export, nodes_by_id, {})

    edge = NodeEdge.objects.get(instance=ic_copy)
    assert edge.from_node.identifier == 'energy_use'
    assert edge.to_node.identifier == 'net_emissions'
    copy_pks = set(ic_copy.nodes.values_list('pk', flat=True))
    assert {edge.from_node_id, edge.to_node_id} <= copy_pks
    # Edge must NOT reference the source's node rows, and config_source is untouched.
    assert edge.from_node_id not in set(ic_src.nodes.values_list('pk', flat=True))
    ic_copy.refresh_from_db()
    assert ic_copy.config_source == 'yaml'


def test_dataset_port_index_survives_export_import(db_source):
    """DatasetPort.dataset_index round-trips, preserving multi-input binding order."""
    import uuid as uuidlib
    from datetime import date
    from decimal import Decimal

    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.tests.factories import (
        DataPointFactory,
        DatasetFactory,
        DatasetMetricFactory,
        DatasetSchemaFactory,
    )

    from nodes.instance_serialization import (
        export_instance,
        import_instance_datasets,
        import_instance_edges_and_ports,
        import_instance_nodes,
    )
    from nodes.models import DatasetPort

    ic_src, node = db_source
    ct = ContentType.objects.get_for_model(ic_src)
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='value', label='Value')
    ds = DatasetFactory.create(schema=schema, scope_content_type=ct, scope_id=ic_src.pk, identifier='port/ds')
    DataPointFactory.create(dataset=ds, metric=metric, date=date(2023, 1, 1), value=Decimal(5))
    DatasetPort.objects.create(
        instance=ic_src,
        node=node,
        dataset=ds,
        metric=metric,
        port_id=uuidlib.uuid4(),
        dataset_index=3,
    )

    export = export_instance(ic_src)
    ic_copy = InstanceConfigFactory.create(identifier='copytest-dst', name='dst', config_source='yaml')
    nodes_by_id = import_instance_nodes(ic_copy, export)
    db_datasets = [d for d in export.datasets if not d.is_external_placeholder and d.data is not None]
    imported = import_instance_datasets(ic_copy, db_datasets, create_missing_dimensions=True)
    datasets_by_id = {d.identifier: d for d in imported if d.identifier is not None}
    import_instance_edges_and_ports(ic_copy, export, nodes_by_id, datasets_by_id)

    port = DatasetPort.objects.get(instance=ic_copy)
    assert port.dataset_index == 3
    assert port.node_id in set(ic_copy.nodes.values_list('pk', flat=True))


# ---------------------------------------------------------------------------
# yaml-stage flags (--write-config-only / --use-existing-yaml): validation
# ---------------------------------------------------------------------------


def test_yaml_stage_flags_require_yaml_mode(db_source):
    with pytest.raises(CommandError, match='only valid in yaml mode'):
        _run('copytest-src', 'copytest-dst', '--mode', 'db', '--use-existing-yaml')


def test_write_config_only_rejects_dry_run(db_source):
    with pytest.raises(CommandError, match='cannot be combined with --dry-run'):
        _run('copytest-src', 'copytest-dst', '--mode', 'yaml', '--write-config-only', '--dry-run')


def test_use_existing_yaml_requires_the_file(db_source):
    # No configs/copytest-dst.yaml exists, so applying from it must fail loudly.
    with pytest.raises(CommandError, match='was not found'):
        _run('copytest-src', 'copytest-dst', '--mode', 'yaml', '--use-existing-yaml')


def test_yaml_mode_rejects_framework_source_in_both_stages(db_source, monkeypatch):
    # The framework guard must fire on the --use-existing-yaml path too, not only
    # when writing the config.
    monkeypatch.setattr(InstanceConfig, 'has_framework_config', lambda _self: True)
    with pytest.raises(CommandError, match='framework-backed'):
        _run('copytest-src', 'copytest-dst', '--mode', 'yaml', '--use-existing-yaml')


# ---------------------------------------------------------------------------
# Wagtail pages in the snapshot (export-only, for verification)
# ---------------------------------------------------------------------------


def test_streamfield_to_identifiers_maps_node_pks():
    from nodes.page_snapshot import _streamfield_to_identifiers

    block = StreamBlock([('outcome', StructBlock([('outcome_node', NodeChooserBlock())]))])
    raw = [{'type': 'outcome', 'id': 'o1', 'value': {'outcome_node': 42}}]
    out = _streamfield_to_identifiers(block, raw, {42: 'net_emissions'})
    assert out[0]['value']['outcome_node'] == 'net_emissions'
    assert 'id' not in out[0]  # random block ids dropped for comparability


def test_export_includes_page_snapshot_with_node_identifiers(db_source):
    from nodes.instance_serialization import export_instance

    ic_src, src_node = db_source
    export = export_instance(ic_src)

    def walk(snaps):
        for s in snaps:
            yield s
            yield from walk(s.children)

    outcome_pages = [s for s in walk(export.pages) if s.type == 'OutcomePage']
    assert outcome_pages, 'page snapshot should include the OutcomePage'
    # The node reference is expressed by identifier (so source/copy compare equal).
    assert outcome_pages[0].outcome_node == src_node.identifier


# ---------------------------------------------------------------------------
# Dataset round-trip: metric-column resolution (datapoints must survive import)
# ---------------------------------------------------------------------------


def _make_snapshot(metrics, fields):
    from nodes.instance_serialization import DatasetSnapshot

    return DatasetSnapshot(
        identifier='ds',
        dimensions=['building_use'],
        metrics=metrics,
        data={'schema': {'fields': fields}, 'data': []},
    )


def test_resolve_metric_data_columns_falls_back_to_label():
    """
    A metric with no ``name`` (identifier == uuid) is keyed to its label column.

    Regression: ``DBDataset.deserialize_df`` names the value column
    ``Coalesce(name, label, uuid)``, so a nameless-but-labelled metric's data
    lives under the label, not the uuid — previously every datapoint was
    dropped on import.
    """
    from kausal_common.i18n.pydantic import TranslatedString

    from nodes.instance_serialization import DatasetMetricSnapshot, _resolve_metric_data_columns

    label = TranslatedString('Floor Area', default_language='en')
    metrics = [DatasetMetricSnapshot(identifier='461f-uuid', label=label, unit='m**2')]
    fields = [{'name': 'Year'}, {'name': 'building_use'}, {'name': 'Floor Area', 'unit': 'm**2'}]
    cols = _resolve_metric_data_columns(_make_snapshot(metrics, fields), ['461f-uuid'], {'building_use': 'building_use'})
    assert cols == {'461f-uuid': 'Floor Area'}


def test_resolve_metric_data_columns_prefers_identifier_when_present():
    from nodes.instance_serialization import DatasetMetricSnapshot, _resolve_metric_data_columns

    metrics = [DatasetMetricSnapshot(identifier='floor_area', label=None, unit='m**2')]
    fields = [{'name': 'Year'}, {'name': 'building_use'}, {'name': 'floor_area', 'unit': 'm**2'}]
    cols = _resolve_metric_data_columns(_make_snapshot(metrics, fields), ['floor_area'], {'building_use': 'building_use'})
    assert cols == {'floor_area': 'floor_area'}
