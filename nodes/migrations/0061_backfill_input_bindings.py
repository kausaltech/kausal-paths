# Backfill the unified NodeInputPortBinding mirror from NodeEdge + DatasetPort.
#
# Binding UUIDs are the legacy row UUIDs, and positions reproduce the order
# the graph builder observes (edges in snapshot order, then dataset ports in
# their canonical sort). ORM access uses historical models; the ordering
# logic and row-to-snapshot mapping are imported from application code — they
# are pure functions over Pydantic values whose classes the migration state
# already references (see the SchemaField serialization in 0060), so they
# carry no schema drift of their own.

from django.db import migrations


def backfill_input_bindings(apps, schema_editor):
    from nodes.instance_serialization import DatasetPortSnapshot, EdgeSnapshot, ordered_binding_snapshots

    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    NodeEdge = apps.get_model('nodes', 'NodeEdge')
    DatasetPort = apps.get_model('nodes', 'DatasetPort')
    NodeInputPortBinding = apps.get_model('nodes', 'NodeInputPortBinding')

    for ic in InstanceConfig.objects.all():
        edge_rows = list(
            NodeEdge.objects
            .filter(instance=ic)
            .select_related('from_node', 'to_node')
            .order_by('from_node_id', 'to_node_id', 'to_port', 'pk')
        )
        port_rows = list(DatasetPort.objects.filter(instance=ic).select_related('node', 'dataset', 'metric'))
        if not edge_rows and not port_rows:
            continue

        edges_by_uuid = {e.uuid: e for e in edge_rows}
        ports_by_uuid = {p.uuid: p for p in port_rows}
        edge_snapshots = [EdgeSnapshot.from_model(e) for e in edge_rows]
        port_snapshots = [DatasetPortSnapshot.from_model(p) for p in port_rows]

        bindings = []
        for item, position in ordered_binding_snapshots(edge_snapshots, port_snapshots):
            if isinstance(item, DatasetPortSnapshot):
                port = ports_by_uuid[item.uuid]
                bindings.append(
                    NodeInputPortBinding(
                        uuid=port.uuid,
                        instance_id=ic.pk,
                        node_id=port.node_id,
                        port_id=port.port_id,
                        position=position,
                        dataset_id=port.dataset_id,
                        metric_id=port.metric_id,
                        transformations=list(port.spec.transformations),
                        tags=list(port.spec.tags),
                    )
                )
                continue
            edge = edges_by_uuid[item.uuid]
            bindings.append(
                NodeInputPortBinding(
                    uuid=edge.uuid,
                    instance_id=ic.pk,
                    node_id=edge.to_node_id,
                    port_id=edge.to_port,
                    position=position,
                    source_node_id=edge.from_node_id,
                    source_port_id=edge.from_port,
                    transformations=list(edge.transformations or []),
                    tags=list(edge.tags or []),
                )
            )
        NodeInputPortBinding.objects.bulk_create(bindings)


def remove_input_bindings(apps, schema_editor):
    NodeInputPortBinding = apps.get_model('nodes', 'NodeInputPortBinding')
    NodeInputPortBinding.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0060_node_input_port_binding'),
    ]

    operations = [
        migrations.RunPython(backfill_input_bindings, remove_input_bindings),
    ]
