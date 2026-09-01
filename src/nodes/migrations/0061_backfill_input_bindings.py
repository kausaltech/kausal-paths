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

    # Row-to-snapshot mapping inlined from the (since removed) legacy
    # ``from_model`` classmethods; the historical rows carry the same
    # attributes. Only the fields the ordering logic reads matter here.
    def edge_snapshot(obj):
        return EdgeSnapshot(
            uuid=obj.uuid,
            from_node=obj.from_node.uuid,
            to_node=obj.to_node.uuid,
            from_port=obj.from_port,
            to_port=obj.to_port,
            transformations=obj.transformations or [],
            tags=obj.tags or [],
        )

    def port_snapshot(obj):
        return DatasetPortSnapshot(
            uuid=obj.uuid,
            node=obj.node.uuid,
            dataset=obj.dataset.identifier or str(obj.dataset.uuid),
            dataset_uuid=obj.dataset.uuid,
            port_id=obj.port_id,
            metric=obj.metric.name or str(obj.metric.uuid),
            metric_uuid=obj.metric.uuid,
            dataset_index=obj.dataset_index,
            spec=obj.spec,
        )

    for ic in InstanceConfig.objects.all():
        edge_rows = list(
            NodeEdge.objects
            .filter(instance=ic)
            .select_related('from_node', 'to_node')
            # Creation order = authored order; must match edge_qs_for().
            .order_by('pk')
        )
        port_rows = list(DatasetPort.objects.filter(instance=ic).select_related('node', 'dataset', 'metric'))
        if not edge_rows and not port_rows:
            continue

        edges_by_uuid = {e.uuid: e for e in edge_rows}
        ports_by_uuid = {p.uuid: p for p in port_rows}
        edge_snapshots = [edge_snapshot(e) for e in edge_rows]
        port_snapshots = [port_snapshot(p) for p in port_rows]

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
