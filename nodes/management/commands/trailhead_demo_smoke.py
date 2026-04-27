"""
End-to-end smoke test for the Trailhead change-tracking chain.

Exercises the CADS self-service mutations and the Tuesday-demo "copy an
action" flow against a live database:

    1. register_user (framework `cads`)
    2. create_instance (aarhus-c4c template)
    3. create_dimension_categories (new action category in the `action` dim)
    4. create_data_point (x N) on `aarhus/energy_actions` for the new action
    5. create_node (new Action node — copy of carbon_capture_and_storage)
    6. create_edge (x 3) wiring the new action to chp_emissions /
                           electricity_demand / energy_costs

After each write, the script dumps the resulting InstanceChangeOperation
rows and their InstanceModelLogEntry children so regressions show up
plainly.

Usage::

    python manage.py trailhead_demo_smoke
    python manage.py trailhead_demo_smoke --identifier demo-foo --keep
"""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, date, datetime
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.test.client import Client

from paths.tests.graphql import PathsTestClient

# ---------------------------------------------------------------------------
# GraphQL operations
# ---------------------------------------------------------------------------

REGISTER_USER = """
mutation RegisterUser($input: RegisterUserInput!) {
    registerUser(input: $input) {
        __typename
        ... on RegisterUserResult { userId email }
        ... on OperationInfo { messages { kind message field } }
    }
}
"""

CREATE_INSTANCE = """
mutation CreateInstance($input: CreateInstanceInput!) {
    createInstance(input: $input) {
        __typename
        ... on CreateInstanceResult { instanceId instanceName }
        ... on OperationInfo { messages { kind message field } }
    }
}
"""

LIST_DIMENSIONS = """
query ListDimensions($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            dimensions {
                id
                identifier
                name
                categories { id identifier label }
            }
        }
    }
}
"""

LIST_DATASETS = """
query ListDatasets($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            datasets {
                id
                identifier
                metrics { id name label unit }
            }
        }
    }
}
"""

CREATE_DIMENSION_CATEGORIES = """
mutation CreateCats($instanceId: ID!, $input: [CreateDimensionCategoryInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        createDimensionCategories(input: $input) {
            ... on InstanceDimension { id categories { id identifier label } }
        }
    }
}
"""

CREATE_DATA_POINT = """
mutation CreateDataPoint($instanceId: ID!, $datasetId: ID!, $input: CreateDataPointInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            createDataPoint(input: $input) {
                __typename
                ... on DataPoint { id date value }
                ... on OperationInfo { messages { kind message field code } }
            }
        }
    }
}
"""

CREATE_NODE = """
mutation CreateNode($instanceId: ID!, $input: CreateNodeInput!) {
    instanceEditor(instanceId: $instanceId) {
        createNode(input: $input) {
            ... on NodeInterface { identifier name kind }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

CREATE_EDGE = """
mutation CreateEdge($instanceId: ID!, $input: CreateEdgeInput!) {
    instanceEditor(instanceId: $instanceId) {
        createEdge(input: $input) {
            __typename
            ... on NodeEdgeType { fromRef { nodeId } toRef { nodeId } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

ADD_NODE_INPUT_PORT = """
mutation AddNodeInputPort($instanceId: ID!, $nodeId: ID!, $input: InputPortInput!) {
    instanceEditor(instanceId: $instanceId) {
        addNodeInputPort(nodeId: $nodeId, input: $input) {
            ... on InputPortType { id quantity unit { standard } }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_action_id(ts: str) -> str:
    return f'demo_copied_ccs_{ts}'


def _unique_suffix() -> str:
    return datetime.now(UTC).strftime('%Y%m%d%H%M%S') + '_' + secrets.token_hex(2)


def _find_dim(dimensions: list[dict[str, Any]], identifier: str) -> dict[str, Any] | None:
    return next((d for d in dimensions if d.get('identifier') == identifier), None)


def _find_dataset(datasets: list[dict[str, Any]], identifier: str) -> dict[str, Any] | None:
    return next((d for d in datasets if d.get('identifier') == identifier), None)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _dump_ops(stdout: Any, instance_pk: int, *, since: datetime | None = None) -> None:
    import json

    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    qs = InstanceChangeOperation.objects.filter(instance_config_id=instance_pk).order_by('created_at')
    if since is not None:
        qs = qs.filter(created_at__gt=since)

    stdout.write('')
    stdout.write('=' * 72)
    stdout.write('InstanceChangeOperation rows for instance pk=%d' % instance_pk)
    stdout.write('=' * 72)
    for op in qs:
        entry_count = InstanceModelLogEntry.objects.filter(operation=op).count()
        user = op.user.email if op.user else '(none)'
        stdout.write(f'  {op.created_at:%H:%M:%S}  {op.action:40s}  source={op.source:<8}  user={user}  entries={entry_count}')

    # Spot-check: show one full entry payload per distinct action.
    seen: set[str] = set()
    sample = []
    for e in InstanceModelLogEntry.objects.filter(operation__in=qs).order_by('id'):
        if e.action in seen:
            continue
        seen.add(e.action)
        sample.append(e)
    if sample:
        stdout.write('')
        stdout.write('Sample entry payload per action (first seen):')
        for e in sample:
            stdout.write(f'  - action={e.action}')
            stdout.write('    data=' + json.dumps(e.data, indent=2, default=str).replace('\n', '\n    '))


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class Command(BaseCommand):
    help = 'Smoke-test the Trailhead change-tracking chain against a live DB.'

    def add_arguments(self, parser):
        parser.add_argument('--framework', default='cads', help='Framework identifier (default: cads)')
        parser.add_argument('--identifier', default=None, help='Instance identifier (default: demo-smoke-<ts>)')
        parser.add_argument('--password', default='DemoSmoke!234', help='Password for the registered user')
        parser.add_argument('--keep', action='store_true', help='Do not delete the instance/user at the end')

    def handle(self, *args, **options):  # noqa: PLR0915  — linear smoke-test narrative
        framework_id = options['framework']
        suffix = options['identifier'] or f'demo-smoke-{_unique_suffix()}'
        password = options['password']
        keep = options['keep']

        email = f'{suffix}@example.com'
        action_id = _new_action_id(suffix.replace('-', '_'))

        self.stdout.write(self.style.NOTICE(f'Framework: {framework_id}'))
        self.stdout.write(self.style.NOTICE(f'Instance identifier: {suffix}'))
        self.stdout.write(self.style.NOTICE(f'User email: {email}'))
        self.stdout.write(self.style.NOTICE(f'New action identifier: {action_id}'))

        # 1. Register user (anonymous)
        anon_client = Client()
        anon_client.raise_request_exception = False
        anon = PathsTestClient(anon_client)
        anon.query_data(
            REGISTER_USER,
            variables={
                'input': {
                    'frameworkId': framework_id,
                    'email': email,
                    'password': password,
                    'firstName': 'Demo',
                    'lastName': 'Smoke',
                }
            },
        )
        self.stdout.write(self.style.SUCCESS(f'✓ Registered {email}'))

        # 2. Force-login the new user
        User = get_user_model()
        user = User.objects.get(email=email)

        auth_client = Client()
        auth_client.raise_request_exception = False
        auth_client.force_login(user)
        client = PathsTestClient(auth_client)

        # 3. Create instance (clones from framework template)
        started_at = datetime.now(UTC)
        resp = client.query_data(
            CREATE_INSTANCE,
            variables={
                'input': {
                    'frameworkId': framework_id,
                    'name': f'Demo Smoke {suffix}',
                    'identifier': suffix,
                    'organizationName': f'Demo Smoke Org {suffix}',
                }
            },
        )
        new_instance_id = resp['createInstance']['instanceId']
        self.stdout.write(self.style.SUCCESS(f'✓ Created instance {new_instance_id}'))

        from nodes.models import InstanceConfig

        ic = InstanceConfig.objects.get(identifier=new_instance_id)
        client.set_instance(ic)

        # 4. Add a new category to the `action` dimension
        dimensions_data = client.query_data(
            LIST_DIMENSIONS,
            variables={'instanceId': str(ic.pk)},
        )['modelInstance']['editor']['dimensions']

        action_dim = _find_dim(dimensions_data, 'action')
        if action_dim is None:
            self._cleanup(ic, user, keep)
            raise CommandError("'action' dimension not found in new instance")

        cat_resp = client.query_data(
            CREATE_DIMENSION_CATEGORIES,
            variables={
                'instanceId': str(ic.pk),
                'input': [
                    {
                        'dimensionId': action_dim['id'],
                        'identifier': action_id,
                        'label': f'Demo CCS copy {suffix}',
                    }
                ],
            },
        )
        new_cat = next(
            c for c in cat_resp['instanceEditor']['createDimensionCategories']['categories'] if c['identifier'] == action_id
        )
        new_cat_uuid = new_cat['id']
        self.stdout.write(self.style.SUCCESS(f'✓ Added category {action_id} (uuid={new_cat_uuid}) to `action`'))

        # 5. Add datapoints on aarhus/energy_actions for the new action
        datasets_data = client.query_data(
            LIST_DATASETS,
            variables={'instanceId': str(ic.pk)},
        )['modelInstance']['editor']['datasets']
        ea_dataset = _find_dataset(datasets_data, 'aarhus/energy_actions')
        if ea_dataset is None:
            self._cleanup(ic, user, keep)
            raise CommandError("'aarhus/energy_actions' dataset not found in new instance")

        ea_id = ea_dataset['id']
        metrics = ea_dataset['metrics']
        # Add one datapoint per metric, for 2024 and 2030, combined with the new action category.
        impact_years = [2024, 2030]
        # The aarhus/energy_actions schema has many dimensions; the simplest
        # payload that validates is providing the new `action` category + one
        # cell per metric per year. If additional required dimensions exist,
        # this will surface as a validation error which is exactly what we
        # want the smoke test to catch.
        dp_count = 0
        for metric in metrics:
            if metric['name'] not in {'emissions', 'energy', 'currency'}:
                continue
            for year in impact_years:
                client.query_data(
                    CREATE_DATA_POINT,
                    variables={
                        'instanceId': str(ic.pk),
                        'datasetId': ea_id,
                        'input': {
                            'date': date(year, 1, 1).isoformat(),
                            'value': 1.0,
                            'metricId': metric['id'],
                            'dimensionCategoryIds': [new_cat_uuid],
                        },
                    },
                )
                dp_count += 1
        self.stdout.write(self.style.SUCCESS(f'✓ Created {dp_count} datapoints on aarhus/energy_actions'))

        # 6. Create the new Action node (copy of CCS's shape). Pre-allocate
        #    output-port UUIDs so we can wire edges against them below.
        port_uuids = {
            'emissions': str(uuid.uuid4()),
            'energy': str(uuid.uuid4()),
            'currency': str(uuid.uuid4()),
        }
        node_input = {
            'identifier': action_id,
            'name': f'Demo CCS copy {suffix}',
            'kind': 'ACTION',
            'config': {'action': {'nodeClass': 'nodes.actions.simple.AdditiveAction'}},
            'color': '#336699',
            # ``columnId`` is required for multi-port nodes — it becomes
            # the NodeMetric.id at runtime, which the hydrate path accesses
            # by attribute.
            'outputPorts': [
                {'id': port_uuids['emissions'], 'unit': 't/a', 'quantity': 'emissions', 'columnId': 'emissions'},
                {'id': port_uuids['energy'], 'unit': 'TJ/a', 'quantity': 'energy', 'columnId': 'energy'},
                {'id': port_uuids['currency'], 'unit': 'DKK/a', 'quantity': 'currency', 'columnId': 'currency'},
            ],
        }
        client.query_data(CREATE_NODE, variables={'instanceId': str(ic.pk), 'input': node_input})
        self.stdout.write(self.style.SUCCESS(f'✓ Created node {action_id}'))

        # 7. Wire edges. For each metric: add a fresh matching input port
        # on the target, then connect the edge with the same dimension
        # transformations CCS uses (flatten unused dimensions). This is
        # what produces a computable model rather than just tracked
        # structure.
        edge_plan = [
            # (metric, target_node, output_unit, output_quantity,
            #  flatten_dimensions_before_target)
            ('emissions', 'chp_emissions', 't/a', 'emissions', ['energy_usage', 'cost_type']),
            ('energy', 'electricity_demand', 'TJ/a', 'energy', ['cost_type', 'sector', 'ghg']),
            ('currency', 'energy_costs', 'DKK/a', 'currency', ['energy_carrier', 'energy_usage', 'ghg']),
        ]
        for metric_key, target, unit, quantity, flatten_dims in edge_plan:
            # 7a. Add a new input port on the target that accepts this quantity.
            port_resp = client.query_data(
                ADD_NODE_INPUT_PORT,
                variables={
                    'instanceId': str(ic.pk),
                    'nodeId': target,
                    'input': {
                        'quantity': quantity,
                        'unit': unit,
                        'multi': False,
                    },
                },
            )
            to_port_id = port_resp['instanceEditor']['addNodeInputPort']['id']

            # 7b. Connect the edge with the matching flatten transformations.
            transformations = [{'flatten': {'dimension': dim}} for dim in flatten_dims]
            resp = client.query(
                CREATE_EDGE,
                variables={
                    'instanceId': str(ic.pk),
                    'input': {
                        'instanceId': str(ic.pk),
                        'fromNodeId': action_id,
                        'fromPort': port_uuids[metric_key],
                        'toNodeId': target,
                        'toPort': to_port_id,
                        'transformations': transformations,
                    },
                },
                assert_no_errors=False,
            )
            if resp.errors:
                msg = resp.errors[0].get('message', str(resp.errors[0]))
                self.stdout.write(self.style.WARNING(f'  edge {metric_key} → {target} failed: {msg}'))
            else:
                self.stdout.write(self.style.SUCCESS(f'✓ Created edge {metric_key} → {target}'))

        # 8. Report the operations emitted during the flow
        _dump_ops(self.stdout, ic.pk, since=started_at)

        # 9. Cleanup (unless --keep)
        self._cleanup(ic, user, keep)

    def _cleanup(self, ic: Any, user: Any, keep: bool) -> None:
        if keep:
            self.stdout.write(
                self.style.NOTICE(
                    f'--keep set; leaving instance {ic.identifier} (pk={ic.pk}) and user {user.email} in place.',
                )
            )
            return
        with transaction.atomic():
            ic.delete()
            user.delete()
        self.stdout.write(self.style.NOTICE('✓ Cleaned up instance + user'))
