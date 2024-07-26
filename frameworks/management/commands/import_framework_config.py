from typing import Any
from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from rich import print


class Command(BaseCommand):
    help = 'Import framework configurations from a source API to a target API'

    def add_arguments(self, parser):
        parser.add_argument('framework_id', type=str, help="Identifier of the framework")
        parser.add_argument('source_api_url', type=str, help='Source API URL')
        parser.add_argument('--target_api_url', type=str, default='http://127.0.0.1:8000', help='Target API URL')

    def handle(self, *args, **options):
        source_api_url = options['source_api_url']
        target_api_url = options['target_api_url']

        source_api_url = f"{source_api_url}/v1/graphql/"
        target_api_url = f"{target_api_url}/v1/graphql/"
        fw_id = options['framework_id']

        self.stdout.write(f"Source API URL: {source_api_url}")
        self.stdout.write(f"Target API URL: {target_api_url}")
        self.stdout.write(f"Framework identifier: {fw_id}")

        src_client = self.get_gql_client(source_api_url)
        target_client = self.get_gql_client(target_api_url)

        self.stdout.write("Getting framework config IDs...")
        data = self.query_fwc_ids(src_client, fw_id)
        fw = data['framework']
        self.stdout.write("Framework name: %s" % fw['name'])
        self.stdout.write("Querying for for existing framework configs...")
        resp = self.query_fwc_ids(target_client, fw_id=fw_id)
        for fwc in fw['configs']:
            self.import_framework_config(src_client, target_client, fw_id, resp, fwc)

    def get_gql_client(self, url: str):
        transport = AIOHTTPTransport(url=url, timeout=40)
        client = Client(transport=transport, fetch_schema_from_transport=False)
        return client

    def load_gql_query(self, query_name: str):
        p = Path(settings.BASE_DIR) / 'gql_client' / 'queries' / f"{query_name}.gql"
        data = p.read_text()
        q = gql(data)
        return q

    def execute_operation(self, client: Client, query_name: str, vals: dict[str, Any]):
        query = self.load_gql_query(query_name)
        result = client.execute(query, variable_values=vals)
        return result

    def query_fwc_ids(self, client: Client, fw_id: str):
        return self.execute_operation(
            client, 'GetFrameworkConfigIDs', vals=dict(frameworkId=fw_id)
        )

    def query_framework_config(self, client: Client, fw_id: str, fwc_id: str):
        return self.execute_operation(
            client, 'GetFrameworkConfig', vals=dict(
                frameworkId=fw_id,
                fwcId=fwc_id,
            )
        )

    def remove_framework_config(self, client: Client, id_or_uuid: str):
        res = self.execute_operation(client, 'DeleteFrameworkConfig', vals=dict(
            id=id_or_uuid,
        ))
        if not res['deleteFrameworkConfig']['ok']:
            print(res)
            raise Exception("deletion failed")

    def create_framework_config(self, client: Client, fw_id: str, data: dict):
        create_config_mutation = self.load_gql_query('CreateFrameworkConfig')
        variables = {
            "frameworkId": fw_id,
            "instanceIdentifier": data['instance']['id'],
            "name": data['organizationName'],
            "baselineYear": data['baselineYear'],
            "uuid": data['uuid'],
        }

        result = client.execute(create_config_mutation, variable_values=variables)
        resp = result['createFrameworkConfig']
        if not resp['ok']:
            raise Exception("Failed to create FrameworkConfig")
        fwc = resp['frameworkConfig']
        self.stdout.write(
            self.style.SUCCESS("Created new FrameworkConfig with ID %s, UUID %s" % (fwc['id'], fwc['uuid']))
        )
        framework_config_id = fwc['id']
        assert resp['frameworkConfig']['uuid'] == data['uuid']

        update_measures_mutation = self.load_gql_query('UpdateMeasures')
        measures = []
        for measure in data['measures']:
            measure_input = {
                "measureTemplateId": measure['measureTemplate']['uuid'],
                "dataPoints": [
                    {
                        "year": dp['year'],
                        "value": dp['value']
                    } for dp in measure['dataPoints']
                ]
            }
            measures.append(measure_input)

        variables = {
            "frameworkConfigId": framework_config_id,
            "measures": measures
        }

        result = client.execute(update_measures_mutation, variable_values=variables)
        resp = result['updateMeasureDataPoints']
        if not resp['ok']:
            raise Exception("Failed to update measure data points")
        self.stdout.write(self.style.SUCCESS(
            f"Created {len(result['updateMeasureDataPoints']['createdDataPoints'])} data points."
        ))

    def import_framework_config(self, src_client: Client, target_client: Client, fw_id: str, existing_data: dict, fwc: dict):
        self.stdout.write(self.style.MIGRATE_HEADING(
            "Importing framework config for %s (UUID %s)" % (fwc['organizationName'], fwc['uuid'])
        ))

        data = self.query_framework_config(src_client, fw_id, fwc['id'])

        old_fwc = existing_data['framework']['configs']
        existing_uuids = [d['uuid'] for d in old_fwc]
        fwc_uuid = fwc['uuid']
        if fwc_uuid in existing_uuids:
            self.stdout.write(
                self.style.WARNING('Config already exists locally, removing')
            )
            self.remove_framework_config(target_client, fwc_uuid)
        self.stdout.write("Creating framework config locally")
        fwc_conf = data['framework']['config']
        self.create_framework_config(target_client, fw_id, fwc_conf)
