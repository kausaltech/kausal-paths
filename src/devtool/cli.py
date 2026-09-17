"""
Command-line front end of the devtool.

    paths-devtool login                       # force a fresh browser sign-in
    paths-devtool logout                      # forget cached tokens
    paths-devtool token                       # print a fresh ID token, for curl
    paths-devtool whoami [--api-url URL]      # authenticated `me` query (the default command)
    paths-devtool instance list [--api-url URL]
    paths-devtool instance export INSTANCE [--api-url URL] [-o FILE]
    paths-devtool instance import FILE [--into ID] [--organization REF] [--name NAME] [--dry-run]

`instance import` writes to the local database (it boots Django); the other
commands talk to a backend over HTTP. ``python -m devtool`` is equivalent.
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2

from devtool.auth import DEFAULT_CALLBACK_PORT, DEFAULT_CLIENT_ID, DEFAULT_OIDC_ENDPOINT, SsoAuth, SsoConfig
from devtool.client import HttpTransport, PathsClient
from devtool.errors import DevtoolError, GraphQLError
from devtool.instances import default_export_path, describe_export, format_instance_table, save_export_document

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_API_URL = 'http://127.0.0.1:8000'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='paths-devtool',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--oidc-endpoint', default=DEFAULT_OIDC_ENDPOINT)
    parser.add_argument('--client-id', default=DEFAULT_CLIENT_ID)
    parser.add_argument('--port', type=int, default=DEFAULT_CALLBACK_PORT, help='Loopback callback port for login')
    parser.set_defaults(command='whoami', api_url=DEFAULT_API_URL)

    commands = parser.add_subparsers(dest='command')

    def with_api_url(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument('--api-url', default=DEFAULT_API_URL, help='Paths backend base URL (default: %(default)s)')
        return sub

    commands.add_parser('login', help='Force a fresh browser sign-in')
    commands.add_parser('logout', help='Forget cached tokens')
    commands.add_parser('token', help='Print a fresh ID token')
    with_api_url(commands.add_parser('whoami', help='Show who the backend thinks you are'))

    instance = commands.add_parser('instance', help='List, export and import instances')
    instance.set_defaults(subcommand='list')
    instance_commands = instance.add_subparsers(dest='subcommand')

    with_api_url(instance_commands.add_parser('list', help='List the instances you can edit'))

    export = with_api_url(instance_commands.add_parser('export', help='Download an InstanceExport document'))
    export.add_argument('instance', help='Instance identifier')
    export.add_argument('-o', '--output', type=Path, help='Output file (default: <identifier>.instance-export.json)')

    imp = instance_commands.add_parser('import', help='Load an InstanceExport file into the local database')
    imp.add_argument('file', type=Path)
    imp.add_argument('--into', help='Target instance identifier (default: the one in the document)')
    imp.add_argument('--organization', help='Organization UUID or name for a newly created instance')
    imp.add_argument('--name', help='Name for a newly created instance (default: from the document)')
    imp.add_argument('--dry-run', action='store_true', help='Run the import inside a transaction and roll it back')
    return parser


def whoami(client: PathsClient) -> int:
    try:
        me = client.me()
    except GraphQLError as e:
        print(f'The backend rejected the query: {e}')
        return 1
    if me is None:
        print('Not authenticated: the API answered, but `me` is null.')
        print('Likely causes:')
        print(' - the backend has no KAUSAL_SSO_DEVTOOL_CLIENT_ID configured')
        print(' - your Keycloak identity has no associated user (sign in to the admin UI once first)')
        return 1
    print(f'Authenticated as: {me["email"]}')
    return 0


def list_command(client: PathsClient) -> int:
    print(format_instance_table(client.list_instances()))
    return 0


def export_command(client: PathsClient, identifier: str, output: Path | None) -> int:
    document = client.export_instance(identifier)
    path = output if output is not None else default_export_path(document)
    save_export_document(document, path)
    print(describe_export(document))
    print(f'saved to:        {path}')
    return 0


def import_command(args: argparse.Namespace) -> int:
    from devtool.local_import import import_export_file

    summary = import_export_file(
        args.file,
        identifier=args.into,
        organization=args.organization,
        name=args.name,
        dry_run=args.dry_run,
    )
    print(summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'instance' and args.subcommand == 'import':
            return import_command(args)

        config = SsoConfig(oidc_endpoint=args.oidc_endpoint, client_id=args.client_id, callback_port=args.port)
        http = httpx2.Client(timeout=10)
        auth = SsoAuth(config, http=http)
        if args.command == 'logout':
            auth.logout()
            print('Cached tokens removed.')
            return 0
        if args.command == 'login':
            tokens = auth.login()
            print(f'Logged in as {tokens.subject_label} (sub={tokens.claims.get("sub")}).')
            return 0
        if args.command == 'token':
            print(auth.get_id_token())
            return 0

        client = PathsClient(HttpTransport(args.api_url, http=http), auth=auth)
        if args.command == 'instance':
            if args.subcommand == 'export':
                return export_command(client, args.instance, args.output)
            return list_command(client)
        return whoami(client)
    except DevtoolError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
