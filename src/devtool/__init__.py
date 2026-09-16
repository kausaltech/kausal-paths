"""
Kausal Paths devtool: an authenticated GraphQL client for the Paths backend.

Phase A3 of ``docs/plans/devtool-auth-and-spec-tui.md``. Pure client code:
no Django, no server-side imports. ``devtool.auth`` performs the OIDC login
against Kausal SSO, ``devtool.client`` runs GraphQL operations against a
backend, and ``devtool.cli`` is the command-line front end
(``python -m devtool`` or the ``paths-devtool`` script).
"""

from devtool.auth import SsoAuth, SsoConfig, TokenCache
from devtool.client import HttpTransport, PathsClient
from devtool.errors import AuthError, DevtoolError, GraphQLError, TransportError

__all__ = [
    'AuthError',
    'DevtoolError',
    'GraphQLError',
    'HttpTransport',
    'PathsClient',
    'SsoAuth',
    'SsoConfig',
    'TokenCache',
    'TransportError',
]
