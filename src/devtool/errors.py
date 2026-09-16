from typing import Any


class DevtoolError(Exception):
    """Base class for errors the devtool reports to the user rather than as tracebacks."""


class AuthError(DevtoolError):
    """Signing in to Kausal SSO failed or produced no usable token."""


class TransportError(DevtoolError):
    """The backend did not answer with a well-formed GraphQL response."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = '') -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class GraphQLError(DevtoolError):
    """The backend answered, but the response carries ``errors``."""

    def __init__(self, errors: list[dict[str, Any]], data: dict[str, Any] | None = None) -> None:
        messages = '; '.join(str(e.get('message', e)) for e in errors) or 'unknown GraphQL error'
        super().__init__(messages)
        self.errors = errors
        self.data = data
