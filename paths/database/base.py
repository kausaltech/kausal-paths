from django.db.backends.postgresql.base import (
    DatabaseWrapper as Psycopg2DatabaseWrapper
)
from django.db import close_old_connections, connection as db_connection
from psycopg2 import InterfaceError


class DatabaseWrapper(Psycopg2DatabaseWrapper):
    def create_cursor(self, name=None):
        try:
            return super().create_cursor(name=name)
        except InterfaceError:
            close_old_connections()
            db_connection.connect()
            return super().create_cursor(name=name)
