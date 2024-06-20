from uwsgidecorators import postfork # type: ignore


@postfork
def close_conns_post_fork():
    from django.db import connections
    for conn in connections.all():
        conn.close()
