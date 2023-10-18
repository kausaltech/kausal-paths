from django.db.models import QuerySet
from aplans.graphql_types import GQLInfo

def query(queryset: QuerySet, info: GQLInfo, **options): ...
