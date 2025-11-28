from __future__ import annotations

import json

from django.http import HttpRequest, HttpResponse
from django.views import View
from wagtail.search.backends import get_search_backend

from dal.views import ViewMixin as DALViewMixin

from nodes.models import InstanceConfig, NodeConfig


def format_result(node_config) -> dict[str, str]:
    text = f'{node_config.name} ({node_config.identifier})'
    return {
        'id': node_config.identifier,
        'text': text,
        'selected_text': text,
    }


class NodeAutocompleteView(DALViewMixin, View):
    include_dimensional: bool

    def filter_result(self, node_config: NodeConfig) -> bool:
        if self.include_dimensional:
            return True
        node = node_config.get_node()
        if node is None:
            return False
        if node.output_dimensions:
            return False
        return True

    def get(self, request: HttpRequest) -> HttpResponse:
        query = request.GET.get('q')
        forwarded = self.forwarded
        instance = forwarded.get('instance')
        self.include_dimensional = bool(forwarded.get('include_dimensional'))
        if not instance:
            results = []
        else:
            queryset = NodeConfig.objects.filter(
                instance=InstanceConfig.objects.get(identifier=instance),
            )
            backend = get_search_backend()
            results = backend.autocomplete(query, queryset)
        json_result = json.dumps({
            'pagination': { 'more': False },
            'results': [
                format_result(node_config) for node_config in results[:20]
                if self.filter_result(node_config)
            ]
        }).encode('utf-8')
        return HttpResponse(json_result, content_type='application/json')
