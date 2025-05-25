from __future__ import annotations

from kausal_common.logging.request_log.models import BaseLoggedRequest


class LoggedRequest(BaseLoggedRequest):
    class Meta(BaseLoggedRequest.Meta):
        abstract = False
        app_label = 'request_log'
    pass
