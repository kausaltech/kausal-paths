import importlib.util

from django.http import HttpRequest
from rest_framework.authentication import TokenAuthentication
from rest_framework import exceptions
from django.utils.translation import gettext_lazy as _


from nodes.models import InstanceConfig


class InstanceTokenAuthentication(TokenAuthentication):
    def authenticate(self, request: HttpRequest) -> str | None:
        return super().authenticate(request)  # type: ignore

    def authenticate_credentials(self, key):
        return key

        try:
            instance = InstanceConfig.objects.get(tokens__token=key)
        except InstanceConfig.DoesNotExist:
            raise exceptions.AuthenticationFailed(_('Invalid token.'))
        return instance


if importlib.util.find_spec('kausal_paths_extensions') is not None:
    from kausal_paths_extensions.auth.authentication import IDTokenAuthentication
else:
    IDTokenAuthentication = None  # type: ignore
