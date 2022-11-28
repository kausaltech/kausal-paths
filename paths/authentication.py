from django.http import HttpRequest
from rest_framework.authentication import TokenAuthentication
from rest_framework import exceptions
from django.utils.translation import gettext_lazy as _
from oauth2_provider.views.mixins import OAuthLibMixin
from oauth2_provider.oauth2_validators import OAuth2Validator

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


class IDTokenAuthentication(TokenAuthentication):
    keyword = 'Bearer'

    def authenticate_credentials(self, key: str):
        validator_class = OAuthLibMixin.get_validator_class()
        validator: OAuth2Validator = validator_class()  # type: ignore
        token = validator._load_id_token(key)
        if not token:
            return None
        return token.user, token
