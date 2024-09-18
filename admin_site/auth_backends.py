from __future__ import annotations

import uuid
from typing import Any

import sentry_sdk
from loguru import logger
from social_core.backends.azuread_tenant import AzureADTenantOAuth2
from social_core.backends.oauth import BaseOAuth2

from paths.const import FRAMEWORK_ADMIN_ROLE, INSTANCE_ADMIN_ROLE

from frameworks.roles import FrameworkRoleDef


class AzureADAuth(AzureADTenantOAuth2):
    name = 'azure_ad'
    TENANT_ID = 'organizations'
    # AUTHORIZATION_URL = 'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize'
    DEFAULT_SCOPE = ['openid', 'profile', 'email', 'User.Read']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.strategy.request is not None:
            #self.client = Client.objects.for_request(self.strategy.request).first()
            pass
        self.client = None

    @property
    def tenant_id(self):
        if self.client is None or not self.client.azure_ad_tenant_id:
            tenant_id = self.TENANT_ID
        else:
            tenant_id = self.client.azure_ad_tenant_id
        return tenant_id

    def jwks_url(self):
        return 'https://login.microsoftonline.com/common/discovery/keys'

    def auth_complete_params(self, state=None):
        ret = super().auth_complete_params(state)
        # Request access to the graph API
        ret['resource'] = 'https://graph.microsoft.com/'
        return ret

    def get_user_id(self, details, response):
        """Use oid claim as unique id."""
        oid = response['oid']
        # Replace the pairwise 'sub' field with the oid to better play along
        # with helusers.
        response['sub'] = oid
        return oid

    def get_user_details(self, response):
        details = super().get_user_details(response)
        # check `verified_primary_email` and enumerate through
        # `verified_secondary_email` to find possible matches
        # for `Person.email`
        if self.client and self.client.use_id_token_email_field:
            details['email'] = response.get('email') or details.get('email')
        details['uuid'] = response.get('oid')
        return details


NZC_PORTAL_NAMESPACE_UUID = uuid.UUID('71fa1a75-2694-41ff-85ee-ebfc22ff33b3')


class NZCPortalOAuth2(BaseOAuth2):
    name = 'nzcportal'
    AUTHORIZATION_URL = 'https://netzerocities.app/sso/authorize'
    ACCESS_TOKEN_URL = 'https://netzerocities.app/sso/token'  # noqa: S105
    ACCESS_TOKEN_METHOD = 'POST'  # noqa: S105
    STATE_PARAMETER = 'state'
    REDIRECT_STATE = False
    DEFAULT_SCOPE = ['basic']
    EXTRA_DATA = [
        ('access_token', 'access_token'),
        ('refresh_token', 'refresh_token'),
        ('expires_in', 'expires'),
    ]

    TYPE_TO_ROLE = {
        'cityAdmin': INSTANCE_ADMIN_ROLE,
        'consortiumUser': FRAMEWORK_ADMIN_ROLE,
        'cityUser': INSTANCE_ADMIN_ROLE,
    }

    def canonize_email(self, resp_email: str) -> str:
        return resp_email.strip().lower()

    def _get_user_details(self, response: dict[str, Any]) -> dict[str, Any]:
        # FIXME: Remove later
        logger.debug("User details: %s" % response)

        user_type = response.get('userType')
        role_id = self.TYPE_TO_ROLE.get(user_type or '')
        if role_id is None:
            sentry_sdk.capture_message('Unknown role: %s' % user_type)
        role = FrameworkRoleDef(
            framework_id='nzc',
            role_id=role_id,
            org_slug=response.get('userCity'),
            org_id=response.get('cityUID'),
        )
        logger.debug("Determined role: %s" % repr(role))
        return {
            'email': self.canonize_email(response['Mail']),
            'first_name': response.get('FirstName'),
            'last_name': response.get('LastName'),
            'framework_roles': [role],
        }

    def get_user_details(self, response: dict[str, Any]) -> dict[str, Any]:
        with sentry_sdk.new_scope():
            sentry_sdk.set_context('sso-response', response)
            return self._get_user_details(response)

    def user_data(self, access_token: str, *args, **kwargs):
        url = 'https://netzerocities.app/sso/user'
        headers = {'Authorization': f'Bearer {access_token}'}
        return self.get_json(url, headers=headers)

    def get_user_id(self, details, response: dict):
        email = self.canonize_email(response['Mail'])
        uid = uuid.uuid5(NZC_PORTAL_NAMESPACE_UUID, email)
        return uid

    def auth_params(self, state=None):
        params = super().auth_params(state)
        params['response_type'] = 'code'
        return params

    def auth_complete_params(self, state=None):
        params = super().auth_complete_params(state)
        params.update({
            'grant_type': 'authorization_code',
            'redirect_uri': self.get_redirect_uri(),
        })
        return params

    def get_key_and_secret(self):
        return self.setting('CLIENT_ID'), self.setting('CLIENT_SECRET')
