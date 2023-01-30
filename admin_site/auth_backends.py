from social_core.backends.azuread_tenant import AzureADTenantOAuth2

# from .models import Client


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
