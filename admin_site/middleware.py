from social_django.middleware import SocialAuthExceptionMiddleware
from django.urls import reverse


class AuthExceptionMiddleware(SocialAuthExceptionMiddleware):
    def raise_exception(self, request, exception):
        return False

    def get_redirect_uri(self, request, exception):
        return reverse('wagtailadmin_login')
