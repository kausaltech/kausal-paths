from __future__ import annotations

from django.urls import reverse

import sentry_sdk
from social_django.middleware import SocialAuthExceptionMiddleware


class AuthExceptionMiddleware(SocialAuthExceptionMiddleware):
    def raise_exception(self, request, exception):
        sentry_sdk.capture_exception(exception)
        return False

    def get_redirect_uri(self, request, exception):
        return reverse('wagtailadmin_login')
