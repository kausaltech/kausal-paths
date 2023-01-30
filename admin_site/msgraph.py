import typing
import logging

import requests


if typing.TYPE_CHECKING:
    from users.models import User


logger = logging.getLogger(__name__)


def _get_token(user: 'User'):
    auth = user.social_auth.filter(provider='azure_ad').first()
    if not auth:
        backends = [x.provider for x in user.social_auth.all()]
        logger.error('User logged in with %s, not with Azure AD' % ', '.join(backends))
        return None

    return auth.extra_data['access_token']


def graph_get(resource: str, token: str):
    headers = dict(authorization='Bearer %s' % token)
    return requests.get('https://graph.microsoft.com/v1.0/%s' % resource, headers=headers, timeout=5)


def graph_get_json(resource: str, token: str):
    res = graph_get(resource, token)
    res.raise_for_status()
    return res.json()


def get_user_data(user: 'User', principal_name: str | None = None):
    token = _get_token(user)
    if not token:
        return
    if principal_name:
        resource = 'users/%s' % principal_name
    else:
        resource = 'me/'
    data = graph_get_json(resource, token)
    return data


def get_user_photo(user: 'User'):
    token = _get_token(user)
    if not token:
        return
    out = graph_get('me/photo/$value', token)
    if out.status_code == 404:
        return
    return out
