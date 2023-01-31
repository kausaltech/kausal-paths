import io
import logging

from sentry_sdk import capture_exception
from social_core.backends.oauth import OAuthAuth
from wagtail.users.models import UserProfile

from admin_site.msgraph import get_user_photo

from users.models import User
from users.base import uuid_to_username



logger = logging.getLogger('users.login')


def log_login_attempt(backend, details, *args, **kwargs):
    response = kwargs.get('response', {})
    request = kwargs['request']

    host = request.get_host()
    id_parts = ['backend=%s' % backend.name, 'host=%s' % host]
    email = response.get('email')
    if email:
        id_parts.append('email=%s' % email)
    tid = response.get('tid')
    if tid:
        id_parts.append('tid=%s' % tid)

    oid = response.get('oid')
    if oid:
        id_parts.append('oid=%s' % oid)
    else:
        sub = response.get('sub')
        if sub:
            id_parts.append('sub=%s' % sub)

    logger.info('Login attempt (%s)' % ', '.join(id_parts))
    if 'id_token' in response:
        logger.debug('ID token: %s' % response['id_token'])

    if isinstance(backend, OAuthAuth):
        try:
            backend.validate_state()
        except Exception as e:
            logger.warning('Login failed with invalid state: %s' % str(e))


def find_user_by_email(backend, details, user=None, social=None, *args, **kwargs):
    if user is not None:
        return

    details['email'] = details['email'].lower()
    try:
        user = User.objects.get(email=details['email'])
    except User.DoesNotExist:
        return

    return {
        'user': user,
        'is_new': False,
    }


def create_or_update_user(backend, details, user, *args, **kwargs):
    if user is None:
        uuid = details.get('uuid') or kwargs.get('uid')
        user = User(uuid=uuid)
        msg = 'Created new user'
    else:
        msg = 'Existing user found'
        uuid = user.uuid
    logger.info('%s (uuid=%s, email=%s)' % (msg, uuid, details.get('email')))

    changed = False
    for field in ('first_name', 'last_name', 'email'):
        old_val = getattr(user, field)
        new_val = details.get(field)
        if field in ('first_name', 'last_name'):
            if old_val is None:
                old_val = ''
            if new_val is None:
                new_val = ''

        if new_val != old_val:
            setattr(user, field, new_val)
            changed = True

    if user.has_usable_password():
        user.set_unusable_password()
        changed = True

    if changed:
        logger.info('User saved (uuid=%s, email=%s)' % (uuid, details.get('email')))
        user.save()

    return {
        'user': user,
    }


def update_avatar(backend, details, user, *args, **kwargs):
    if backend.name != 'azure_ad':
        return
    if user is None:
        return

    logger.info('Updating user photo (uuid=%s, email=%s)' % (user.uuid, details.get('email')))

    photo = None
    try:
        photo = get_user_photo(user)
    except Exception as e:
        logger.error('Failed to get user photo: %s' % str(e))
        capture_exception(e)

    if not photo:
        logger.info('No photo found (uuid=%s, email=%s)' % (user.uuid, details.get('email')))
        return

    # FIXME
    """
    person = user.get_corresponding_person()
    if person:
        try:
            person.set_avatar(photo.content)
        except Exception as e:
            logger.error('Failed to set avatar for person %s: %s' % (str(person), str(e)))
            capture_exception(e)

    profile = UserProfile.get_for_user(user)
    try:
        if not profile.avatar or profile.avatar.read() != photo.content:
            profile.avatar.save('avatar.jpg', io.BytesIO(photo.content))  # type: ignore
    except Exception as e:
        logger.error('Failed to set user profile photo: %s' % str(e))
        capture_exception(e)
    """


def get_username(details, backend, response, *args, **kwargs):
    """Sets the `username` argument.

    If the user exists already, use the existing username. Otherwise
    generate username from the `new_uuid` using the
    `helusers.utils.uuid_to_username` function.
    """

    user = details.get('user')
    if not user:
        user_uuid = kwargs.get('uid')
        if not user_uuid:
            return

        username = uuid_to_username(user_uuid)
    else:
        username = user.username

    return {
        'username': username
    }
