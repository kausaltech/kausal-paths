"""
Conversion between Wagtail ``url_path`` values and the paths the API speaks.

Wagtail stores an absolute path per page, rooted at the Wagtail root and built out of
slugs: ``/longmont/actions/``. The API speaks paths relative to the instance's root page:
``/actions``. Every page the API hands out goes through :func:`to_instance_path`, and every
path the API is given comes back through :func:`to_url_path`.

The two are inverses, and that round trip is the whole contract -- the front end takes a
page's ``urlPath`` and hands it straight back to ``page(path:)``.

They used to be implemented apart. The outbound side stripped
``^/<instance-id>(-[0-9]+)?/`` by regex while the inbound side prepended the root page's
``url_path``, so the two agreed only while the root page's slug happened to equal the
instance identifier -- something nothing in the codebase maintained. A rename or an admin
slug edit broke the assumption silently, and every subpage of the instance 404ed while the
front page kept working, because the front page's relative path is empty and survives the
mismatch either way. That cost two incidents (``augsburg-bisko`` in July 2026, both
Longmont instances in September 2026).

Both directions now take the prefix from the same place -- the root page's own
``url_path`` -- so a slug that disagrees with the identifier no longer separates them. Keep
it that way: if one of these functions changes, the other has to change with it, and
``test_url_paths.py`` round-trips them against slugs that do not match their identifier for
exactly that reason.
"""

from __future__ import annotations


def to_instance_path(url_path: str, root_url_path: str) -> str:
    """
    Convert a Wagtail ``url_path`` into the instance-relative path the API exposes.

    The instance's own root page becomes ``/``; anything below it keeps the remainder with
    no trailing slash. A page that does not sit under ``root_url_path`` is returned
    unchanged -- the API only ever exposes pages from the requested instance's tree, so
    that case means something upstream is already wrong and is not this function's to
    paper over.
    """
    prefix = root_url_path.rstrip('/')
    if prefix and url_path.startswith(prefix + '/'):
        url_path = url_path[len(prefix) :]
    if not url_path.startswith('/'):
        url_path = '/' + url_path
    if len(url_path) > 1:
        url_path = url_path.rstrip('/')
    return url_path


def to_url_path(instance_path: str, root_url_path: str) -> str:
    """
    Convert an instance-relative path into the Wagtail ``url_path`` to look up.

    Inverse of :func:`to_instance_path`. Wagtail's ``url_path`` always carries a trailing
    slash, so the result does too.
    """
    if not instance_path.startswith('/'):
        instance_path = '/' + instance_path
    if not instance_path.endswith('/'):
        instance_path += '/'
    return root_url_path.rstrip('/') + instance_path
