"""
Tests for the ``url_path`` conversions the page API round-trips through.

The point of these is the round trip, not either direction on its own. The front end takes
a page's ``urlPath`` and hands it straight back to ``page(path:)``, so the two functions
have to remain inverses -- and specifically have to remain inverses when the root page's
slug does *not* match the instance identifier, which is the case that broke twice before
the prefix was taken from one place.
"""

from __future__ import annotations

import pytest

from pages.url_paths import to_instance_path, to_url_path

# These are pure functions, but conftest's autouse `instance_config` fixture writes to
# the database, so the marker is required regardless.
pytestmark = pytest.mark.django_db

# Root page paths that all have to behave identically. Only the first matches the shape the
# old regex assumed; the rest are what a rename or an admin slug edit actually leaves.
ROOTS = [
    '/longmont/',
    '/longmont-dev/',
    '/longmont-1/',
    '/longmont-dev-1/',
    '/klimabilanz/',
    '/longmont-2021/',
]

RELATIVE_PATHS = ['/', '/actions', '/avoided', '/actions/deep', '/a-b_c']


@pytest.mark.parametrize('root_url_path', ROOTS)
@pytest.mark.parametrize('relative', RELATIVE_PATHS)
def test_round_trip_is_lossless_whatever_the_root_slug(root_url_path: str, relative: str) -> None:
    """What the API hands out has to be what the API can look back up."""
    url_path = to_url_path(relative, root_url_path)
    assert url_path.startswith(root_url_path)
    assert url_path.endswith('/')
    assert to_instance_path(url_path, root_url_path) == relative


@pytest.mark.parametrize('root_url_path', ROOTS)
def test_the_root_page_itself_is_the_bare_slash(root_url_path: str) -> None:
    """
    The front page is the case that hid both incidents.

    Its relative path is ``/`` regardless of the root slug, so it kept resolving while
    every child 404ed -- which is why the failure looked like a content problem rather than
    a routing one.
    """
    assert to_instance_path(root_url_path, root_url_path) == '/'
    assert to_url_path('/', root_url_path) == root_url_path


@pytest.mark.parametrize(
    ('url_path', 'root_url_path', 'expected'),
    [
        # A slug that disagrees with the identifier is no longer special.
        ('/longmont-dev/actions/', '/longmont-dev/', '/actions'),
        # The old regex stripped `<id>-<digits>` but not `<id>-dev`, so a translated root
        # whose slug had drifted was mangled twice over.
        ('/longmont-dev-1/avoided/', '/longmont-dev-1/', '/avoided'),
        # A root page whose slug happens to prefix another's must not over-strip.
        ('/longmont-2021/actions/', '/longmont-2021/', '/actions'),
    ],
)
def test_paths_the_old_regex_got_wrong(url_path: str, root_url_path: str, expected: str) -> None:
    assert to_instance_path(url_path, root_url_path) == expected


def test_a_sibling_root_is_not_treated_as_a_descendant() -> None:
    """
    ``/longmont-2021/`` is not under ``/longmont/`` and must not be read as if it were.

    A prefix test on the bare slug rather than on the full segment would strip ``/longmont``
    off the front of ``/longmont-2021/`` and invent a page.
    """
    assert to_instance_path('/longmont-2021/actions/', '/longmont/') == '/longmont-2021/actions'


def test_a_path_outside_the_instance_is_left_alone() -> None:
    assert to_instance_path('/elsewhere/actions/', '/longmont/') == '/elsewhere/actions'


@pytest.mark.parametrize('relative', ['/actions', 'actions', '/actions/'])
def test_inbound_paths_are_normalised(relative: str) -> None:
    """The front end is not required to be careful about leading or trailing slashes."""
    assert to_url_path(relative, '/longmont/') == '/longmont/actions/'
