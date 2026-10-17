"""
Bring an instance's root page slugs back into line with its identifier.

    python manage.py repair_instance_page_slugs longmont longmont-2021          # plan only
    python manage.py repair_instance_page_slugs longmont longmont-2021 --apply
    python manage.py repair_instance_page_slugs --all --apply

Nothing is written without ``--apply``.

## Why this is needed at all

Page routing keys off the *identifier*, not off ``InstanceConfig.root_page``.
``PathsPage.resolve_url_path`` (``pages/page_interface.py``) strips
``^/<instance-id>(-[0-9]+)?/`` from the page's ``url_path``, and ``Query.resolve_page``
(``pages/schema.py``) then prepends the translated root page's ``url_path`` to the path it
is given. The two only agree while the root page's slug equals the identifier.

Nothing keeps them in step. The slug is set from the identifier once, when the pages are
created (``InstanceConfig._create_default_pages``), and never revisited -- so **renaming an
instance silently breaks every subpage of it**, and so does editing the root page's slug in
the Wagtail admin. The failure is a 404 on the children while the front page keeps working,
because the front page's path is empty and survives the mismatch: with identifier
``longmont`` and slug ``longmont-dev``, the strip does not match, ``resolve_url_path``
returns ``/longmont-dev/actions``, and ``resolve_page`` looks up
``/longmont-dev/longmont-dev/actions/``, which exists nowhere.

This has now happened twice -- ``augsburg-bisko`` in July 2026 via an admin slug edit, and
both Longmont instances in September 2026 via a rename -- which is why it is a command and
not a third one-off script.

## Translations are part of the fix

A translated root page carries a numeric suffix (``longmont-1``), which is what the
``(-[0-9]+)?`` in the regex is for, so the Spanish site breaks and heals by the same rule.
Translations are resolved by ``translation_key`` and locale
(``InstanceConfig.get_translated_root_page``), never by slug, so renaming them moves no
references. Every tree the instance serves is repaired the same way: slug, descendant
paths, and revisions.

## Two passes, because the slugs collide

Root pages are siblings under the Wagtail root and Wagtail requires a slug to be unique
among siblings, so a rename that swaps two identifiers cannot be applied in one pass: the
old instance still holds the slug the new one needs. Every page that has to move is first
parked on a temporary slug and then given its final one. That also makes the command
idempotent -- a page already correct is not touched, and re-running after a partial failure
converges.

``Page.save()`` cascades ``url_path`` to descendants when the slug changes, so the children
follow. Descendants whose ``url_path`` is *already* inconsistent with their parent -- which
Wagtail leaves behind when rows are orphaned -- are reported and rewritten too, since the
cascade only rewrites paths that share the old prefix. Those rewrites are worked out
*after* the renames, from live rows, because the renames move the very prefix they are
built from.

## Revisions have to move with the live rows

``Page.save()`` rewrites live rows only. A revision keeps the ``slug`` and ``url_path`` it
was saved with, and the Wagtail edit form populates from the latest revision -- so after a
rename, one *Save draft* in the admin writes the old slug straight back, and the instance
breaks again with nobody having touched a slug field.

On a swap it is worse than a revert. When Longmont was renamed by hand in September 2026,
``longmont-2021``'s Spanish root kept a revision saying ``longmont-1``, a slug that by then
belonged to ``longmont``'s Spanish root; publishing it would have put two live siblings on
one ``url_path``, where ``resolve_page`` returns whichever the queryset yields first. So
every revision of every page in the repaired trees is rewritten to the live values.

That rewrite is also why this command has something to do on an instance whose slugs are
already correct: a rename applied without carrying the revisions along leaves exactly that
state behind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from wagtail.models import Page

from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser


class SlugChange:
    """One root page that has to move, and where to."""

    def __init__(self, page: Page, desired: str, instance_identifier: str) -> None:
        self.page = page
        self.desired = desired
        self.instance_identifier = instance_identifier

    @property
    def needed(self) -> bool:
        return self.page.slug != self.desired


def _parent_prefix(page: Page) -> str:
    """``url_path`` of the page's parent, which every child path is built on top of."""
    parent = page.get_parent()
    return parent.url_path if parent is not None else '/'


def _own_url_path(page: Page) -> str:
    """Return the ``url_path`` this page should have for the slug it currently carries."""
    return f'{_parent_prefix(page)}{page.slug}/'


def _root_pages(ic: InstanceConfig) -> list[Page]:
    """
    Return the instance's root page and every translation of it.

    Resolved by ``translation_key``, the same way ``InstanceConfig.get_translated_root_page``
    finds them, so this covers every tree the instance actually serves -- not just the
    primary-language one that ``root_page`` points at.
    """
    root = ic.root_page
    if root is None:
        return []
    translations = Page.objects.filter(translation_key=root.translation_key).exclude(pk=root.pk).order_by('locale_id', 'pk')
    return [root, *translations]


def _desired_slugs(ic: InstanceConfig) -> list[SlugChange]:
    """
    Desired slug for the instance's root page and each of its translations.

    The primary-language page takes the identifier itself; translations take
    ``<identifier>-<n>``, numbered in a stable order so a re-run is a no-op. The number
    carries no meaning -- ``resolve_url_path`` accepts any digits -- it only has to be
    unique among the siblings.
    """
    pages = _root_pages(ic)
    if not pages:
        return []
    root, *translations = pages
    changes = [SlugChange(root, ic.identifier, ic.identifier)]
    for n, page in enumerate(translations, start=1):
        changes.append(SlugChange(page, f'{ic.identifier}-{n}', ic.identifier))
    return changes


def _expected_paths(root: Page, root_url_path: str) -> dict[int, str]:
    """
    ``url_path`` for the root and every descendant, given the root's final path.

    Walked top-down (``path`` order puts parents before children) so each level builds on
    the corrected level above it rather than on what is currently in the database. That is
    what lets the planning stage quote targets in post-rename terms.
    """
    expected = {root.pk: root_url_path}
    for page in root.get_descendants().order_by('path'):
        parent = page.get_parent()
        base = expected.get(parent.pk) if parent is not None else None
        if base is None:
            continue
        expected[page.pk] = f'{base}{page.slug}/'
    return expected


def _disconnected_pages(root: Page, root_url_path: str) -> list[tuple[Page, str]]:
    """
    Pages whose ``url_path`` does not continue their parent's, with the value to write.

    These are the rows the ``Page.save()`` cascade cannot repair: it rewrites only paths
    sharing the old prefix, so anything Wagtail left orphaned keeps its stale path straight
    through a rename.

    Detection compares against the parent's *current* path -- a page merely about to be
    moved by a rename is not disconnected, the cascade has that covered -- while the target
    comes from the post-rename layout, so what is reported is what will be written.
    """
    target = _expected_paths(root, root_url_path)
    out: list[tuple[Page, str]] = []
    for page in [root, *root.get_descendants().order_by('path')]:
        if page.url_path == _own_url_path(page):
            continue
        want = target.get(page.pk)
        if want is not None and want != page.url_path:
            out.append((page, want))
    return out


def _revision_reverts(content: dict[str, Any], want_slug: str, want_path: str | None) -> bool:
    """Whether publishing this revision would put back a slug or path the repair removes."""
    if 'slug' in content and content['slug'] != want_slug:
        return True
    return want_path is not None and 'url_path' in content and content['url_path'] != want_path


def _stale_revisions(root: Page, root_url_path: str, root_slug: str) -> list[tuple[Page, int]]:
    """
    Pages holding revisions that would undo the repair, with how many each has.

    Compared against the values the page will have *after* the repair, so this catches both
    a rename that has not happened yet and one that was applied without the revisions being
    carried along.
    """
    target = _expected_paths(root, root_url_path)
    out: list[tuple[Page, int]] = []
    for page in [root, *root.get_descendants().order_by('path')]:
        want_slug = root_slug if page.pk == root.pk else page.slug
        want_path = target.get(page.pk)
        n = sum(1 for rev in page.revisions.all() if _revision_reverts(rev.content, want_slug, want_path))
        if n:
            out.append((page, n))
    return out


def _force_subtree_paths(root_pk: int) -> list[tuple[int, str, str]]:
    """
    Rewrite ``url_path`` down a tree so every level continues its parent.

    Read from live rows and run after the slug passes, so it needs nothing carried over
    from planning -- which is what keeps it correct when a rename has just moved the prefix
    underneath the rows being repaired.

    ``queryset.update`` deliberately bypasses ``Page.save()``: the slug cascade has already
    run, and re-entering it here would re-derive the paths this is correcting.
    """
    root = Page.objects.get(pk=root_pk)
    fixed: list[tuple[int, str, str]] = []

    want_root = _own_url_path(root)
    if root.url_path != want_root:
        fixed.append((root.pk, root.url_path, want_root))
        Page.objects.filter(pk=root.pk).update(url_path=want_root)

    expected = {root.pk: want_root}
    for page in root.get_descendants().order_by('path'):
        parent = page.get_parent()
        base = expected.get(parent.pk) if parent is not None else None
        if base is None:
            continue
        want = f'{base}{page.slug}/'
        expected[page.pk] = want
        if page.url_path != want:
            fixed.append((page.pk, page.url_path, want))
            Page.objects.filter(pk=page.pk).update(url_path=want)
    return fixed


def _sync_revisions(page: Page) -> int:
    """Rewrite ``slug``/``url_path`` in every revision of the page to match the live row."""
    n = 0
    for revision in page.revisions.all():
        content = revision.content
        changed = False
        if 'slug' in content and content['slug'] != page.slug:
            content['slug'] = page.slug
            changed = True
        if 'url_path' in content and content['url_path'] != page.url_path:
            content['url_path'] = page.url_path
            changed = True
        if changed:
            revision.content = content
            revision.save(update_fields=['content'])
            n += 1
    return n


def _resolve_instances(identifiers: list[str], use_all: bool) -> list[InstanceConfig]:
    if use_all:
        if identifiers:
            raise CommandError('Give either --all or a list of instances, not both.')
        return list(InstanceConfig.objects.filter(is_active=True).order_by('identifier'))
    if not identifiers:
        raise CommandError('Name at least one instance, or pass --all.')
    instances: list[InstanceConfig] = []
    for identifier in identifiers:
        ic = InstanceConfig.objects.filter(identifier=identifier).first()
        if ic is None:
            raise CommandError(f'No instance {identifier!r}.')
        instances.append(ic)
    return instances


class Command(BaseCommand):
    help = 'Realign instance root page slugs with their identifiers, so subpages stop 404ing.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='*', metavar='INSTANCE_ID')
        parser.add_argument('--all', action='store_true', help='Check every active instance.')
        parser.add_argument('--apply', action='store_true', help='Write the changes (default: plan only).')

    def handle(self, *args: Any, **options: Any) -> None:
        instances = _resolve_instances(options['instances'], options['all'])
        changes, path_fixes, stale_revisions = self._collect(instances)

        if not changes and not path_fixes and not stale_revisions:
            self.stdout.write(self.style.SUCCESS('Nothing to do: every root page slug matches its identifier.'))
            return

        self._report(changes, path_fixes, stale_revisions)
        if not options['apply']:
            self.stdout.write(self.style.WARNING('\nPlan only. Re-run with --apply to write.'))
            return

        n_paths, n_revisions = self._apply(instances, changes)
        self.stdout.write(
            self.style.SUCCESS(f'\nApplied {len(changes)} slug changes, {n_paths} path fixes, {n_revisions} revision rewrites.')
        )

    def _collect(
        self, instances: list[InstanceConfig]
    ) -> tuple[list[SlugChange], list[tuple[Page, str]], list[tuple[Page, int]]]:
        changes: list[SlugChange] = []
        path_fixes: list[tuple[Page, str]] = []
        stale_revisions: list[tuple[Page, int]] = []
        for ic in instances:
            if ic.root_page is None:
                self.stdout.write(f'{ic.identifier}: no root page, skipped')
                continue
            for change in _desired_slugs(ic):
                if change.needed:
                    changes.append(change)
                # Every tree the instance serves, whether or not its own slug has to move:
                # a rename applied earlier without the revisions leaves work here too.
                root_url_path = f'{_parent_prefix(change.page)}{change.desired}/'
                path_fixes.extend(_disconnected_pages(change.page, root_url_path))
                stale_revisions.extend(_stale_revisions(change.page, root_url_path, change.desired))
        return changes, path_fixes, stale_revisions

    def _report(
        self,
        changes: list[SlugChange],
        path_fixes: list[tuple[Page, str]],
        stale_revisions: list[tuple[Page, int]],
    ) -> None:
        for change in changes:
            self.stdout.write(
                f'{change.instance_identifier}: page {change.page.pk} '
                f'{change.page.slug!r} -> {change.desired!r} (url_path {change.page.url_path!r})'
            )
        for page, expected in path_fixes:
            self.stdout.write(f'  page {page.pk} url_path {page.url_path!r} -> {expected!r} (was disconnected)')
        for page, count in stale_revisions:
            self.stdout.write(f'  page {page.pk} {count} revision(s) would put back the old slug/url_path')

    def _apply(self, instances: list[InstanceConfig], changes: list[SlugChange]) -> tuple[int, int]:
        n_paths = n_revisions = 0
        with transaction.atomic():
            # Park every mover on a slug nothing can collide with, so a swap between two
            # instances does not hit the sibling uniqueness constraint mid-way.
            for change in changes:
                change.page.slug = f'tmp-{change.page.pk}-{uuid4().hex[:8]}'
                change.page.save()
            for change in changes:
                change.page.slug = change.desired
                change.page.save()

            # Recomputed here rather than reused from the plan: the renames above have just
            # moved the prefix these paths are built from.
            for ic in instances:
                for root in _root_pages(ic):
                    n_paths += len(_force_subtree_paths(root.pk))
                    for page in Page.objects.get(pk=root.pk).get_descendants(inclusive=True):
                        n_revisions += _sync_revisions(page)
                ic.invalidate_cache()
        return n_paths, n_revisions
