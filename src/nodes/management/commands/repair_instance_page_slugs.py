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
references.

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
cascade only rewrites paths that share the old prefix.
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


def _desired_slugs(ic: InstanceConfig) -> list[SlugChange]:
    """
    Desired slug for the instance's root page and each of its translations.

    The primary-language page takes the identifier itself; translations take
    ``<identifier>-<n>``, numbered in a stable order so a re-run is a no-op. The number
    carries no meaning -- ``resolve_url_path`` accepts any digits -- it only has to be
    unique among the siblings.
    """
    root = ic.root_page
    if root is None:
        return []
    changes = [SlugChange(root, ic.identifier, ic.identifier)]
    translations = Page.objects.filter(translation_key=root.translation_key).exclude(pk=root.pk).order_by('locale_id', 'pk')
    for n, page in enumerate(translations, start=1):
        changes.append(SlugChange(page, f'{ic.identifier}-{n}', ic.identifier))
    return changes


def _inconsistent_descendants(root: Page) -> list[tuple[Page, str]]:
    """Descendants whose ``url_path`` does not continue their parent's, with the corrected value."""
    out: list[tuple[Page, str]] = []
    by_pk = {root.pk: root.url_path}
    for page in root.get_descendants().order_by('path'):
        parent = page.get_parent()
        parent_path = by_pk.get(parent.pk) if parent is not None else None
        if parent_path is None:
            continue
        expected = f'{parent_path}{page.slug}/'
        by_pk[page.pk] = expected
        if page.url_path != expected:
            out.append((page, expected))
    return out


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
        changes, path_fixes = self._collect(instances)

        if not changes and not path_fixes:
            self.stdout.write(self.style.SUCCESS('Nothing to do: every root page slug matches its identifier.'))
            return

        self._report(changes, path_fixes)
        if not options['apply']:
            self.stdout.write(self.style.WARNING('\nPlan only. Re-run with --apply to write.'))
            return

        self._apply(instances, changes, path_fixes)
        self.stdout.write(self.style.SUCCESS(f'\nApplied {len(changes)} slug changes, {len(path_fixes)} path fixes.'))

    def _collect(self, instances: list[InstanceConfig]) -> tuple[list[SlugChange], list[tuple[Page, str]]]:
        changes: list[SlugChange] = []
        path_fixes: list[tuple[Page, str]] = []
        for ic in instances:
            if ic.root_page is None:
                self.stdout.write(f'{ic.identifier}: no root page, skipped')
                continue
            changes.extend(change for change in _desired_slugs(ic) if change.needed)
            path_fixes.extend(_inconsistent_descendants(ic.root_page))
        return changes, path_fixes

    def _report(self, changes: list[SlugChange], path_fixes: list[tuple[Page, str]]) -> None:
        for change in changes:
            self.stdout.write(
                f'{change.instance_identifier}: page {change.page.pk} '
                f'{change.page.slug!r} -> {change.desired!r} (url_path {change.page.url_path!r})'
            )
        for page, expected in path_fixes:
            self.stdout.write(f'  page {page.pk} url_path {page.url_path!r} -> {expected!r} (was inconsistent)')

    def _apply(
        self,
        instances: list[InstanceConfig],
        changes: list[SlugChange],
        path_fixes: list[tuple[Page, str]],
    ) -> None:
        with transaction.atomic():
            # Park every mover on a slug nothing can collide with, so a swap between two
            # instances does not hit the sibling uniqueness constraint mid-way.
            for change in changes:
                change.page.slug = f'tmp-{change.page.pk}-{uuid4().hex[:8]}'
                change.page.save()
            for change in changes:
                change.page.slug = change.desired
                change.page.save()
            for page, expected in path_fixes:
                fresh = Page.objects.get(pk=page.pk)
                if fresh.url_path != expected:
                    Page.objects.filter(pk=page.pk).update(url_path=expected)

            for ic in instances:
                ic.invalidate_cache()
