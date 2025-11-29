from __future__ import annotations

from uuid import UUID


class UUIDSetTracker:
    """Base class for tracking UUID sets before and after operations."""

    uuids_before: set[UUID] | None
    uuids_after: set[UUID] | None

    def __init__(self, queryset):
        self.queryset = queryset
        self.uuids_before = None
        self.uuids_after = None

    def __enter__(self):
        self.uuids_before = set(self.queryset.values_list('uuid', flat=True))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uuids_after = set(self.queryset.values_list('uuid', flat=True))
        return False


class AssertIdenticalUUIDs(UUIDSetTracker):
    """Assert that the UUID set remains unchanged (for updates and failed operations)."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        assert self.uuids_after is not None
        assert self.uuids_before is not None
        assert self.uuids_after == self.uuids_before, (
            f'Expected identical UUID sets, but {len(self.uuids_after - self.uuids_before)} added '
            f'and {len(self.uuids_before - self.uuids_after)} removed'
        )
        return False


class AssertNewUUID(UUIDSetTracker):
    """Assert that exactly one new UUID was added and it matches the expected UUID."""

    def __init__(self, queryset, bulk: bool = False):
        super().__init__(queryset)
        self.bulk = bulk

    def assert_created(self, response_data, expected_num_created: int = 1):
        """Call this after exiting the context to verify the created UUIDs."""
        assert self.uuids_after is not None
        assert self.uuids_before is not None
        new_uuids = self.uuids_after - self.uuids_before
        assert len(new_uuids) == expected_num_created, \
            f'Expected exactly {expected_num_created} new UUIDs, but found {len(new_uuids)}'
        if self.bulk:
            created_uuids = {UUID(d['uuid']) for d in response_data}
        else:
            created_uuids = {UUID(response_data['uuid'])}
        assert new_uuids == created_uuids, \
            f'Expected new UUIDs to match response UUIDs {created_uuids}'


class AssertRemovedUUID(UUIDSetTracker):
    """Assert that exactly one UUID was removed and it matches the expected UUID."""

    def __init__(self, queryset, removed_uuid):
        super().__init__(queryset)
        self.removed_uuid = removed_uuid

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        assert self.uuids_after is not None
        assert self.uuids_before is not None
        assert self.uuids_after == self.uuids_before - {self.removed_uuid}, \
            f'Expected UUID {self.removed_uuid} to be removed, but UUID set changed incorrectly'
        return False
