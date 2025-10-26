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

    def assert_created(self, response_data):
        """Call this after exiting the context to verify the created UUID."""
        assert self.uuids_after is not None
        assert self.uuids_before is not None
        new_uuids = self.uuids_after - self.uuids_before
        assert len(new_uuids) == 1, \
            f'Expected exactly 1 new UUID, but found {len(new_uuids)}'
        created_uuid = UUID(response_data['uuid'])
        assert new_uuids.pop() == created_uuid, \
            f'Expected new UUID to match response UUID {created_uuid}'


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
