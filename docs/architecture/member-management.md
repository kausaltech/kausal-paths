# BISKO member management

`InstanceMemberAssignment` stores one municipal account's role and suspension
state. `InstanceMemberEvent` records role and lifecycle changes with an actor;
its retention date survives reactivation. The legacy instance role groups are
the active authorization projection. Existing group members are backfilled
into assignments by migration `nodes.0079`.

Municipal roles are `ADMIN`, `EDITOR`, `REVIEWER`, and `VIEWER`. An editor may
change data and finalise a balance but cannot use `instanceAdmin` member
mutations. `SUPER_ADMIN` remains an operator role and cannot be assigned by
municipal member management. The `addUserToInstance` and
`inviteUserToInstance` mutations accept a role and default to admin for old
clients. Invitations store that role for registration. `changeUserRole`,
`suspendUser`, and `reactivateUser` operate on one account. The old
`removeUserFromInstance` mutation now suspends rather than erasing its member
record. A suspended account loses its instance groups and selected instance;
the user and attributed data remain. The assignment records a retention date
five years after suspension.

The BISKO framework sets `max_user_accounts_per_instance` to five. Active
municipal accounts and pending invitations each consume one seat; suspended
members and operator accounts do not. The instance row is locked while changing
membership or reserving an invitation, so concurrent requests cannot claim the
same final seat. The instance GraphQL type exposes `memberSeatLimit` and
`memberSeatsInUse` to member administrators.

This field on `Framework` is the current per-instance seat policy. A separate
`License` model, with dates and state allocation, is still planned. The seat
limit does not represent a licence record. The test database covers the
membership lifecycle; usable Data Studio accounts are created through Kausal
OIDC. `configs/pruefstadt-bisko.yaml` enables user management for the review
fixture, and its existing database spec must be updated when that instance has
already been persisted.
