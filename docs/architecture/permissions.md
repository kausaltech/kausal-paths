# Permission Policy System

## Overview

Kausal Paths uses a layered, policy-based permission system built on top of
Wagtail's `ModelPermissionPolicy`. Every model that requires access control
inherits from `PermissionedModel` and implements a `permission_policy()`
classmethod that returns a policy object encapsulating all permission logic for
that model.

The design separates _what_ the permission logic is (the policy class) from
_where_ it's used (views, GraphQL resolvers, DRF viewsets). Consumer code calls
policy methods; it does not contain permission logic itself.

## Core Classes

All classes live in `kausal_common/models/`:

```
WagtailModelPermissionPolicy       (wagtail.permission_policies.base)
└── ModelPermissionPolicy          (permission_policy.py) — abstract, generic
    ├── ModelReadOnlyPolicy        (permission_policy.py) — everyone can view, only superusers write
    └── ParentInheritedPolicy      (permission_policy.py) — delegates to a parent model's policy
        └── (concrete child policies in paths/dataset_permission_policy.py)

PermissionedModel                  (permissions.py) — abstract base for all permissioned models
PermissionedQuerySet               (permissions.py) — adds .viewable_by(), .modifiable_by(), etc.
PermissionedManager                (permissions.py) — returns PermissionedQuerySet by default
```

### `ModelPermissionPolicy[M, CreateContext, QS]`

The abstract base. Generic over:
- `M` — the model class
- `CreateContext` — type of the context object passed when checking creation
  permission (often a parent model instance)
- `QS` — the queryset class (defaults to `PermissionedQuerySet[M]`)

**Abstract methods that every concrete policy must implement:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `construct_perm_q` | `(user, action) -> Q \| None` | ORM Q object for authenticated users. `None` means no rows allowed. |
| `construct_perm_q_anon` | `(action) -> Q \| None` | ORM Q object for anonymous users. |
| `user_has_perm` | `(user, action, obj) -> bool` | Instance-level check for authenticated users. |
| `anon_has_perm` | `(action, obj) -> bool` | Instance-level check for anonymous users. |
| `user_can_create` | `(user, context) -> bool` | Whether user may create a new object given a context. |

**Concrete methods provided:**

| Method | Purpose |
|--------|---------|
| `filter_by_perm(qs, user, action)` | Applies `construct_perm_q*` to a queryset. Returns `qs.none()` if Q is None. |
| `instances_user_has_permission_for(user, action)` | Returns a filtered queryset from the model's default manager. |
| `instances_user_has_any_permission_for(user, actions)` | Same, but ORs together multiple actions. |
| `user_has_permission_for_instance(user, action, obj)` | Routes to `user_has_perm` / `anon_has_perm` / superuser shortcut. |
| `gql_action_allowed(info, action, obj, context)` | GraphQL entry point — extracts user from `info.context`, dispatches to appropriate check. |
| `is_field_visible(instance, field_name, user)` | True if field is in `public_fields` or user has write access. |
| `creatable_child_models(user, obj)` | Child models the user can create under this object (used in GraphQL). |
| `_construct_q(user, action)` | Internal: routes to `construct_perm_q_anon` / `Q()` for superuser / `construct_perm_q`. |

**Superuser shortcut:** `_construct_q` returns `Q()` (match all rows) for
superusers, so policies only need to handle regular users.

### `ModelReadOnlyPolicy`

Everyone can `view`. Only superusers can `add`, `change`, or `delete`.
Use for reference / catalogue data that model builders manage but coordinators
only read: `Framework`, `Section`, `MeasureTemplate`.

### `ParentInheritedPolicy[M, ParentM, QS]`

Delegates all permission checks to the parent model's policy. Constructed with:
```python
ParentInheritedPolicy(model, parent_model, parent_field, disallowed_actions=())
```

- `parent_field` — name of the FK field on `M` pointing to `ParentM`
- `disallowed_actions` — actions to unconditionally deny (e.g. `('add', 'delete')`)

`construct_perm_q` translates to `parent_field__in=<parent policy queryset>`,
so database filtering stays in a single query. `user_can_create` defaults to
requiring `change` permission on the parent.

Also registers the child in `parent_model.child_models`, which feeds
`creatable_child_models`.

### `PermissionedModel`

Abstract Django model base. Requires:
```python
@classmethod
@abstractmethod
def permission_policy(cls) -> ModelPermissionPolicy[Self, Any, Any]: ...
```

Adds two convenience methods for GraphQL:
- `gql_action_allowed(info, action)` — delegates to the policy
- `ensure_gql_action_allowed(info, action)` — raises `GraphQLError` if denied

Django system-check `kausal_common.P001` / `P002` will warn at startup if any
concrete `PermissionedModel` subclass lacks a working `permission_policy()`.

### `PermissionedQuerySet`

Adds shorthand methods that call the model's policy:

```python
qs.viewable_by(user)      # filter_by_perm(user, 'view')
qs.modifiable_by(user)    # filter_by_perm(user, 'change')
qs.deletable_by(user)     # filter_by_perm(user, 'delete')
qs.filter_by_perm(user, action)
```

The `_pp` property instantiates the policy on demand via `self.model.permission_policy()`.

## Defining a New Policy

A minimal role-based policy looks like this:

```python
class MyModelPermissionPolicy(ModelPermissionPolicy['MyModel', 'ParentModel', 'MyModelQuerySet']):
    def __init__(self):
        from .models import MyModel
        from nodes.roles import instance_admin_role, instance_viewer_role
        self.admin_role = instance_admin_role
        self.viewer_role = instance_viewer_role
        super().__init__(MyModel)

    def construct_perm_q(self, user: User, action: ObjectSpecificAction) -> Q | None:
        admin_q = self.admin_role.role_q(user, prefix='instance_config')
        if action == 'view':
            viewer_q = self.viewer_role.role_q(user, prefix='instance_config')
            return admin_q | viewer_q
        return admin_q  # only admins can change/delete

    def construct_perm_q_anon(self, action: ObjectSpecificAction) -> Q | None:
        return None  # no anonymous access

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: MyModel) -> bool:
        if user.has_instance_role(self.admin_role, obj.instance_config):
            return True
        if action == 'view' and user.has_instance_role(self.viewer_role, obj.instance_config):
            return True
        return False

    def anon_has_perm(self, action: ObjectSpecificAction, obj: MyModel) -> bool:
        return False

    def user_can_create(self, user: User, context: ParentModel) -> bool:
        return user.has_instance_role(self.admin_role, context)
```

Then on the model:

```python
class MyModel(PermissionedModel):
    @classmethod
    def permission_policy(cls) -> MyModelPermissionPolicy:
        return MyModelPermissionPolicy()
```

Key points:
- Policies are instantiated fresh on each call. Do not rely on instance state
  persisting across requests.
- Store role references as instance attributes in `__init__` to avoid importing
  them repeatedly.
- Use lazy imports in `__init__` for the model class itself to break circular
  imports.
- `construct_perm_q` and `user_has_perm` must be consistent: if the Q filter
  would include an object, `user_has_perm` must return `True` for it, and vice
  versa. Inconsistency causes subtle security bugs.

## Role System

Roles (defined in `kausal_common/models/roles.py`) map users to Django Groups
which hold Django `Permission` objects. Policy classes use roles in two ways:

**For Q-object construction** (list queries):
```python
admin_q = self.admin_role.role_q(user, prefix='instance_config')
# Produces: Q(instance_config__admin_group__user=user)
```

`prefix` specifies how to traverse from the queryset's model to the field that
holds the role's group. The role resolves the actual field name.

**For instance-level checks**:
```python
user.has_instance_role(self.admin_role, instance_config_obj)
```

Standard role identifiers (from `nodes/roles.py` and `frameworks/roles.py`):
- `instance_admin_role` — full edit access on an InstanceConfig
- `instance_viewer_role` — read-only on an InstanceConfig
- `instance_reviewer_role` — can comment/review data
- `framework_admin_role` — manages a Framework and its FrameworkConfigs
- `framework_viewer_role` — reads framework data

## Using Permissions in Different Layers

### GraphQL (Strawberry)

**Filtering a list query:**
```python
qs = MyModel.permission_policy().instances_user_has_permission_for(info.context.user, 'view')
```

**Fetching a single object with automatic 404-or-403:**
```python
from kausal_common.strawberry.helpers import get_or_error

obj = get_or_error(info, MyModel, id=some_id, for_action='view')
# Raises NotFoundError if not found or not permitted (avoids 403 information leak)
```

**Checking before a mutation:**
```python
obj.ensure_gql_action_allowed(info, 'change')  # raises GraphQLError if denied

# Or for creation:
pp = MyModel.permission_policy()
pp.gql_action_allowed(info, 'add', context=parent_obj)
```

### REST API (DRF)

Subclass `PermissionPolicyDRFPermission` and implement
`get_create_context_from_api_view`. Set `Meta.model` to the model. HTTP methods
are automatically mapped to actions: `GET→view`, `POST→add`, `PUT/PATCH→change`,
`DELETE→delete`.

```python
class MyPermission(PermissionPolicyDRFPermission['MyModel', 'ParentModel']):
    class Meta:
        model = MyModel

    def get_create_context_from_api_view(self, view):
        return get_object_or_404(ParentModel, pk=view.kwargs['parent_pk'])
```

For nested resources (child under a URL-embedded parent), use
`NestedResourcePermissionPolicyDRFPermission`.

Note: `has_object_permission` returns 404 (not 403) when the user cannot view
an object, to avoid leaking its existence.

### Admin Views (Wagtail / Django Admin)

`PermissionedViewSet` in `kausal_common/admin_site/permissioned_views.py`
wires the permission policy into the standard Wagtail edit/create/delete views.
The model's `permission_policy()` is called by each view to check the
appropriate action before rendering.

For Wagtail hook-registered buttons (e.g. sidebar actions):
```python
pp = MyModel.permission_policy()
if pp.user_can_create(request.user, context=parent_obj):
    buttons.append(...)
```

### QuerySet Shorthand

```python
# In any manager/view code:
MyModel.objects.all().viewable_by(user)
MyModel.objects.all().modifiable_by(user)
```

## Field-Level Visibility

`is_field_visible(instance, field_name, user)` returns `True` if:
1. The field is listed in `model.public_fields` (or `policy.public_fields`), OR
2. The user has `change`, `add`, or `delete` permission on the instance.

This is used in GraphQL resolvers to selectively expose sensitive fields.

## `UserPermissions` Helper

`get_user_permissions_for_instance(user, obj)` returns a `UserPermissions`
Pydantic model with `view`, `change`, `delete` booleans and the list of
`creatable_related_models`. Useful for GraphQL types that expose a `permissions`
field.

## Action Vocabulary

| Action | Meaning |
|--------|---------|
| `view` | Read the object |
| `change` | Edit the object |
| `delete` | Delete the object |
| `add` | Create a new object (model-level, not object-level; uses `user_can_create`) |

`ObjectSpecificAction = Literal['view', 'change', 'delete']`
`BaseObjectAction = Literal['view', 'add', 'change', 'delete']`

Unknown actions passed to `instances_user_has_permission_for` are logged as
errors and return an empty queryset.

## Anonymous Users

`user_is_authenticated(user)` returns `False` for `None`, `AnonymousUser`, or
inactive users. Policies receive the `AnonymousUser` instance in
`construct_perm_q_anon` / `anon_has_perm`. Most policies return `None` / `False`
for all anonymous access; read-only reference data (Framework, Section) returns
`Q()` for `view`.

## File Map

| File | Contents |
|------|----------|
| `kausal_common/models/permission_policy.py` | `ModelPermissionPolicy`, `ModelReadOnlyPolicy`, `ParentInheritedPolicy` |
| `kausal_common/models/permissions.py` | `PermissionedModel`, `PermissionedQuerySet`, `PermissionedManager`, `UserPermissions` |
| `kausal_common/models/roles.py` | Role base classes and `role_q` / `has_instance_role` helpers |
| `kausal_common/api/permissions.py` | DRF integration: `PermissionPolicyDRFPermission`, `NestedResourcePermissionPolicyDRFPermission` |
| `kausal_common/strawberry/helpers.py` | `get_or_error` — permission-aware single-object fetch for GraphQL |
| `kausal_common/admin_site/permissioned_views.py` | Admin view base classes using permission policy |
| `frameworks/permissions.py` | `FrameworkPermissionPolicy`, `FrameworkConfigPermissionPolicy`, etc. |
| `paths/dataset_permission_policy.py` | Dataset hierarchy policies, `InstanceConfigScopedPermissionPolicy` |
| `nodes/models.py` | `InstanceConfigPermissionPolicy` (inline on the model) |
| `orgs/permission_policy.py` | `OrganizationPermissionPolicy` |
| `people/permissions.py` | `PersonGroupPermissionPolicy` |
