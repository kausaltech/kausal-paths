# Keycloak-backed devtool auth and the spec TUI

## Status

2026-08-31: A1 and A2 implemented and tested; A3 and the TUI phases not
started. Client topology decided and provisioned via Pulumi: **one
confidential login client per backend deployment** (exact redirect URI,
per-deployment secret; `aud`/`azp` are validated by the same deployment
that ran the flow, so per-deployment clients need no audience gymnastics)
plus **one realm-wide public devtool client**, whose loopback redirect is
deployment-independent and whose client id every deployment's validator
trusts via `KAUSAL_SSO_DEVTOOL_CLIENT_ID`.

## Goal

An expert terminal tool for data scientists to investigate and edit
instance-related specs — starting with the fields `NodeSnapshot` exposes,
where edits land on `NodeConfig` columns and `NodeConfig.spec`. The tool is a
GraphQL client from day one: locally it executes against the schema
in-process, remotely over HTTP, with the same bearer-token authentication in
both modes. Entity UUID references resolve to human-readable identifiers in
both directions (display and input).

## Decisions already made

These came out of the design discussion; the phases below implement them.

1. **GraphQL only, no ORM writes.** The editor mutation layer
   (`nodes/graphql/editor.py`) already carries the invariants a direct-ORM
   tool would have to duplicate: `record_change` audit entries, the
   ClusterableModel `queryset.update()` workaround, optimistic locking via
   `expected_version` / `draft_head_token`, cache invalidation. Going remote
   later is a transport swap, not a second backend.
2. **Start with the existing query/mutation surface.** A raw
   "give me the spec" escape hatch will likely be wanted eventually, but is
   deferred; gaps are closed by adding mutations, which Trailhead's web
   editor needs anyway.
3. **Edits to yaml-sourced instances are refused server-side**, in the
   mutation layer — not in the TUI — so the guard holds for every client.
   (`NodeSnapshot` is a merged read model for yaml-sourced instances;
   editing through the merge would surprise users.)
4. **Schema-driven editors.** Field editors are generated from GraphQL
   introspection, not hand-built per field. Field descriptions flow from
   Pydantic `Field(description=...)` through Strawberry into the schema;
   missing descriptions surfacing as blank help text is a feature — it shows
   where the models are underdocumented.
5. **Ref metadata lives in the type annotations.** `paths/refs.py` already
   has the `Annotated[...]` pattern (`NodeRef`, `DimensionRef`); it grows
   markers that declare a reference's target entity and resolution
   semantics, projected into the GraphQL schema (named scalars or a
   directive) so a remote client sees the same information through
   introspection as a local one.
6. **Auth via the internal Keycloak**, replacing the `azure_ad` web-login
   backend, plus a second public client for the devtool whose ID tokens the
   API learns to trust. Users must exist and have signed in to the admin
   once before a devtool token works; authorization stays with the existing
   permission policies — the token only authenticates.

## Phase A1 — Switch web login from Azure AD to Keycloak (DONE)

New Keycloak client `kausal-paths` (confidential, authorization code flow)
in the internal realm.

- Add an `OpenIdConnectAuth` subclass for the internal Keycloak (in
  `kausal_common/auth/backends.py`, next to `AzureADAuth`). The generic
  social_core OIDC backend does discovery, JWKS fetch and ID-token
  validation; only `OIDC_ENDPOINT`, `name`, and key/secret settings are
  needed. Backend `name` is **`kausal`** — it fixes both the settings
  prefix (`SOCIAL_AUTH_KAUSAL_*`) and the `UserSocialAuth.provider` value
  that the phase-A2 lookup keys on.
- Route staff logins in `check_login_method` (`admin_site/api.py`) by email
  domain: a new `SOCIAL_AUTH_KAUSAL_EMAIL_DOMAINS` setting (default `[]`;
  compare the lowercased part after the last `@`). The
  `has_usable_password()` check keeps precedence — a configured password is
  deliberate (testing accounts), since users are normally provisioned
  without one. A matching domain then returns method `'kausal'` instead of
  falling through to `'azure_ad'`. Non-matching emails keep the existing
  routing. The
  login frontend needs the `'kausal'` → begin-URL mapping; the
  cross-cluster check passes `method` through opaquely and needs no change.
- Replace `kausal_common.auth.backends.AzureADAuth` in
  `AUTHENTICATION_BACKENDS` (`paths/settings.py`) and add the
  `SOCIAL_AUTH_<NAME>_KEY/SECRET` settings sourced from env.
- Existing team members carry `UserSocialAuth` rows under the Azure
  provider. First Keycloak login attaches to the existing user through
  `kausal_common.auth.pipeline.find_user_by_email`, already present in
  `SOCIAL_AUTH_PIPELINE`. Email-based association is safe here because the
  IdP verifies emails; confirm the Keycloak realm enforces verified emails
  before flipping.
- `kausal_common` is a shared submodule (Watch uses it too); the backend
  addition must not assume Paths-side settings exist. Keep Azure AD code in
  place until the switch has soaked.

Deliverable: team members sign in to the Wagtail admin through Keycloak,
and each login creates/refreshes the `UserSocialAuth(provider, uid=sub)`
association that phase A2 depends on.

## Phase A2 — Trust devtool ID tokens as API bearers (DONE)

New Keycloak client `kausal-paths-devtool` (public, PKCE, no secret) in the
same realm.

Implemented as `authenticate_devtool_id_token()` in
`kausal_common/auth/tokens.py`, called first from
`authenticate_from_authorization_header()` — that (via the ASGI
`GeneralRequestMiddleware`) is the live GraphQL bearer path;
`authenticate_api_request` turned out to have no Paths callers. The
validation backend is `KausalDevtoolAuth` (`kausal_common/auth/backends.py`),
a `KausalAuth` subclass with its own `SOCIAL_AUTH_KAUSAL_DEVTOOL_*` settings
and `VALIDATE_AT_HASH = False` — the bearer *is* the ID token, and the access
token of the same grant never reaches the API, so the at_hash binding cannot
be checked server-side. `SOCIAL_AUTH_KAUSAL_DEVTOOL_ID_TOKEN_ISSUER` is set
to the realm URL (the Keycloak issuer), skipping a discovery round trip.
Design notes as planned and verified:

- Instantiate the OIDC backend configured with `client_id =
  kausal-paths-devtool` via `load_strategy()` (no request needed) and
  validate with **`decode_and_validate_id_token(id_token, None)` +
  `validate_temporal_claims()`** — the same pair the library uses for
  refresh-returned tokens. Not `validate_and_return_id_token()`: that calls
  `validate_claims()`, which requires a nonce stored in the strategy's
  association storage and always fails for a bearer presented over the API.
- This gives signature (JWKS), issuer, audience, `azp`, `exp`/`nbf`, and
  `iat` max-age checks. JWKS is already cached (`@cache(ttl=86400)`) with
  automatic refetch on unknown `kid`, so key rotation is handled and no
  extra caching layer is needed.
- **No audience mapper.** A devtool token with `aud` extended to
  `kausal-paths` would still carry `azp: kausal-paths-devtool`, and
  social_core's `validate_authorized_party` rejects any `azp != client_id`
  (correctly, per OIDC Core). Validating against the devtool client's own
  audience gives the same trust boundary — "minted for a client this API
  accepts bearers from" — without fighting the spec. A token minted for any
  other internal service fails the `aud` check.
- User association is a bare lookup, no pipeline:
  `UserSocialAuth.objects.get(provider='kausal', uid=claims['sub'])`.
  Keycloak's `sub` is realm-scoped, so the association created by web login
  through `kausal-paths` matches tokens minted by `kausal-paths-devtool`.
  Unknown `sub` → refuse with an error telling the user to sign in to the
  admin once first. No user is ever created on this path.
- Routing: the validator runs first but self-selects with a cheap
  unverified peek at `iss` — only JWTs claiming the internal realm are
  handled; opaque local-AS tokens and C4C JWTs fall through unchanged to
  the oauthlib path. A token that *does* claim our issuer is handled fully
  (user or error), never passed on.

Deliverable: `curl -H "Authorization: Bearer <id_token>" .../v1/graphql/`
executes as the associated user, locally and on deployments.

## Phase A3 — Devtool-side login flow (first version in `tools/paths_devtool.py`)

Implemented as a plain OAuth client (no Django imports), invoked as
`python -m tools.paths_devtool {login,whoami,token}`: PKCE flow with a
loopback callback on port 8765 (`http://127.0.0.1:8765/callback` must be
registered on the Keycloak client), silent refresh keyed to both `exp` and
the server's 600 s `iat` freshness bound, and a `me { email }` GraphQL smoke
test. `token` prints a fresh ID token for curl use. **Deviation from the
plan below: tokens are cached in a 0600-mode file**
(`~/.config/kausal-paths-devtool/tokens.json`), not the OS keyring — same
posture as `gh` without a keyring backend. Moving to `keyring` (new
dependency) is an open follow-up.

- Authorization code + PKCE against `kausal-paths-devtool`: open the
  browser, catch the redirect on a localhost loopback port, exchange the
  code. No social_core on the client side; this is ~100 lines with `httpx`.
- Keycloak ID tokens are short-lived and social_core additionally enforces
  `iat` within `ID_TOKEN_MAX_AGE` (600 s), so the client must refresh
  silently and frequently with the refresh token, re-presenting a fresh ID
  token. Refresh token goes in the OS keyring; never on disk in plaintext.
- Device flow is a later fallback for SSH sessions; don't build it in v1.
- Local escape hatch so auth never blocks TUI development: an `--as-user
  <email>` flag using `dangerously_force_authenticated_user()`
  (`kausal_common/auth/tokens.py`), which is already gated to
  `is_development_environment() and DEBUG`.

## Phase B — TUI foundation

Textual app under `tools/spec_tui/`, launched as `python -m tools.spec_tui`
(module form, same reasoning as `tools/debug_instance`). Textual goes in the
dev dependency group.

- **One client, two transports.** Remote: HTTP POST to `/v1/graphql/`.
  Local: `django.test.Client` against the same path — in-process, no
  server, but through the full middleware stack, which is how the
  `test_graphql` management command already executes queries. Because the
  local transport goes through middleware, the same `Authorization` header
  and the `@instance(version:)` directive plumbing
  (`DetermineInstanceContextExtension` → `gql_change_operation` reading
  `user` / `expected_version` off the context) work identically in both
  modes. No third context-construction path.
- **Optimistic locking from the start.** Track `draftHeadToken`, send the
  expected version with mutations, and treat a version conflict as "someone
  else (likely a Trailhead session) edited — reload".
- **Save semantics.** `NodeEditorMutation.update` resolves its return value
  through a full draft instance rebuild (`_resolve_runtime_node` →
  `require_instance(source=DRAFT, refresh=True)`), so an edit that breaks
  the parse errors in the same round trip — but after the write committed.
  The TUI must present post-write errors as "the draft is now broken,
  here's why", not "your edit was rejected".
- Server-side change (small, precedes or accompanies this phase): editor
  mutations refuse instances whose `config_source` is yaml, if they don't
  already.

## Phase C — Schema-driven editors and ref resolution

- Build field editors from GraphQL introspection of the node/spec types:
  type → widget, description → help text, enums → selects. Sprinkle custom
  widgets only where the generic rendering is genuinely inadequate.
- Extend `paths/refs.py` with introspectable reference markers on the
  `Annotated[...]` pattern: target entity kind, UUID- vs identifier-based,
  and an explicit marker for intra-spec identities (input-port ids) that
  resolve to no DB row. Project these into the GraphQL schema as named
  scalars (e.g. `NodeUUID`) or a schema directive, generated from the
  annotation metadata — one source of truth that survives going remote, and
  that Trailhead's web editor can consume too.
- Resolution is bidirectional: display UUID → identifier + name; input
  identifier → UUID (with completion). Following a reference navigates to
  that entity's view; v1 only renders node views, but screens are keyed by
  entity ref so other entity kinds slot in.

## Non-goals for now

- Raw set-spec escape hatch (decide later whether it belongs in the public
  schema at all).
- Publish / revert from the TUI (`revert_to_published` is still
  `NotImplementedError`; the agreed revert-to-published rule lands
  separately).
- Editing yaml-sourced instances, in any client.
- Device flow, non-Kausal IdPs, token-granted authorization.

## Open questions

- Keycloak realm details: does the realm enforce verified emails (gates the
  A1 flip)? Token lifespans — if ID-token lifetime is much shorter than the
  600 s `ID_TOKEN_MAX_AGE`, the refresh cadence follows the token, not the
  setting.
- Whether Watch wants the same A2 validator; if so it should live entirely
  in `kausal_common` behind a settings flag from the start.
