# CLAUDE.md

## What Is Kausal Paths?

Kausal Paths is a computational modeling platform for urban climate action
planning. Cities and regions use it to build quantitative emissions models:
directed acyclic graphs (DAGs) of calculation nodes that take historical
data (energy consumption, vehicle mileage, building stock) and project
future emissions under different policy scenarios.

The core question Paths answers: "If we implement these climate actions,
what happens to our emissions by 2030?"

Each city gets a model instance — a self-contained DAG with its own data,
parameters, and scenarios. The models are multilingual (Finnish, German,
English, etc.), unit-aware (pint handles dimensional analysis), and
scenario-driven (toggle actions on/off, adjust parameters, compare outcomes).

The platform serves two audiences:
- **Climate coordinators** in city governments, who use the public UI
  to explore scenarios and communicate with decision-makers
- **Model builders** (currently Kausal staff, soon the coordinators
  themselves via "Trailhead"), who define the computation graphs

The backend is Django + Wagtail, the computation engine uses Polars,
and the API is GraphQL (migrating from Graphene to Strawberry).

## Development Commands

### Environment Setup (one time)
```bash
# If mise is not installed:
curl https://mise.run | sh

# Ensure tooling is installed and allow "mise prepare"
mise settings experimental=true
mise install

# Install Python interpreter, venv and dependencies
mise prepare  # venv will live in `.venv/`
```

### Core Django Commands
- `python manage.py runserver` - Start development server
- `python manage.py migrate` - Run database migrations
- `python manage.py shell_plus` - Enhanced Django shell (from django-extensions)
- `python manage.py shell_plus --quiet-load -c "print(Plan.objects.last())"` - Run one-off Python with all models auto-imported; useful for quick DB lookups

### Testing and Quality
```bash
# Run tests (--reuse-db avoids recreating the test database each run)
python -m pytest --reuse-db

# Only if the tests are complaining about missing static files:
python manage.py collectstatic

# Run specific test file
python -m pytest --reuse-db path/to/test_file.py

# Run mypy type checking (filter out known baseline errors)
mypy . | mypy-baseline filter

# Run ruff linting
ruff check .

# Run ruff formatting
ruff format .

# Auto-fix import ordering & other safe fixes
ruff check --fix file.py
```

### Interactive Debugger (`/debugging-code` skill)

When a bug involves **control flow through code you didn't write** — library
internals, metaclass machinery, framework callbacks — use the `debugging-code`
skill instead of writing throwaway Python snippets. The skill uses `dap`, a
CLI DAP debugger.

**When to reach for it:**
- You need the **call stack** across library boundaries ("who called this?",
  "how did execution reach here?")
- You need to **inspect locals in upper frames** (e.g. what did the caller
  pass? what does the framework's internal state look like at this point?)
- You've written 3+ exploratory snippets without converging — a single
  conditional breakpoint in the right place will likely answer it faster

**When snippets are better:** testing a hypothesis about a single value or
return type, quick attribute checks, anything where the question is about
*what* not *how we got here*.

### GraphQL API
- GraphQL endpoint: `http://127.0.0.1:8000/v1/graphql/`

```bash
# Export schema and diff against production
python manage.py export_schema paths.schema > schema.graphql
pnpx @graphql-inspector/cli diff https://api.paths.kausal.dev/v1/graphql/ schema.graphql
```

## Architecture Overview

### Core Components

#### Node-Based Calculation System
- **Nodes**: Core calculation units organized in a directed acyclic graph
- **Actions**: Special nodes representing climate actions with configurable parameters
- **Datasets**: Data sources (JSON, Parquet files) feeding into nodes
- **Dimensions**: Multi-dimensional data handling (time, geography, sectors)
- **Instances**: Complete calculation setups for specific cities/regions

#### Key Django Apps
1. **`nodes/`** - Core calculation engine, node graph system, emissions calculations
2. **`frameworks/`** - Framework configuration management (GPC, NZC, etc.)
3. **`pages/`** - Wagtail CMS integration for content management
4. **`admin_site/`** - Custom admin interface with authentication
5. **`users/`** - User management with framework-specific roles
6. **`params/`** - Parameter management for nodes and actions
7. **`kausal_common/`** - git submodule for code that is shared between Kausal Paths and Kausal Watch


#### Data Flow
1. **Configuration**: YAML files in `/configs/` define instance configurations
2. **Data Processing**: Polars/Pandas for data manipulation with Pint for units
3. **Calculations**: Node graph executes calculations with real-time updates
4. **API**: GraphQL provides unified access to results and configurations

#### Internationalization (i18n)
Three co-existing translation mechanisms (django-modeltrans, Wagtail locales,
`TranslatedString`), each with its own language code format. See
[`docs/architecture/i18n.md`](docs/architecture/i18n.md) for the full picture.

#### Permission Policies
Every model requiring access control inherits from `PermissionedModel` and
implements `permission_policy()`. Policies extend `ModelPermissionPolicy` and
express access rules as ORM Q objects (for list filtering) and per-instance
booleans (for object-level checks). See
[`docs/architecture/permissions.md`](docs/architecture/permissions.md) for
the full architecture, usage patterns across GraphQL/DRF/admin, and how to
implement a new policy.

### Important Patterns

#### Multi-Instance Architecture
- Each city/region has its own `Instance` with specific configuration
- Instance configurations stored in database as `InstanceConfig`
- Context objects provide runtime parameters and scenarios

#### Framework System
- **Framework**: Calculation methodologies (GPC, NZC, etc.)
- **Measures**: Templates for actions within frameworks
- **Sections**: Organizational units within frameworks

#### Real-Time Features
- Django Channels for WebSocket connections
- Async calculation engine with live progress updates
- Caching system for performance optimization

### Design Principles

See [`docs/architecture/principles.md`](docs/architecture/principles.md) for
full rationale. In brief:

1. **Design for where things are going.** Every fix reshapes the codebase.
   Move toward the long-term shape, not just past the immediate breakage.
2. **Root-cause before fix.** Understand *why* before deciding *what*. A
   workaround that suppresses the symptom is debt with interest.
3. **Type system carries the contract.** Required fields stay required.
   Abstract classes use `@abstractmethod`. Prefer construction-time errors.
4. **Don't confabulate mechanisms.** Distinguish known from inferred from
   guessed. "I don't know yet" beats a plausible-sounding wrong answer.
5. **Migration paths: explicit and contained.** Backward compat goes behind
   explicit entry points (e.g. `from_yaml_config`), not in the common path.

### Code Conventions

#### Python
- Type hints required (mypy checking enabled)
- Use `from __future__ import annotations` for forward references
- Follow Django model patterns with proper managers and querysets
- Use Pydantic for data validation where appropriate

#### Ruff & TYPE_CHECKING imports
- Ruff config lives in `kausal_common/configs/ruff.toml`
- When a base class or decorator causes ruff's `TC004` rule to incorrectly
  demand moving imports out of `TYPE_CHECKING` (which would cause circular
  imports), add the decorator/base class to `runtime-evaluated-decorators`
  or `runtime-evaluated-base-classes` in the `[lint.flake8-type-checking]`
  section of that file. This tells ruff that the listed classes/decorators
  evaluate type annotations at runtime (via Pydantic), so imports used only
  in annotations of those classes are allowed to stay behind `TYPE_CHECKING`.

#### Models
- All models inherit from appropriate base classes (`UUIDIdentifiedModel`, `PathsModel`)
- Use proper type annotations for fields and relationships
- Implement permission policies for access control
- **Reverse FK managers must be explicitly annotated** on the model class using
  the helpers in `kausal_common/models/types.py` (e.g. `RevMany[ChildModel]`).
  Django auto-generates these managers at runtime, but the type checker cannot
  see them. Never use `getattr()` or `# type: ignore` to work around missing
  reverse managers — add the annotation instead.

#### GraphQL
- Uses both Graphene and Strawberry (migration in progress; see `docs/graphene-to-strawberry-migration.md`)
- Maintain schema consistency across different apps
- Include proper error handling and validation

### Key Files and Directories

#### Configuration
- `/configs/` - Instance configurations and framework definitions
- `pyproject.toml` - Project dependencies and tool configurations

#### Data and Processing
- `/datasets/` - Data files and processing modules
- `/nodes/actions/` - Action implementations
- `/nodes/finland/`, `/nodes/ch/` - Region-specific modules

#### Templates and Static Files
- `/templates/` - Django templates
- `/static/` - Frontend assets
- `/locale/` - Translation files

### Development Notes

#### Database
- PostgreSQL required for production
- Uses Django migrations for schema changes
- Atomic transactions enabled by default

#### Dependencies
- Python 3.13+ required
- Heavy use of scientific computing libraries (Polars, Pandas, NumPy)
- Wagtail CMS for content management
- Django Channels for WebSocket support

#### Testing
- Use pytest with Django plugin
- Factory Boy for test data generation
- **Always call `Factory.create()`** (not bare `Factory()`), so the return type
  is the model class, not the factory class. This matters for mypy and attribute access.
- Use `PathsTestClient` from `paths/tests/graphql.py` for new GraphQL tests
  (replaces the old `graphql_client_query_data` fixture). Key methods:
  `query_data()` (asserts no errors, returns `dict`), `query_errors()` (asserts
  errors present, returns the error list).
- Test files should be in respective app directories

#### Performance
- Use Polars for large data processing (preferred over Pandas)
- Implement proper caching strategies
- Monitor memory usage in calculation-heavy operations

### Common Tasks

#### Debugging Calculations

```bash
# Compute & show model outputs for a node
python load_nodes.py -i <instance-id> --node <node-id>
```

### Trailhead Migration

@docs/trailhead/tools.md
