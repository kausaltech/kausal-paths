# Design Principles

These principles guide architectural and implementation decisions in Kausal
Paths. They emerged from practice, not theory — each one reflects a real
cost that was paid when the principle was violated.


## 1. Design for where things are going

Every change reshapes the codebase. When solving an immediate problem,
consider whether the solution moves the system toward its long-term shape
or away from it. A fix that does both is worth the extra thought. A fix
that makes the next change harder is a debt, even if it works today.

**Example:** Adding `primary_language` and `other_languages` to
`InstanceSpec` fixed the DB deserialization problem, but it also made specs
self-contained and serializable outside Django — a requirement for the
upcoming Trailhead editor. The fix and the roadmap pointed the same
direction.

**Why:** Data models and interfaces are expensive to change once they have
consumers. Shaping them toward the target state early avoids painful
migrations later.


## 2. Understand the root cause before applying a fix

When something breaks, invest in understanding *why* before deciding *what*
to do. A workaround that suppresses the symptom is not a fix — it's a debt
with interest. The time spent understanding the mechanism almost always
pays for itself, either in a better fix or in knowledge that prevents the
next bug.

**Example:** The `Parameter` instantiation bug could have been "fixed" by
calling `model_rebuild(force=True)` after class creation. Instead, tracing
it to re-entrant `model_rebuild` during Python 3.14 annotation evaluation
revealed the actual mechanism (self-referential `PrivateAttr` annotation
triggering `__class_getitem__`), which led to a targeted fix and a
reproducible bug report for Pydantic upstream.

**Why:** Workarounds accumulate and interact. Root-cause fixes resolve.


## 3. Let the type system carry the contract

Required fields should be required. Abstract classes should be abstract.
Prefer a construction-time error over a runtime surprise. When an invariant
matters, encode it in the type system so that violations are caught by the
toolchain, not by a user in production.

**Example:** `InstanceSpec.primary_language` is a required field with no
default. `Parameter` uses `ABC` with `@abstractmethod` on `clean()`. Both
choices cause immediate, clear errors when the contract is violated, rather
than producing objects that look valid but behave wrong downstream.

**Why:** The cost of strictness is paid once, at the call site that needs
fixing. The cost of leniency is paid every time someone misunderstands the
invariant — and the error shows up far from the cause.


## 4. Don't confabulate mechanisms

When explaining why something works or doesn't, distinguish between what
you know, what you're inferring, and what you're guessing. A
plausible-sounding explanation that turns out to be wrong is worse than
saying "I don't know yet" — it sends the investigation in the wrong
direction and wastes everyone's time.

This applies to code comments and commit messages too: don't write a causal
explanation unless you've verified the mechanism.

**Why:** Confidence without verification is expensive. It wastes time
following false leads and erodes trust in future explanations.


## 5. Keep migration paths explicit and contained

When backward compatibility is needed, isolate it behind an explicit entry
point rather than weaving it into the common path. The common path should
enforce the target contract; the legacy path should be visible, named, and
deletable.

**Example:** `I18nBaseModel.from_yaml_config()` handles the YAML-era
`name_en` suffix convention as a separate classmethod. The standard
`model_validate()` path stays strict and knows nothing about suffixes.
When the YAML configs are fully migrated, `from_yaml_config` can be
deleted without touching any other code.

**Why:** If backward compatibility lives in the common path, you can't
distinguish "this works because it's correct" from "this works because the
compat shim caught it." Migration bugs hide until production, and the compat
code outlives the migration because nobody is sure it's safe to remove.


## 6. Keep change-sensitive knowledge with the behavior it describes

The module that owns variable behavior should also own the description of
that behavior's semantic dependencies and identity. Consumers may compose
those descriptions, but should not reconstruct them by importing concrete
implementations and branching on their types.

A type switch in a consumer often creates a shadow implementation: a second,
necessarily incomplete account of what each implementation reads and does.
Adding or changing an implementation then requires a coordinated edit in a
module with no structural connection to the change. Correctness depends on
someone remembering that hidden coupling.

Prefer an explicit semantic projection on the abstraction. For example, a
transformation can declare the authored configuration and contextual values
that determine its materialized output. The pipeline composes those
declarations in order, while the cache remains responsible for serialization,
hashing, and storage. The transformation does not need to know about Redis,
and the cache does not need to know how a `FilterColumnOp` works.

**Example:** Dataset pipeline caching originally inspected concrete
transformation classes in `DatasetWithFilters` to infer which parameters,
timeline values, and dimensions belonged in the cache key. A change to an
operation could silently leave that parallel model stale. Moving the
materialization identity to the operation makes the dependency explicit and
keeps the change with its semantic owner; the dataset layer only composes the
result.

**Review test:** If this class changes, which other modules must its author
remember to inspect? Imports used primarily for `isinstance` chains or type
switches are a warning that a consequence of the behavior lives outside its
owner.

**Boundary:** This does not mean pushing every downstream concern into the
behavioral class. It should expose semantic facts or dependencies, not cache
keys, hash algorithms, log formats, or storage policy. Each consumer still
owns its mechanism.

**Why:** Things that must change together should be owned together. Things
that merely compose should depend on an explicit contract, not knowledge of
one another's internals.
