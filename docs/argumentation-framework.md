# Paths as a Framework for Climate Argumentation

*Produced by Claude Opus 5.5 on 2026-11-14.*
*Responsible: Jouni Tuomisto.*

This note states what Paths assumes when it is used to argue for or against climate actions, and
what follows from those assumptions. It is for expert users who build models, facilitate their
use, or explain results to decision-makers. The reasoning behind each point is developed
elsewhere (see [Where this is developed](#where-this-is-developed)); this page collects the
principles in one place.

## 1. Contribution, not consensus

International climate negotiations, and many local ones, work by consensus: an agreement needs
every party, so the outcome is what the most reluctant party will accept. Paths starts from a
different question. **Anyone may hold any opinion. What matters is whether the opinion can
contribute to a shared, global framework.**

A contribution is admitted on coherence and physics, never on popularity. A position everyone
else dislikes, but which is coherent, stays in. One that is incoherent is not admitted, and the
platform says why. The aim is a systematic description of what "we" believe, where "we" are the
participants whose contributions are coherent. Persuasion is a side effect: a participant shown
why a claim was not admitted may drop it.

Popularity is still worth knowing, and it is collected (section 3). It just does not decide what
is admitted.

## 2. What is fixed in advance

Only four properties are fixed. Everything else is open to argument.

1. **The framework is global.** Greenhouse gases have the same physical effect whoever emits
   them and wherever. The atmosphere is shared, so the framework is too. Any answer to how the
   task of reducing emissions is shared out, "nobody shares it out" included, has a global
   consequence, because the tonnes add up regardless.
2. **It is internally consistent.** A position must hold when everyone applies it. Shares of the
   budget must add up to no more than the budget; every tonne is counted exactly once; nobody
   uses one rule for themselves and another for others.
3. **Every argument has an explicit chain from the action to what is valued.** This is what a
   Paths model is: a computational Theory of Change. The chain may be long (the action changes
   traffic, traffic changes emissions and air quality, those change health, and health is
   valued) or it may have no causal steps at all, when the value judgement is about the action
   itself (section 3, *Short chains*). What is not admitted is a claimed *effect* with no path
   to it: an action is credited with changing an outcome only if the model traces how.
4. **Factual claims are open to checking.** Whoever takes part accepts that their factual
   premises may be tested against evidence. This is the minimum asked of participants.

## 3. How disagreement is represented

Disagreement is not resolved; it is located.

* **About facts.** Each disputed fact becomes a hypothesis: a category of a `hypotheses`
  dimension, or a scenario with different parameter values. Later each hypothesis gets a
  probability. Evidence matters when it changes a probability *in a way that could change a
  decision* (its value of information); a claim that moves no decision is noted, not promoted.
* **About values.** Each coherent set of value choices is a *value framework*. A policy is
  tested against every admitted value framework:
  * compatible with all of them: **robust**, and nobody on the platform can object to it on
    these grounds;
  * compatible with some: **negotiable**, and the pattern shows exactly which value choice is
    at issue;
  * compatible with none: a strong reason to **reject** it.
* **Short chains.** Some important political theories of change have no causal steps: an
  action and a value judgement about the action itself. The clearest case outside climate
  policy is abortion, where "taking this action is a sin" judges the act, not its
  consequences. Climate policy has such judgements too: "it is wrong to put a price on
  nature", "carbon offsets are bought indulgences", "geoengineering is hubris". These are
  valid chains, only very short, and they enter as value choices. They still face the
  consistency test of section 2: the judgement must apply to everyone who takes the same
  action, the speaker included.

  Many apparently direct judgements are shorthand for an implicit theory of change.
  "Putting any money into X is plain stupid" usually means "X will not work" or "the money
  would do more good elsewhere", and both of those are causal claims that can be written out
  and checked. Asking which one is meant is one of the most useful things a facilitator can
  do: the judgement either stands as a value about the act, or it becomes a hypothesis.
* **Does it matter?** Running the model under each hypothesis and each value framework shows
  whether a disagreement changes the conclusion. Many do not. A new value framework adds to the
  shared understanding only if it changes at least one verdict; one that changes nothing is
  already represented.

The verdicts need no weighting of one value framework against another and no count of how many
people hold each. A robust verdict holds under every admitted framework, so admission should ask
for no more than coherence: the more diverse the admitted frameworks, the stronger a robust or a
rejecting verdict.

**How many people hold each value framework is collected all the same.** It is the input to a
second layer of analysis on top of the verdicts. Two examples:
* For a negotiable action, popularity shows how wide its support is likely to be and which
  value choice the negotiation will turn on.
* For compensation (section 4), it shows how large the opposing group is.

The two layers stay apart. Admission and verdicts rest on coherence alone, so they hold however
the numbers change. Popularity describes the political landscape that those verdicts land in.

## 4. A consequence: the advice does not depend on who decides

Nothing in sections 2 and 3 refers to the decision-maker. A model of congestion charges traces
their effects on emissions, the city budget, health and traffic, and who gains and who loses. It
says nothing about whether the mayor, the city board or the council decides, nor about which
party could trade its vote for a concession elsewhere. Its advice is conditional:
**implement this action if you hold these values and accept these facts**, or, for a robust
verdict, implement it whichever admitted values you hold.

The advice is therefore the same for anyone in power. It does not change with the voting
system, the number of parties or the last election result. Much of political debate is about
who is in a position to do what; this framework is deliberately insensitive to that, and so it
can bring some consistency to the debate: a party that changes its position when it changes
from opposition to government has to show which fact or value changed.

Three qualifications keep this claim accurate.

* **The advice depends on the decision-maker's authority, though not on who holds it.** The
  available actions depend on competence: a city can set a congestion charge but not a national
  carbon price. The set of alternatives that the advice compares is defined by the office, so
  the advice is the same for whoever holds that office.
* **Politics enters as causal facts, not as legitimacy.** Whether a successor is likely to
  reverse a decision, or whether public opposition delays implementation, changes the expected
  effect of an action. Those are facts about the world and belong in the model like any other.
  What stays outside is who has the right to decide.
* **A package is an action like any other.** An action rarely travels alone. It may come with a
  concession to a party whose vote it needs, or with compensation for those who lose by it:
  residents of the charged zone, small businesses inside it, commuters who have no alternative.
  The package (charge plus compensation) can be modelled and judged as a whole. The framework is
  blind to who negotiated the package, not to what it does.

Compensation deserves more than a footnote, because the power to block or slow an action is
usually greater than the power to push it through. An action that produces a greater good while
leaving a group clearly worse off creates that group's reason to oppose it, and opposition costs
time: a delayed action emits more along the way, and often locks in more capital stock that keeps
emitting. A successful policy therefore asks early who loses, by how much, and what fair
compensation would cost. The model can answer the factual parts. It shows who gains and who loses
by each action, what a given compensation costs, and, through the `action_delay` parameter, what
a delay of a given length costs in emissions (the extra lock-in of capital stock is not yet
modelled, so this cost is understated). Whether compensation is owed, and how much counts
as fair, are value choices, and the value frameworks of section 3 judge them like any other.

## 5. The corresponding weakness

Advice has no political power of its own. It is one voice among many, and a consistent argument
does not win just by being consistent. It matters only if people gather around it and take it
into a political process, where its tools are useful: showing which premises an opposing view
needs, testing whether the view survives its own assumptions, and showing when it is
inconsistent. Paths offers better tools to those with better arguments. It does not replace the
political work of using them.

## 6. What exists and what does not

| Piece | State |
|---|---|
| Causal models of actions and outcomes | The core of Paths |
| `hypotheses` dimensions and scenarios for disputed facts | Available; see [`paths-purpose-and-design.md`](paths-purpose-and-design.md) |
| Argument nodes documenting objections | Available; moving to output ports, see [`plans/argument-links-on-output-ports.md`](plans/argument-links-on-output-ports.md) |
| Value frameworks with robust / negotiable / rejected verdicts | Implemented for the fair share of the global CO₂ budget, in [`configs/modules/fair_share/`](../configs/modules/fair_share/) |
| Probabilities over hypotheses, value of information | Not built |
| Counting how many people hold each value framework | Planned, not built |
| Comparing verdicts before and after a contribution; a store for contributions | Not built |

## Where this is developed

* [`paths-purpose-and-design.md`](paths-purpose-and-design.md): model philosophy, the
  objection typology (empirical disputes, value conflicts, interests presented as values) and
  the `hypotheses` dimension.
* [`configs/modules/fair_share/objections.md`](../configs/modules/fair_share/objections.md):
  objections to a fair-share argument, including "I reject a global system everyone must obey".
* Kausal's value catalogue (paths-data, `modules/values/`, internal): what a global value
  framework requires and why (`docs/value-frameworks.md`), the contribution test
  (`docs/shared-understanding-plan.md`), and the decisions behind both (`docs/decisions.md`).
