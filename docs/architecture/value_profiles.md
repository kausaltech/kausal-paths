# Moral Impact Assessment Tool: Mathematical Framework

## Overview

This document specifies the mathematical operationalization of empirically supported
moral principles for use in a scenario comparison tool. Each principle is implemented
as a separate scoring node. The tool is grounded in experimental moral philosophy,
particularly studies using the veil of ignorance paradigm, where participants reason
about distributive justice without knowledge of their own position in the outcome
distribution.

**Core references:**
- Frohlich & Oppenheimer (1992) *Choosing Justice*, University of California Press
- Konow (2003) "Which Is the Fairest One of All?" *Journal of Economic Literature* 41(4)
- Atkinson (1970) "On the Measurement of Inequality" *Journal of Economic Theory* 2(3)
- Alkire & Foster (2011) *Journal of Public Economics* 95(7-8)

---

## Part I: Foundation

### 1.1 Welfare Vector Definition

Before any moral principle can be applied, a welfare measure must be defined for
each group or individual i in the model.

**If single-dimensional** (e.g., income, consumption, QALYs):

$$u_i \text{ is measured directly}$$

**If multi-dimensional** (recommended for impact assessment):

$$u_i = \sum_k w_k \cdot d_{ik}$$

Where:
- $d_{ik}$ = welfare of group $i$ on dimension $k$
- $w_k$ = weight assigned to dimension $k$, with $\sum_k w_k = 1$

Equal weights across dimensions are the defensible default. Sensitivity to
alternative weightings should be tested.

**Source:** Alkire & Foster (2011) for multidimensional operationalization;
Nussbaum (2000) *Women and Human Development* for philosophical grounding
of dimension selection.

---

### 1.2 Normalization

Normalization is required before combining principles or comparing across dimensions.

**If comparing absolute welfare levels:**

$$\hat{u}_i = \frac{u_i - u_{\min}}{u_{\max} - u_{\min}} \in [0,1]$$

**If comparing changes relative to a baseline scenario:**

$$\Delta u_i = u_i(\text{scenario}) - u_i(\text{baseline})$$

**If welfare values can be zero or negative** (required for Atkinson function):

$$u'_i = u_i + |u_{\min}| + \delta$$

Where $\delta$ is a small positive constant to ensure $u'_i > 0$.

---

## Part II: Moral Scoring Nodes

### Node 1: Utilitarian Average (W_U)

#### Description

The utilitarian principle holds that the morally best outcome maximizes total or
average welfare across all individuals or groups. It treats each person's welfare
equally and is indifferent to distribution. Empirically, efficiency preferences
of this type frequently dominate inequality concerns when trade-offs are made
explicit, particularly when the gains to average welfare are large.

This node serves as the baseline reference against which inequality-sensitive
measures are compared.

**Empirical basis:**
- Engelmann & Strobel (2004) "Inequality Aversion, Efficiency, and Maximin
  Preferences in Simple Distribution Experiments" *American Economic Review* 94(4)
- Charness & Rabin (2002) "Understanding Social Preferences with Simple Tests"
  *Quarterly Journal of Economics* 117(3)

#### Mathematical Operationalization

$$W_U = \frac{1}{n} \sum_{i=1}^{n} u_i$$

Where:
- $n$ = number of groups or individuals in the model
- $u_i$ = welfare of group $i$

**Output:** A single scalar. Higher values indicate higher average welfare.
Scenarios are ranked by $W_U$ directly.

---

### Node 2: Maximin / Rawlsian (W_R)

#### Description

The maximin principle holds that a just outcome maximizes the welfare of the
worst-off group, regardless of what happens to others. This is the principle
Rawls (1971) argued rational agents would choose behind the veil of ignorance.
Empirical studies find that pure maximin is rarely chosen in deliberative
experiments, but it represents an important limiting case and is valuable as a
sensitivity bound. It answers the specific question: which scenario is best for
the worst-off group?

**Empirical basis:**
- Rawls (1971) *A Theory of Justice*, Harvard University Press
- Frohlich & Oppenheimer (1992) show this is rarely chosen but provides
  a useful normative reference

#### Mathematical Operationalization

$$W_R = \min_i(u_i)$$

**Output:** A single scalar representing the welfare of the worst-off group.
Scenarios are ranked by $W_R$ directly. Use as a lower-bound sensitivity
check alongside other nodes.

---

### Node 3: Atkinson Social Welfare Function (W_A)

#### Description

The Atkinson function is the most analytically important node in the tool.
It parameterizes continuously across the full spectrum from pure utilitarianism
to pure maximin through a single inequality aversion parameter $\varepsilon$.
This allows the tool to report how scenario rankings change as the moral weight
placed on inequality increases, making moral assumptions explicit and testable
rather than hidden.

The Atkinson index additionally provides a directly interpretable welfare loss
measure: the fraction of total welfare lost due to inequality.

Prioritarianism - giving extra weight to worse-off groups without ignoring
overall welfare - corresponds to intermediate values of $\varepsilon$ and is
the position most consistently supported in empirical moral judgment studies.

**Empirical basis:**
- Atkinson (1970) *Journal of Economic Theory* 2(3) - foundational mathematical
  framework
- Fehr & Schmidt (1999) "A Theory of Fairness, Competition, and Cooperation"
  *Quarterly Journal of Economics* 114(3) - empirical support for inequality
  aversion

#### Mathematical Operationalization

**For $\varepsilon \neq 1$:**

$$W_A(\varepsilon) = \left[ \frac{1}{n} \sum_{i=1}^{n} u_i^{1-\varepsilon} \right]^{\frac{1}{1-\varepsilon}}$$

**For $\varepsilon = 1$ (geometric mean):**

$$W_A(1) = \exp\left( \frac{1}{n} \sum_{i=1}^{n} \ln(u_i) \right)$$

**Inequality index:**

$$A(\varepsilon) = 1 - \frac{W_A(\varepsilon)}{\bar{u}}$$

Where $\bar{u} = W_U$ is the arithmetic mean.

**Interpretation of $\varepsilon$:**

| $\varepsilon$ | Moral interpretation | Reduces to |
|---------------|---------------------|------------|
| 0 | No inequality concern | Utilitarian mean |
| 0.5 | Mild prioritarianism | — |
| 1 | Moderate inequality aversion | Geometric mean |
| 2 | Strong inequality aversion | — |
| $\to \infty$ | Only worst-off matters | Maximin |

**Requirement:** $u_i > 0$ for all $i$. Apply shift correction from Section 1.2
if necessary.

**Output:** Run across the parameter range $\varepsilon \in \{0, 0.5, 1, 2, 5, \infty\}$
and report the full ranking profile. Report $A(\varepsilon)$ as the welfare
loss attributable to inequality under each scenario.

---

### Node 4: Floor-Constrained Average (W_FC)

#### Description

The floor-constrained average is the most robustly supported principle in
empirical veil of ignorance studies. Across multiple countries and subject
populations, deliberating groups consistently chose to maximize average welfare
subject to a guaranteed minimum rather than applying either pure maximin or
pure utilitarianism. The structure of this principle - not its specific threshold
level - showed strong cross-cultural replication.

This node reflects the moral judgment that large average gains do not justify
outcomes below a minimum acceptable threshold.

**Empirical basis:**
- Frohlich, Oppenheimer & Eavey (1987) "Choices of Principles of Distributive
  Justice in Experimental Groups" *American Journal of Political Science* 31(3)
- Frohlich & Oppenheimer (1992) *Choosing Justice*, University of California
  Press - replicated across US, Canada, Poland, Australia

#### Mathematical Operationalization

**Hard constraint version** (lexicographic - use when the floor is treated as
inviolable):

$$W_{FC}(f) = \bar{u} \quad \text{if } \min_i(u_i) \geq f, \quad \text{otherwise scenario is inadmissible}$$

**Soft penalty version** (continuous - use when comparing scenarios on a
common scale):

$$W_{FC}(f, \lambda) = \bar{u} - \frac{\lambda}{n} \sum_{i=1}^{n} \max(0, f - u_i)$$

Where:
- $f$ = floor threshold
- $\lambda$ = penalty weight; set $\lambda > 1$ to make the floor genuinely
  binding relative to average gains; $\lambda = 2$ is a reasonable default

**Setting the floor threshold $f$ - three defensible approaches:**

| Approach | Formula | Use case |
|----------|---------|----------|
| Absolute | Fixed subsistence/poverty threshold in model units | When absolute deprivation is the concern |
| Relative | $f = \alpha \cdot \bar{u}_{\text{baseline}}$, e.g. $\alpha = 0.5$ | When relative standing matters |
| Need-based | Derived from Node 6 need thresholds | When multidimensional deprivation is modeled |

**Output:** Under the hard constraint version, flag inadmissible scenarios
before ranking. Under the soft penalty version, rank directly by $W_{FC}$.
Report the floor threshold used and test sensitivity to $\alpha$.

---

### Node 5: Sen-Gini Social Welfare Function (W_Sen)

#### Description

The Sen-Gini function combines average welfare and the Gini coefficient of
inequality into a single measure. It provides a theoretically grounded
alternative to the Atkinson function that is more transparent in its components
and directly connected to the widely used Gini coefficient. While conceptually
related to Node 3, it is retained as a separate node because the Gini captures
a different aspect of the distribution - specifically, pairwise differences
between all groups - and is more familiar to policy audiences.

**Empirical basis:**
- Sen (1974) "Informational Bases of Alternative Welfare Approaches"
  *Journal of Public Economics* 3(4)
- Gini (1912) - foundational inequality measure

#### Mathematical Operationalization

$$W_{Sen} = \bar{u} \cdot (1 - G)$$

Where the Gini coefficient is:

$$G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |u_i - u_j|}{2n^2 \bar{u}}$$

**Interpretation:**
- $G = 0$ (perfect equality): $W_{Sen} = \bar{u}$
- $G = 1$ (maximum inequality): $W_{Sen} \to 0$
- The gap between $W_U$ and $W_{Sen}$ for a given scenario is the welfare
  cost of inequality under that scenario

**Output:** A single scalar per scenario. The difference $W_U - W_{Sen}$
is directly interpretable as the welfare cost of inequality.

---

### Node 6: Need Satisfaction (W_N)

#### Description

The need principle is empirically distinct from the floor constraint. While
the floor is defined in terms of aggregate welfare, the need principle addresses
specific substantive dimensions - such as nutrition, shelter, health, or
physical safety - that are required for basic human functioning regardless of
contribution or overall welfare level. Empirical studies consistently support
the need principle as an independent distributive criterion alongside efficiency
and accountability.

This node is operationalized following the Alkire-Foster counting approach,
which underlies the UN Multidimensional Poverty Index and has been extensively
validated.

**Empirical basis:**
- Konow (2003) "Which Is the Fairest One of All?" *Journal of Economic
  Literature* 41(4) - identifies need as a robust independent principle
- Deutsch (1975) "Equity, Equality, and Need" *Journal of Social Issues* 31(3)
- Alkire & Foster (2011) *Journal of Public Economics* 95(7-8) -
  operationalization framework

#### Mathematical Operationalization

Let $n_k$ denote the threshold for dimension $k$, and $u_{ik}$ denote the
welfare of group $i$ on dimension $k$.

**Binary (threshold) version:**

$$NS_i = \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}(u_{ik} \geq n_k)$$

$$W_N = \frac{1}{n} \sum_{i=1}^{n} NS_i$$

**Continuous version** (smoother for optimization and comparison):

$$NS_i = \frac{1}{K} \sum_{k=1}^{K} \min\left(1, \frac{u_{ik}}{n_k}\right)$$

$$W_N = \frac{1}{n} \sum_{i=1}^{n} NS_i$$

Where:
- $K$ = number of need dimensions
- $NS_i \in [0,1]$ = need satisfaction score for group $i$
- $W_N \in [0,1]$ = population-level need satisfaction score

**Output:** $W_N \in [0,1]$. A score of 1 means all groups meet all need
thresholds on all dimensions. Report dimension-level scores $\frac{1}{n}
\sum_i \mathbf{1}(u_{ik} \geq n_k)$ separately to identify which dimensions
drive shortfalls.

---

### Node 7: Accountability-Adjusted Welfare (W_Acc)

#### Description

The accountability principle reflects the empirically robust moral judgment
that individuals should be compensated for disadvantages arising from unchosen
circumstances (birth conditions, disability, geography) but not for disadvantages
arising from freely made choices. Inequality due to circumstances is morally
arbitrary and should be penalized; inequality due to choices may be acceptable
or even desirable.

This is the empirical counterpart of luck egalitarianism in philosophy, and is
identified by Konow (2003) as one of three robust distributive principles
alongside efficiency and need.

This node is the most complex to operationalize because it requires the model
to distinguish chosen from unchosen factors. This is only feasible if the
model structure encodes causal pathways.

**Empirical basis:**
- Konow (2000) "Fair Shares: Accountability and Cognitive Dissonance in
  Allocation Decisions" *American Economic Review* 90(4)
- Konow (2003) as above
- Roemer (1998) *Equality of Opportunity*, Harvard University Press -
  foundational framework
- Bourguignon, Ferreira & Menéndez (2007) *Journal of Economic Inequality*
  5(2) - practical implementation

#### Mathematical Operationalization

**Step 1 - Decompose welfare into circumstance and effort components:**

$$u_i = f(C_i, E_i)$$

Where:
- $C_i$ = circumstances (unchosen: structural position, endowments, geography)
- $E_i$ = effort or choices (discretionary inputs under individual control)

**Step 2 - Estimate circumstance-predicted welfare:**

$$\hat{u}_i(C) = \mathbb{E}[u \mid C_i]$$

In practice, estimate via regression of $u_i$ on circumstance variables only,
yielding a predicted value $\hat{u}_i(C)$ representing what welfare would be
if determined solely by unchosen factors.

**Step 3 - Apply inequality aversion to the circumstance component only:**

$$W_{Acc} = W_A(\varepsilon) \text{ applied to } \hat{u}_i(C)$$

This penalizes circumstance-driven inequality while leaving effort-driven
variance unpenalized.

**Simplified version** (if full decomposition is not feasible):

Weight groups inversely by their circumstance disadvantage:

$$w_i = \frac{1/\hat{u}_i(C)}{\sum_j 1/\hat{u}_j(C)}$$

$$W_{Acc} = \sum_{i=1}^{n} w_i \cdot u_i$$

**Output:** A scalar per scenario. Compare with $W_A(\varepsilon)$ applied to
$u_i$ directly to quantify the moral difference that the circumstance-choice
distinction makes.

---

### Node 8: Equality Reference (W_Eq)

#### Description

Equal division functions as an empirically robust reference point and default
in distributive experiments. Even when other principles dominate, equal splits
serve as a focal point from which deviations require justification. This node
is included not as a primary moral criterion but as a reference against which
inequality in all other nodes can be assessed.

**Empirical basis:**
- Engelmann & Strobel (2004) as above
- Güth, Schmittberger & Schwarze (1982) "An Experimental Analysis of Ultimatum
  Bargaining" *Journal of Economic Behavior and Organization* 3(4)

#### Mathematical Operationalization

**Variance penalty:**

$$W_{Eq} = -\frac{1}{n} \sum_{i=1}^{n} (u_i - \bar{u})^2$$

Higher (less negative) values indicate distributions closer to equality.

**Normalized equality score:**

$$E_{score} = 1 - \frac{\text{Var}(u)}{\text{Var}_{\max}}$$

Where $\text{Var}_{\max}$ is the maximum possible variance given the same mean,
providing a $[0,1]$ score where 1 is perfect equality.

**Output:** Use primarily for comparison and for reporting alongside other
nodes. It should not typically be the sole ranking criterion.

---

## Part III: Stepwise Procedure

### Step 1: Define the Welfare Measure

1. Identify the outcome variables produced by your model
2. Determine whether welfare is single- or multi-dimensional
3. If multi-dimensional, specify dimensions $k$ and assign weights $w_k$
4. Define whether $u_i$ represents levels or changes relative to baseline
5. Normalize using the appropriate formula from Section 1.2
6. Verify $u_i > 0$ for all groups and scenarios; apply shift correction
   if needed

---

### Step 2: Specify Parameters

For each run, set the following parameters explicitly and record them:

| Parameter | Symbol | Required by nodes | Default |
|-----------|--------|------------------|---------|
| Inequality aversion | $\varepsilon$ | W_A, W_Acc | Run multiple |
| Floor threshold | $f$ | W_FC | Context-dependent |
| Floor penalty weight | $\lambda$ | W_FC (soft) | 2 |
| Need thresholds | $n_k$ | W_N | Context-dependent |
| Dimension weights | $w_k$ | Multi-dim $u_i$ | Equal |
| Circumstance variables | $C_i$ | W_Acc | Model-dependent |

---

### Step 3: Compute All Node Scores

For each scenario $s$ and each node $p$, compute $W_p(s)$.

Organize results in a scenario × node matrix:

| Scenario | $W_U$ | $W_R$ | $W_A(\varepsilon=1)$ | $W_A(\varepsilon=2)$ | $W_{FC}$ | $W_{Sen}$ | $W_N$ | $W_{Acc}$ | $G$ |
|----------|-------|-------|---------------------|---------------------|---------|----------|-------|----------|-----|
| Baseline | | | | | | | | | |
| S1 | | | | | | | | | |
| S2 | | | | | | | | | |

---

### Step 4: Apply Lexicographic Screening (if applicable)

If the hard floor or need threshold is treated as a minimum requirement:

1. **Screen on need:** Eliminate scenarios where $W_N < $ threshold
2. **Screen on floor:** Eliminate scenarios where $\min_i(u_i) < f$
3. **Rank remaining** scenarios by $W_A(\varepsilon)$ or $W_{FC}$ (soft)

Inadmissible scenarios are reported separately with the specific criterion
they violate.

---

### Step 5: Dominance Analysis

Before applying any weighting, identify scenarios that are unambiguously
preferred:

**Strong dominance:** Scenario A dominates B if $W_p(A) \geq W_p(B)$ for
all nodes $p$, with strict inequality for at least one.

**Report:** Which scenarios are dominated and can be eliminated regardless of
moral weights. Dominated scenarios need not be considered further.

---

### Step 6: Sensitivity Analysis

Run the full scoring across the parameter space:

For ε in {0, 0.5, 1, 2, 5, ∞}:
For f in {0.3, 0.5, 0.6} × mean(u_baseline):
Compute W_FC(f) and W_A(ε) for all scenarios
Record ranking of scenarios
Record at which ε* rankings change between scenarios

Output:

- Ranking stability matrix
- Critical ε* values where rankings reverse
- Scenarios robustly preferred across all parameter values

Scenarios that rank highest across all or most parameter combinations are
robustly preferred. Parameter values at which rankings reverse are the
analytically important thresholds to report.

---

### Step 7: Report

For each comparison, report:

1. **Dashboard table:** All node scores for all scenarios
2. **Dominance summary:** Which scenarios are dominated
3. **Robust preference:** Which scenarios rank highest across parameter ranges
4. **Trade-off report:** Where and how rankings depend on moral parameters
5. **Inequality costs:** $A(\varepsilon)$ and $G$ per scenario - welfare lost
   to inequality
6. **Inadmissible scenarios:** Those failing hard floor or need constraints,
   and which criterion they violate
7. **Parameter record:** Full specification of all parameters used

---

## Part IV: Generic Assumptions, Caveats, and Considerations

### A. Assumptions Embedded in the Framework

**A1. Welfare commensurability**
All principles assume that welfare levels across different groups can be
compared on a common scale. This is a strong assumption. If groups are
fundamentally different in kind (e.g., different species, different
generations), comparability may not hold without additional argument.

**A2. Group homogeneity**
If the model assigns one $u_i$ value per group, it assumes each group is
internally homogeneous or that within-group inequality does not matter morally.
If within-group distribution matters, the unit of analysis should be the
individual rather than the group.

**A3. Independence of utility**
The standard functions treat $u_i$ as independent across individuals. If
welfare is relational (e.g., positional goods, social comparison effects),
the functions require modification.

**A4. Static framing**
The framework compares welfare states at a point in time or as averages over
a period. It does not natively capture:
- Path dependency (how a state is reached)
- Welfare trajectories over time
- Intertemporal inequality between generations
These require explicit extension (e.g., applying the framework separately
to each time period or discounting across periods).

**A5. Veil of ignorance interpretation**
The moral principles are grounded in reasoning under uncertainty about one's
own position. The tool inherits the assumption that this is the appropriate
moral standpoint. Communitarian or relational moral frameworks would produce
different principles.

---

### B. Methodological Caveats

**B1. The ε parameter lacks a consensus empirical value**
The Atkinson function requires a value for $\varepsilon$, but empirical studies
do not converge on a single estimate. Reported values range from 0.5 to 4 across
different studies and contexts. The recommended approach is to report results
across the full range rather than choosing a single value. Treat $\varepsilon$
as a moral sensitivity parameter, not a calibrated empirical constant.

**B2. Floor threshold selection is normative**
The choice of $f$ encodes a moral judgment about what constitutes an acceptable
minimum. Relative thresholds (e.g., 50% or 60% of median) are conventional in
poverty measurement but are not derived from first principles. Absolute thresholds
require empirical specification from outside the moral framework. This choice
should be reported transparently and tested in sensitivity analysis.

**B3. Dimension weights in multidimensional welfare**
If $u_i$ is constructed from multiple dimensions, the weights $w_k$ are
themselves a moral choice. Equal weights are the standard default but imply
that all dimensions matter equally. Alternative weighting schemes (e.g.,
derived from principal components or stated preference surveys) are defensible
but introduce additional assumptions.

**B4. The accountability node requires causal model structure**
Node 7 can only be implemented if the model encodes causal pathways that
allow $C_i$ and $E_i$ to be separated. If the model does not distinguish
between circumstance-driven and choice-driven outcomes, this node should not
be included or should be clearly flagged as approximate.

**B5. Conflict between principles is not a failure of the tool**
When different nodes rank scenarios differently, this is not a defect to be
resolved by further aggregation. It reflects genuine moral trade-offs. The
tool's function is to make these trade-offs visible and explicit, not to
resolve them. A weighting scheme that combines all nodes into a single score
hides these trade-offs and should be used with caution.

---

### C. Practical Considerations

**C1. Aggregation vs. dashboard**
Presenting a full dashboard of node scores is preferable to a single
aggregated score for most purposes. Single-score aggregation requires
assigning weights to the moral principles themselves, which is itself a
moral judgment. If a single score is required (e.g., for ranking in a
policy process), the weights must be explicitly stated and justified,
and sensitivity to those weights tested.

**C2. Negative and zero welfare values**
Several nodes (particularly W_A and W_Sen) are undefined or degenerate
when $u_i \leq 0$. If model outcomes can include zero or negative values
(e.g., net losses), the shift correction in Section 1.2 must be applied.
The choice of shift affects results and should be reported.

**C3. The framework does not resolve disputes about what counts as welfare**
The moral functions operate on welfare values $u_i$ provided by the model.
The framework takes no position on whether welfare should be measured as
preference satisfaction, objective functioning, hedonic experience, or
resource endowment. This choice is made upstream and is of equal or greater
importance than the choice of moral function.

**C4. Procedural fairness is not captured**
The framework addresses distributive fairness (how outcomes are distributed)
but not procedural fairness (whether the process producing outcomes was fair).
Experimental evidence finds that process fairness independently affects moral
judgments. If procedural fairness is relevant to the scenarios compared, it
requires separate treatment.

**C5. Empirical grounding of the principles**
The principles in this tool are grounded in experimental studies of moral
judgment, primarily conducted with populations in OECD countries. While
Frohlich & Oppenheimer found cross-national consistency across several
countries, the generalizability of these findings to all cultural contexts
is not fully established. This is a relevant caveat if the tool is applied
in contexts where the underlying experimental evidence is thin.

**C6. The tool informs but does not decide**
The framework provides a structured method for comparing scenarios against
multiple empirically supported moral criteria. It is designed to inform
deliberation by making trade-offs explicit, not to replace deliberation
or automatically determine policy choices. The output should be interpreted
by decision-makers with full awareness of the assumptions and parameter
choices embedded in each run.

---

## Appendix: Node Reference Summary

| Node | Function | Formula | Key parameter | Empirical basis |
|------|----------|---------|---------------|----------------|
| W_U | Utilitarian average | $\frac{1}{n}\sum u_i$ | None | Engelmann & Strobel (2004) |
| W_R | Maximin / Rawlsian | $\min_i(u_i)$ | None | Rawls (1971) |
| W_A | Atkinson SWF | $[\frac{1}{n}\sum u_i^{1-\varepsilon}]^{1/(1-\varepsilon)}$ | $\varepsilon \in [0,\infty)$ | Atkinson (1970) |
| W_FC | Floor-constrained average | $\bar{u}$ s.t. $\min(u) \geq f$ | $f$, $\lambda$ | Frohlich & Oppenheimer (1992) |
| W_Sen | Sen-Gini SWF | $\bar{u}(1-G)$ | None | Sen (1974) |
| W_N | Need satisfaction | $\frac{1}{n}\sum_i \frac{1}{K}\sum_k \min(1, u_{ik}/n_k)$ | $n_k$ per dimension | Konow (2003), Alkire & Foster (2011) |
| W_Acc | Accountability-adjusted | $W_A(\varepsilon)$ on $\hat{u}_i(C)$ | $\varepsilon$, $C_i$ definition | Roemer (1998), Konow (2000) |
| W_Eq | Equality reference | $-\frac{1}{n}\sum(u_i - \bar{u})^2$ | None | Engelmann & Strobel (2004) |
