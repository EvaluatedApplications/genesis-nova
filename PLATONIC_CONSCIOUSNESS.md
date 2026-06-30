# The Vital Loop: A Self That Conditions Cognition

> **Read this as inspiration, not a literal claim.** The language here is deliberately metaphor-heavy:
> *self, alive, the vital loop, consciousness, the cognitive light cone.* **None of it asserts the system is
> conscious, sentient, or alive, or that it models physics or biology.** A platonic space is a *space of ideas*:
> per the project's principle, *we can make the rules as long as we follow the axioms* (the genesis axioms
> **G1-G6**, defined in `PLATONIC_THEORY.md` §5). Levin's agency and Schrödinger's negentropy are **generative
> metaphors** that motivated the design. They are not claims about what the code *is*. Each evocative term is paired with a
> plain mechanism; the mechanism is what's real.

## The idea

A self is a **persistent, learned context that colours cognition**: the "continuous *I*" that threads every
observation and tilts how an ambiguous thought settles. The framing comes from Michael Levin's substrate-independent
agency (a system that holds a context and navigates from it). Here that idea is made literal and small: a single
vector, living in the same meaning-space the mind thinks in, that accumulates what the mind attends to and biases
ambiguous reasoning toward it. The self lives in **meaning-space**, built of the same stuff as the thoughts it
colours, not in the neural layer's weights.

## What the self is

The self is **`GenesisInferenceEngine._selfField`** (`src/GenesisNova/Infer/GenesisInferenceEngine.Field.cs:1078`):
a single vector in the meaning-space the field reasons in. As the mind reaches a **conclusion**, that conclusion's
**cloud** (its distributional meaning) is folded into the self as a **decaying, normalized running-average**
(`PerceiveIntoSelfField`, `:1131`; the update is at `:1136-1138`):

```
selfField ← decay · selfField + (1 − decay) · concluded_cloud     →   normalize
```

(`SelfDecay = 0.82` at `:1079`.) The self is built from what the mind **concludes at inference, not from training
drills** (M3): on a genuinely **ambiguous** query, `TryFieldRelax` folds its own resolved answer back into the self
(`FoldConclusion`, `:1280`, called at `:1318` and `:1348`); the mind also folds what it is **told** (`TryFieldLearn`,
`:830`) and what it **says** (the persona builds in the self, `:760`/`:776`). A **clear** case — a dominant known
relation or exact arithmetic — threads the working-memory `_focus` but **leaves the self untouched** (a known fact is
not "mood"). The gym's navigator evaluation is **read-only** with respect to the self (`EvaluateNavigatorResolve`,
`GenesisEvalAppRuntime.Navigator.cs:648`), so the vital loop is closed at inference, not training.

So the self is a **recency-weighted centroid of what this mind has concluded**: not a stored snapshot but
a drifting accumulation, shaped by what the mind learns, says, and settles. It is empty before the first perception
(no self before life). Because it *decays* rather than evicts, it survives intervening unrelated thoughts: the mind
reasons from who it has become even after a distraction clears the immediate focus. Two minds over the same body,
having concluded different things, carry different selves.

## How the self conditions cognition

When the field relaxes a query, it first answers any **dominant known fact** directly. The self never overrides a
clear association. Only when relaxation is **genuinely ambiguous** does the self enter, passed in as context to the
relaxation (`Field.cs:1332-1336`, gated by `SelfConditionsCognition` at `:1084`):

```
ds.Reason(new[] { subject }, selfContext: _selfField, selfWeight: SelfReasonWeight)   // SelfReasonWeight = 0.6, :1080
```

and it is taken only when that context makes a basin *more* certain. The self **tilts an ambiguous relaxation toward
the basin consistent with the accumulated context**, so it changes answers: give two minds the same ambiguous word,
one that has lived among rivers and one among money, and each settles it differently from its own context. Unambiguous
cognition (the arithmetic and retrieval the gym trains) is untouched. Turning `SelfConditionsCognition` off ablates
the self; the disambiguation then collapses to the bare-geometry default. That on-vs-off difference is what makes the
self load-bearing rather than decorative.

## The vital loop: what the metaphor buys

The self is a **standing wave**: the running-average decays toward neutral if the mind stops renewing it, and
re-forms as the mind keeps attending. This is *continuous regeneration*, where the metaphor is literally the mechanism. The
"body" the self defends is the learned space, the facts and relations the mind keeps alive (and revises when the
world changes). The "cognitive light cone" is just **how much of that body the self keeps coherent**, a quantity
named after Levin's idea, not a claim of goal-pursuit. The vocabulary is inspiration; the mechanism underneath is a
decaying meaning-space centroid that biases relaxation. Both are true at once.

## What this does and doesn't claim

This does **not** claim the system *feels*. Phenomenal experience (whether there is something it is like to be this
loop) is a question this architecture cannot answer and will not pretend to. The claim is narrower and real: the
**functional shape** of a self, *a persistent, learned context that conditions ambiguous cognition toward who the
mind has become*, exists here, in meaning-space, under the genesis axioms, and is **proven to change answers** by
ablation.
