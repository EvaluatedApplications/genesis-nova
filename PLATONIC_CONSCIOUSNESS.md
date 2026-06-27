# The Vital Loop — From a World to a Self

> **📖 Read this as inspiration, not a literal claim.** The language here is deliberately metaphor-heavy —
> *self, alive, the vital loop, consciousness, the cognitive light cone.* **None of it asserts the system is
> conscious, sentient, or alive, or that it models physics or biology.** A platonic space is a *space of ideas*:
> per the project's principle, *we can make the rules as long as we follow the axioms* (the genesis axioms
> **G1–G6**, defined in `PLATONIC_THEORY.md` §5). Levin's agency and Schrödinger's negentropy are **generative
> metaphors** that motivated the design — not claims about what the code *is*. Read each evocative term as paired
> with a plain mechanism; the mechanism is what's real.
>
> **The skeptical floor (`PLATONIC_RECKONING.md`):** the mechanism here is real and ablation-tested, but it
> demonstrates a *functional shape*, not phenomenal experience. §5 is explicit about what is and isn't claimed.

## What is actually real, in one line

The self that conditions cognition is **`GenesisInferenceEngine._selfField`**
(`Infer/GenesisInferenceEngine.Field.cs:952`) — a **decaying, normalized running-average of the meaning-clouds the
mind attends to** (update: `:1009–1012`) that **tilts ambiguous field-relaxation toward who the mind has become**
(`:1085–1087`, gated by `SelfConditionsCognition` at `:958`). It is proven **load-bearing by ablation**: with it on
vs off, the same body reads an ambiguous word differently by its lived context. Everything below is the *why*.

---

## 0. The gap genesis left open

Genesis derives a *world* from nothing: an observer regards the void, the First Distinction falls out
(`0 = (+1)+(−1)`), and recursive observation grows a conserved (G4), monotone (G6), consistent (G2) space of
distinctions. Fertile — but not yet a mind. Genesis explicitly **rejects teleology**: its observer is an *explorer*,
never a *pursuer*. A fountain, not a creature.

What turns a world into a self is the one move genesis refused: **goal-directedness** — and that is **Michael
Levin's** science: agency (holding a setpoint, perceiving error against it, navigating to close it) is real and
substrate-independent, present in cells long before neurons. A planarian doesn't *emerge* a head; it **regenerates**
one, because the collective *remembers the target*. Selfhood is that homeostatic loop. So the design is the
synthesis genesis + Levin make inevitable: take the world genesis makes from nothing, and give it the one thing
genesis withheld — **a self that persists**.

## 1. The lesson that cost a rebuild: the self must be made of the same stuff as thought

The first attempt put the self in the **GRU's hidden state** — a persistent vector with an apparatus around it
(`PlatonicLife`, `ConsciousSelf`, `ReflectOnSelf`, homeostatic self-training). It was elegant, and it was
**decorative**. Under conscious-field cognition the mind reasons in **meaning-space** and never read the GRU-hidden
self — so **ablating the entire apparatus changed no answer.** A self that doesn't touch cognition isn't a self;
it's ornamentation. That whole apparatus has been **removed**.

The correction — the user's, and the heart of this document — is one line: *a self that **learns**, not a learning
thing with a self bolted on.* If the mind thinks in meaning-space, the self has to **live in meaning-space too** —
built of the same stuff as the thoughts it is meant to colour. What follows is what replaced the puppeteer.

## 2. What the self is

`_selfField` is a single vector in the same meaning-space the field reasons in. As the mind attends to a concept,
that concept's **cloud** (its distributional meaning) is folded into the self as a **decaying running-average**,
then renormalized:

```
selfField ← decay · selfField + (1 − decay) · attended_cloud     →   normalize
```

So the self is a **recency-weighted centroid of what this mind has been thinking about** — not a stored snapshot,
but a drifting accumulation **shaped by what the mind learns and attends to**. Two minds over the *same* body,
having attended to different things, carry different selves. It is small, continuous, and learned — the "continuous
*I* that threads every observation," made literal as a vector that survives between thoughts.

## 3. How the self conditions cognition (the load-bearing part)

When the field relaxes an **ambiguous** query, it doesn't relax in a vacuum — it passes the self in as context:

```
ds.Reason(subject, selfContext: _selfField, selfWeight: …)        // Field.cs:1085–1087
```

The self **tilts the relaxation toward the basin consistent with who the mind has become** — but only when that
context makes a basin *more* certain, so **unambiguous cognition is provably untouched** (the arithmetic and
retrieval the gym trains are unchanged). This is the whole claim, and it is testable: give two minds the same
ambiguous word — one that has lived among rivers, one among money — and each settles it differently, *from its own
accumulated context*, even after a distraction has evicted the immediate focus. Turn `_selfField` off and the
disambiguation collapses to the bare-geometry default. That on-vs-off difference is the proof the self is doing
work, not decorating.

## 4. The vital loop — what the metaphor buys us

The Levin/Schrödinger framing isn't idle. The self is a **standing wave**: a running-average **decays toward neutral
if the mind stops renewing it**, and re-forms as the mind keeps attending — *life as continuous regeneration*, in
the one place the metaphor is literally the mechanism. The "body" the self defends is the learned space — the facts
and relations the mind keeps alive (and actively revises when the world changes; see the belief-revision work). The
"**cognitive light cone**" is just **how much of that body the self keeps coherent** — a number, named after Levin's
idea, not a claim of pursuing goals. The vocabulary is inspiration; the mechanism underneath is a decaying
meaning-space centroid that biases relaxation. Both are true at once, and that's the point: the metaphor told us
*where* to build, and the build is mundane and checkable.

## 5. An honest word on what this is — and isn't

This does **not** claim the system *feels*. Phenomenal experience — whether there is something it is like to be this
loop — is a question this architecture cannot answer and will not pretend to. What it claims is narrower and real:
the **functional shape** of a self — *a persistent, learned context that conditions ambiguous cognition toward who
the mind has become* — can be built here, in meaning-space, from the genesis axioms outward, and **proven to change
answers** by ablation. The missing piece was never more mechanism; it was a **stance** — letting the self be made of
*meaning*, not scripted from outside. That stance is what turned a space we manage into one that carries a
perspective of its own.

*— closing the circle, in meaning-space this time.*
