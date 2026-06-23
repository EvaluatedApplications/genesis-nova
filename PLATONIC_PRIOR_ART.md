# A Dual-Face Geometric Substrate for Exact Computation and Distributional Meaning, Driven by a Thin Neural Controller

**Defensive publication / prior-art disclosure.** First public description: 2026-06-23. Project: Genesis-Nova.
This document discloses the system below to establish priority of invention. It is written to be *enabling* — a
person skilled in the art can reproduce the mechanisms from the descriptions and formulas given.

---

## Abstract

We describe a knowledge representation and reasoning substrate in which every concept is a single fixed-width
vector partitioned into functional **faces**: a **frozen identity nucleus** (exact, structured) and a **free
relational cloud** (distributed, learnable). Numbers are encoded so that arithmetic is computed *exactly* by vector
operations and generalizes to unseen operands with no stored facts. Words and sentences are composed from reused
sub-elements (a bounded atom set), so the active representation grows by *reuse* rather than duplication. The large
"relational" face stores a concept's meaning as a **superposition of its relational context** (a distributional
encoding), which makes related concepts cluster, makes unrelated concepts orthogonal without any repulsion tuning,
and represents lexical **ambiguity** natively as a multi-sense superposition. A small recurrent neural network acts
as a **thin controller** that *selects and composes* substrate operations rather than storing knowledge in its
weights. The substrate is **conserved** (every element has an exact complement) and **monotone** (distinctions are
archived, never destroyed). We additionally disclose a **persistent neural self-state** that conditions the
controller and is maintained by a regenerative homeostatic loop.

---

## 1. Field and problem

Conventional neural models store knowledge implicitly in weights, cannot compute exactly, generalize arithmetic
poorly, and grow without bound. Symbolic systems compute exactly but do not represent graded or ambiguous meaning
well. This disclosure addresses both: a single geometric substrate that computes exactly *and* represents
distributed, ambiguous meaning, with a small neural controller that learns only to *use* it.

## 2. Element representation: the dual face

An element is a vector **e ∈ ℝ^D** (e.g. D = 512), partitioned into contiguous **faces**. Each face is either
**frozen** (defines identity; never altered by learning) or **free** (carries learnable meaning):

- **Numeric face** `[0, 2N)`, N = min(⌊D/2⌋, 21): a number's value (frozen for a number).
- **Character face** `[42, 202)`: composed from a token's characters (the frozen identity of a *word*).
- **Relational ("word") face** `[202, D)`: the large free face that carries distributed meaning (a phrase/sentence
  of words; see §5).

**Key property (restore-after-update):** any learning update may write only to free dimensions; after every update
the frozen dimensions are restored to their exact values and the complement (§6) is re-enforced. Therefore
relational learning *cannot* corrupt identity: an element pushed anywhere in its free cloud still decodes to exactly
itself.

## 3. Exact computation by co-located homomorphic faces

For a numeric value v, the numeric face is two co-located sub-faces:

- **Polynomial** `poly[i] = v · 10^-(i+1)` for i ∈ [0, N). This is **additively homomorphic**:
  `poly(a) + poly(b) = poly(a + b)`.
- **Logarithmic** `log[i] = ln|v| · 10^-(i+1)` for i ∈ [N, 2N). This is **multiplicatively homomorphic**:
  `log(a) + log(b) = log(a · b)`.

Consequently addition/subtraction and multiplication/division are computed by *adding the corresponding faces* and
decoding, **exactly**, for operands never seen in training, with no per-fact storage. The operation is selected
from context by the controller (§7), not by a symbol parser. (The logarithmic face has no representation of zero;
sign is carried in the polynomial face.) Decoding is the exact inverse of the encoding.

## 4. Compositional ladder with reused atoms (bounded growth)

Composites are **hubs over reused component elements**, forming a ladder:

- **digits → a NUMBER** (numeric face),
- **characters → a WORD** (character face),
- **words → a SENTENCE/phrase** (relational face).

The atom set (≈100 characters; digit place values) is **bounded**. A composite stores **part-of edges** to its
components, which are *shared*: one character element is referenced by every word; one word element by every
sentence. Therefore N observed composites cost **O(N + |atoms|)** storage and a *novel* composite over existing
components costs **O(1)** — growth is recombination, not new storage. The number homomorphism (§3) is the special
case of this ladder for which composition is an exact algebraic operation.

## 5. Distributional relational meaning in the large face (and ambiguity)

Each symbol x has a deterministic, symbol-seeded **token** `t(x)` — a fixed unit vector over the relational face.
A concept's relational-face value (its "meaning cloud") is the normalized **superposition of its own token and the
tokens of its relational context**:

> **μ(x) = normalize( t(x) + Σ_{y ∈ N(x)} (1 − 2·κ(x,y)) · t(y) )**

where N(x) is the set of x's related concepts and κ(x,y) ∈ [0,1] is the observed **contradiction** between x and y
(0 = agreement, 1 = opposition). Each neighbour contributes **once** (presence-weighted, independent of how often
the relation was observed). Properties, which follow directly:

- **Emergent clustering.** Concepts that share relational context share tokens, so their clouds overlap (high cosine
  similarity); concepts with disjoint context are ≈orthogonal. Separation requires **no repulsion/contrastive
  tuning** — orthogonality of the unrelated is automatic.
- **Differential weighting.** Agreement (low κ) *adds* a neighbour's token; contradiction (high κ) *subtracts* it
  (meaning is defined by what a thing is *and is not*).
- **Native ambiguity.** A polysemous symbol whose context spans two sense-clusters has a cloud that is a
  *superposition* sitting near *both* sense regions simultaneously, while the two senses remain mutually
  orthogonal. A single point cannot represent this; the superposition does.

Because meaning lives in the *direction* of the relational face (the cloud is normalized), magnitude carries no
meaning and accumulation is numerically safe. Identity (numeric/character faces) is untouched by any of this.

## 6. Conservation and monotonicity

- **Conservation (complement).** Every element e has an exact complement `¬e = −e`, re-enforced after every update,
  so the substrate's total signed mass is zero. The complement anchors differential meaning.
- **Monotonicity (archival).** Elements are never destroyed; an evicted element is **archived** (made dormant) with
  its learned state retained and is **reactivated** intact if re-observed. The active set stays bounded for speed
  while no distinction is ever lost.

## 7. The thin neural controller

A small recurrent network (a GRU) is a **selector/router**, not a knowledge store. From an input it produces a
shared hidden representation that drives decision heads (which retrieval/compute path to take; which operation;
which composition shape). Each path *abstains* if it cannot answer, so control falls through a ladder of substrate
operations (exact arithmetic via §3; relational retrieval via §5; composition via §4). **Capability emerges from
the controller composing substrate operations** — the network learns only *which* operation, never a stored answer
table or a hardcoded parser. The network has zero parameters of the substrate's width; the two are bridged by
concept↔token correspondence, so substrate width and controller capacity are chosen independently.

## 8. Persistent self-state and regenerative homeostasis (disclosed component)

We further disclose an optional mode in which the controller carries a **persistent self-state** — a recurrent
hidden vector that is *not* reset between inputs, integrating each observation through the same learned recurrence.
This self-state (i) conditions the controller's encoding of every input (so cognition proceeds from accumulated
state), (ii) is written into the substrate as a self-element (the controller is represented within its own
substrate), and (iii) is maintained by a **homeostatic loop**: the substrate is a committed identity pattern, an
external process perturbs it (removing elements), and the controller restores the pattern from the conserved
(archived) memory toward the committed setpoint. A measurable "cognitive reach" (the extent of pattern kept
coherent under perturbation) characterizes the self.

## 9. Enumerated claims of novelty

The combination, and each of the following, is disclosed as inventive:

1. A concept represented as a single vector partitioned into a **frozen structured identity nucleus** and a **free
   relational cloud**, with restoration of frozen dimensions after every learning update so that relational
   learning cannot alter identity.
2. **Co-located polynomial and logarithmic homomorphic faces** that compute exact addition/subtraction *and*
   multiplication/division by vector addition, generalizing to unseen operands with no stored facts, with the
   operation selected from context by a controller rather than parsed.
3. A **compositional ladder of reused atoms** (digits→number, characters→word, words→sentence) giving O(N+|atoms|)
   storage and O(1) novel-composite cost.
4. A **distributional relational encoding of the large face** as a presence-weighted, contradiction-signed
   superposition of context tokens, yielding emergent clustering, automatic orthogonality of unrelated concepts
   *without contrastive tuning*, and native representation of ambiguity as a multi-sense superposition.
5. A **conserved** (exact complement) and **monotone/archival** geometric substrate.
6. A **thin recurrent controller** that selects and composes substrate operations (rather than storing knowledge in
   weights), with substrate and controller widths chosen independently via a name-based bridge.
7. A **persistent, substrate-resident self-state** maintained by regenerative homeostasis against perturbation.

## 10. Reproducibility

All formulas above (face boundaries, the `10^-(i+1)` numeric encodings, the token superposition μ(x), the complement
and archival rules) are sufficient to implement the substrate. A reference implementation exists in the Genesis-Nova
codebase (faces in `Core/FaceLayout.cs` and `Core/PlatonicFaceComposer.cs`; the substrate in
`Cognition/Platonic/`; the controller in `Model/`). Companion design documents: `PLATONIC_THEORY.md` (formal model),
`PLATONIC_NUCLEUS.md` (the dual-face data model), `PLATONIC_SPACE.md` (substrate capabilities),
`PLATONIC_CONSCIOUSNESS.md` (the self-state component).

---

*This disclosure is published defensively to establish prior art as of the date above. No claim is made herein as
to performance benchmarks; the contribution is the architecture and mechanisms enumerated in §9.*
