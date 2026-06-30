# A Dual-Face Geometric Substrate for Exact Computation, Distributional Meaning, and Compositional Memory, Driven by a Thin Neural Controller

**Defensive publication / prior-art disclosure.** First public description: 2026-06-27. Project: Genesis-Nova.
This document discloses the system below to establish priority of invention. It is written to be *enabling*: a
person skilled in the art can reproduce the mechanisms from the descriptions and formulas given.

> **Framing note (inspiration vs. mechanism).** Genesis-Nova is organised around a *platonic space*, a space of
> ideas whose rules we are free to choose so long as they obey a small set of axioms (G1..G6: born-neutral elements,
> differential/dialectical meaning, conservation by exact complement, monotone/archival irreversibility, etc.). Some
> of the project's vocabulary ("conscious field", "self", "homeostasis", "cognitive reach") is *generative
> metaphor* drawn from physics/biology/consciousness, used to motivate the design. It is **not** a claim that the
> system models physics or is conscious. This disclosure describes only the concrete mechanisms and the axioms they
> satisfy; the inspirational language is flagged where it appears.

---

## Abstract

We describe a knowledge representation and reasoning substrate in which every concept is a single fixed-width
vector partitioned into functional **faces**: a **frozen identity nucleus** (exact, structured) and a **free
relational cloud** (distributed, learnable). Numbers are encoded so that arithmetic is computed *exactly* by vector
operations and generalizes to unseen operands with no stored facts. Words and sentences are composed from reused
sub-elements (a bounded atom set), so the active representation grows by *reuse* rather than duplication. The large
"relational" face stores a concept's meaning as a **superposition of its relational context** (a distributional
encoding), which makes related concepts cluster, makes unrelated concepts orthogonal without any repulsion tuning,
and represents lexical **ambiguity** natively as a multi-sense superposition. Relations are themselves **positioned,
recursively-composable elements**, so a told fact is stored as a structured (non-flattening) composition with belief
revision, and distinct compound keys resolve distinctly. A token's **structural role** (function vs. content) is
recognized **without labels** from its neighbourhood graph topology, so grammatical structure is learned from
exposure. A small recurrent neural network acts as a **thin controller** that *selects and composes* substrate
operations rather than storing knowledge in its weights. The substrate is **conserved** (an exact
complement map `¬e = −e`; total signed charge zero by construction) and **monotone** (no distinction is ever unmade:
live eviction de-materialises an element to stay bounded, but an evicted symbol re-derives its exact identity from
its deterministic codec coordinate when re-observed). We additionally disclose a **persistent
self-state** (realized in the substrate's own meaning-space, not as a hidden weight vector) that conditions
reasoning and is persisted as conserved memory (restored verbatim from a checkpoint).

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

The `[42, 202)` / `[202, D)` boundaries above are the small-dimension layout. In the production substrate
(`D ≥ 512`; the default `D = 1024`) the frozen identity nucleus is partitioned more finely into fixed,
codec-derived, decodable bands — a kind code, a per-character *spelling* band, an ordered *structure* band (child
digests + role/label codes, which records a composite's components inside the frozen address), and an op code —
with the single learned *orbital* tail `[416, D)` carrying the relational cloud. The numeric faces (`[0, 21)` poly,
`[21, 42)` log) are byte-identical in both layouts. The dual-face principle (frozen, exact identity nucleus +
free, learned relational cloud) is unchanged; only the internal subdivision of the nucleus differs.

**Key property (restore-after-update):** any learning update may write only to free dimensions; the frozen
dimensions always carry their exact values. Therefore relational learning *cannot* corrupt identity: an element
pushed anywhere in its free cloud still decodes to exactly itself. In the **current default substrate** the identity
faces are not stored at all — they are re-assembled fresh from a deterministic codec on every read (the element
stores only the learned orbital tail), so there is nothing to drift and nothing to "restore": the property holds by
construction. The explicit *restore-the-frozen-dims-and-re-enforce-the-complement-after-each-update* step (and the
stored complement of §8) belonged to the **legacy stored-face memory** (`PlatonicSpaceMemory`), not to the default
codec core.

## 3. Exact computation by co-located homomorphic faces

For a numeric value v, the numeric face is two co-located sub-faces:

- **Polynomial** `poly[i] = v · 10^-(i+1)` for i ∈ [0, N). This is **additively homomorphic**:
  `poly(a) + poly(b) = poly(a + b)`.
- **Logarithmic** `log[N+k] = ln|v| · 10^-(k+1)` for k ∈ [0, N) — the log block sits at offset N but **reuses the
  same `10^-(k+1)` decay as the polynomial block** (the exponent restarts at the block boundary; it is *not* a
  continued `10^-(i+1)` across the whole `[0, 2N)` range). This is **multiplicatively homomorphic**:
  `log(a) + log(b) = log(a · b)`. (Decoding reads the block head — `value ≈ poly[0]·10` for the polynomial face,
  `value ≈ exp(log[N]·10)` for the logarithmic — confirming the per-block restart.)

Consequently addition/subtraction and multiplication/division are computed by *adding the corresponding faces* and
decoding, **exactly**, for operands never seen in training, with no per-fact storage. The operation is selected
from context by the controller (§9), not by a symbol parser. (The logarithmic face has no representation of zero;
sign is carried in the polynomial face.) Decoding is the exact inverse of the encoding.

## 4. Compositional ladder with reused atoms (bounded growth)

Composites are **hubs over reused component elements**, forming a ladder:

- **digits → a NUMBER** (numeric face) — realized by the §3 homomorphism, *not* by stored digit part-of edges (a
  number element has no components; its place values are implicit in the encoding),
- **characters → a WORD** (character face),
- **words → a SENTENCE/phrase** (relational face).

The atom set (≈100 characters) is **bounded**. A text composite stores **part-of edges** to its components, which
are *shared*: in the default core a single word's element references one character element per character (kind
`Composition`), and a multi-word phrase's element references one word element per word; so one character element is
referenced by every word, one word element by every sentence. (A configurable *generative-atoms* mode instead
treats the whole token as its own atom and lazily discovers recurring sub-atoms/morphemes on demand; the default is
the eager character composition described here.) Therefore N observed composites cost **O(N + |atoms|)** storage and a *novel* composite over existing
components costs **O(1)**. Growth is recombination, not new storage. The number homomorphism (§3) is the special
case of this ladder for which composition is an exact algebraic operation.

## 5. Distributional relational meaning in the large face (and ambiguity)

Each symbol x has a deterministic, symbol-seeded **token** `t(x)`, a fixed unit vector over the relational face.
A concept's relational-face value (its "meaning cloud") is the normalized **superposition of its own token and the
tokens of its relational context**:

> **μ(x) = normalize( t(x) + Σ_{y ∈ N(x)} (1 − 2·κ(x,y)) · t(y) )**

where N(x) is the set of x's related concepts and κ(x,y) ∈ [0,1] is the observed **contradiction** between x and y
(0 = agreement, 1 = opposition). Each neighbour contributes **once** (presence-weighted, independent of how often
the relation was observed). Properties, which follow directly:

- **Emergent clustering.** Concepts that share relational context share tokens, so their clouds overlap (high cosine
  similarity); concepts with disjoint context are ≈orthogonal. Separation requires **no repulsion/contrastive
  tuning**: orthogonality of the unrelated is automatic.
- **Differential weighting.** Agreement (low κ) *adds* a neighbour's token; contradiction (high κ) *subtracts* it
  (meaning is defined by what a thing is *and is not*).
- **Native ambiguity.** A polysemous symbol whose context spans two sense-clusters has a cloud that is a
  *superposition* sitting near *both* sense regions simultaneously, while the two senses remain mutually
  orthogonal. A single point cannot represent this; the superposition does.

Because meaning lives in the *direction* of the relational face (the cloud is normalized), magnitude carries no
meaning and accumulation is numerically safe. Identity (numeric/character faces) is untouched by any of this.

## 6. Compositional fact memory: relation as a recursive, labeled element

A relation between two elements is itself a **positioned element**: a binary, labeled composition `⟨a · label · b⟩`
placed at the blend of its endpoints' meaning-clouds (§5), recording its two components and a relation label.
Because the output of the composition is itself an element, composition is **recursive**: a composition may be an
endpoint of another, so compound structures (e.g. the key `my favorite color`) are built bottom-up as nested
compositions, **not** flattened into a single averaged point.

Storing a told fact is creating such a composition with a relation label (e.g. `is`): a key (possibly itself a
nested composition) is linked to a value. Recall traverses the composition back from the key to its value.
Consequently two distinct compound keys (e.g. `my name` vs. `your name`) remain **distinct structured objects** and
resolve to **different** values, a discrimination a single averaged representation cannot make. A newly asserted
fact **revises** prior ones: asserting a new value for an existing key weakens the prior relation through the
contradiction mechanism (§5), so the latest assertion governs recall without erasing the substrate's archived
history (§8). This realizes associative, editable, **inspectable** memory as structure *in the space* rather than as
adjustments to controller weights. The operation is the substrate-native form of binary recursive labeled
composition, the minimal operation from which hierarchical structure is built.

## 7. Learned structural roles from graph topology (label-free)

Whether a token plays a **structural / function role** (determiners, copulas, prepositions, conjunctions: the
closed-class "glue" of a language) or a **content role** (entities, attributes) is recognized **without labels**,
from the token's position in the relation graph. A function word **bridges otherwise-unrelated concepts**: its
neighbours do not connect to one another, so the **clustering coefficient** of its neighbourhood is low. A content
word sits inside a cluster of mutually-related kin, so its neighbourhood clustering is high. The signature is the
**local clustering coefficient** of the token's neighbourhood (the fraction of neighbour *pairs* that are themselves
connected). The decision threshold is the **valley of the population's clustering distribution** (an Otsu
between-class split), so it tracks the bimodal glue/content separation as the graph grows rather than being a fixed
constant; the signal **self-abstains** (classifies nothing) on a cold or too-small graph. Once a token has read as a
confident low-clustering outlier on enough independent recomputations its function classification is **conserved**
(monotone — see §8): later graph growth, which only adds edges, can never unmake it. All of this is an emergent
property of how broadly a token co-occurs, with **no** hand-written stop-list, part-of-speech tagger, or supervised
labels.

Because the signal is purely distributional, the mechanism that learns roles from a small curriculum is, in
principle, the same mechanism by which the system would acquire them from a natural corpus: grammatical structure is
**learned from exposure**, not encoded. This role signal gates the compositional parse (§6): the structural tokens
mark *how* content tokens compose.

## 8. Conservation and monotonicity

- **Conservation (complement).** The substrate defines an exact complement `¬e = −e` as a deterministic negation of
  an assembled face, so the total signed charge is zero **by construction** (the default core reports
  `TotalCharge ≡ 0` as an identity, not by summing live elements). In the default codec core this is an *axiom
  satisfied by construction* rather than an active runtime loop: the negation map exists but is not invoked by the
  live pipeline, and because identity is codec-derived and never stored there is nothing to "re-enforce" after an
  update. (A stored, periodically re-imposed complement was a mechanism of the legacy stored-face memory.) The
  *differential* structure that actually shapes meaning at runtime is the **contradiction-signed superposition** of
  §5: a contradicted neighbour's token is *subtracted* from a concept's cloud (the `(1 − 2·κ)` weight), so a thing is
  defined by what it is *and is not*.
- **Monotonicity (G6) by latent coordinate.** No distinction is ever unmade, but *not* because elements are never
  deleted: live eviction genuinely **de-materialises** an element to keep the active set bounded under churn. G6
  holds because identity is a deterministic, decodable **address** — a pure function of the symbol — so an evicted
  symbol that is re-observed **re-derives its exact frozen identity**; the distinction is conserved by the coordinate
  system, not by the live entry. The non-derivable learned tail (the orbital cloud) **re-accumulates** on
  re-observation rather than being preserved verbatim through eviction. A separate **dormant/archival** path that
  retains an element's learned state intact and reactivates it exists, but its live use is restoring dormancy on
  checkpoint import, not eviction. The active set stays bounded for speed while no *identity* is ever lost.

## 9. The thin neural controller

A small recurrent network (a GRU) is a **selector/router**, not a knowledge store. From an input it produces a
shared hidden representation that can drive decision heads (which retrieval/compute path to take; which operation;
which composition shape). Each path *abstains* if it cannot answer, so control falls through a ladder of substrate
operations (exact arithmetic via §3; relational retrieval via §5; compositional recall via §6). **Capability emerges
from composing substrate operations**: the controller learns only *which* operation, never a stored answer table or
a hardcoded parser. The network has zero parameters of the substrate's width; the two are bridged by concept↔token
correspondence, so substrate width and controller capacity are chosen independently.

In the primary control path the substrate's own settling drives selection directly: a capability is chosen by the
*structure* of the prompt (which operands, operators, and cues are present) and each reduction runs on the
substrate's general primitives, with the network's classification heads bypassed; a query that nothing settles
**abstains** rather than emitting an invented answer. (Classifier heads remain as a legacy, off-by-default selection
path; the reduction logic they select is itself classifier-free.) The disclosed novelty is the ladder of abstaining
substrate operations and the thin-controller/substrate split, independent of which selection path drives it.

### 9.1 Learned navigation: reasoning as a situated walk

Beyond single-shot path selection, we disclose a **learned navigator**: a thin policy network that reasons by
**walking the substrate's relation graph step by step** to reach answers a one-shot retrieval cannot. It is consulted
only on the **ambiguous branch** — a query with no dominant relation, where the ladder's single hop lands short — and
otherwise falls through, cold-safe, to the existing path, so it never degrades the cases the ballpark structure
already nails. The navigator is **query-conditioned and goal-emergent**: from a query it takes an anchor concept and a
*learned* abstraction-level cue (genus / domain / root, derived from the answer's graph depth, with no word list) and,
for compositional queries, a learned goal-region; it is given no goal coordinate. At each step it scores the current
concept's relational neighbours plus a HALT action on answer-free differential features (candidate − goal, candidate −
current, candidate, contradiction), conditioned by the persistent self-state (§10), and either steps or halts. It is
trained by **imitation (DAgger) of a flow-field oracle** — a backward-Dijkstra cost/next field over the relation graph
yielding globally optimal next-steps to a goal — so the walk learns to route around locally-near distractors toward a
multi-hop answer, the lookahead a greedy heuristic lacks. The result is multi-hop reasoning (cross-relation
composition; climbing to a queried abstraction level) reached by a *learned walk* where the ladder's one-hop retrieval
structurally cannot — the last-mile mechanism that lifts the ambiguous cases the heuristic ceiling leaves wrong.

## 10. Persistent self-state (disclosed component)

We disclose a **persistent self-state** that is held in the substrate's own meaning-space (the large relational
face) rather than as a hidden weight vector. It is a decaying accumulation of the meaning-clouds (§5) of the
concepts cognition has recently attended to; it is *not* reset between inputs. This self-state (i) conditions
reasoning by biasing the substrate's relaxation/retrieval toward the accumulated context (so an ambiguous query is
resolved in the direction of what the system has been "thinking about"), and (ii) is itself shaped by learning:
the same clouds the system sharpens by observation are what the self is built from. Because identity lives in the
frozen faces and is conserved (§8) by the latent decodable coordinate, an evicted concept's identity is re-derived
exactly when re-observed, and persisted learned state (including the self-state itself, restored verbatim from a
checkpoint) is recovered from conserved memory.

> *Inspirational framing (not a literal claim).* The language of a "homeostatic loop", a committed identity
> "setpoint" restored under "perturbation", and a measurable "cognitive reach" is generative metaphor for the
> conserved/archival mechanism above (G4 conservation, G6 irreversibility); it is not a claim that the system
> implements biological homeostasis or possesses consciousness. The self-state is the meaning-space accumulation
> described here, not a recurrent hidden weight vector.

## 11. Enumerated claims of novelty

The combination, and each of the following, is disclosed as inventive:

1. A concept represented as a single vector partitioned into a **frozen structured identity nucleus** and a **free
   relational cloud**, with restoration of frozen dimensions after every learning update (or codec-derived identity
   that is never stored) so that relational learning cannot alter identity.
2. **Co-located polynomial and logarithmic homomorphic faces** that compute exact addition/subtraction *and*
   multiplication/division by vector addition, generalizing to unseen operands with no stored facts, with the
   operation selected from context by a controller rather than parsed.
3. A **compositional ladder of reused atoms** (digits→number, characters→word, words→sentence) giving O(N+|atoms|)
   storage and O(1) novel-composite cost.
4. A **distributional relational encoding of the large face** as a presence-weighted, contradiction-signed
   superposition of context tokens, yielding emergent clustering, automatic orthogonality of unrelated concepts
   *without contrastive tuning*, and native representation of ambiguity as a multi-sense superposition.
5. A **compositional, recursive, labeled fact memory** in which a relation is a positioned element (`⟨a·label·b⟩`)
   placed at the blend of its endpoints and recursively composable, giving structured **non-flattening** key→value
   storage (distinct compound keys resolve distinctly) with belief revision via the contradiction mechanism:
   associative memory as inspectable structure rather than weight updates.
6. **Label-free recognition of a token's structural role** (function vs. content) from the **clustering coefficient
   of its neighbourhood** in the relation graph (a function word bridges unrelated concepts with low clustering,
   content clusters with its kin with high clustering), enabling distributional, supervision-free acquisition of
   grammatical structure.
7. A **conserved** (exact complement map `¬e = −e`; total signed charge zero by construction) and **monotone**
   geometric substrate, in which monotonicity (G6) is realized by a **latent decodable coordinate**: eviction
   de-materialises an element to stay bounded, yet a re-observed symbol re-derives its exact frozen identity, so no
   distinction is unmade.
8. A **thin recurrent controller** that selects and composes substrate operations (rather than storing knowledge in
   weights), with substrate and controller widths chosen independently via a name-based bridge, falling through a
   ladder of **abstaining** substrate operations.
9. A **persistent, substrate-resident self-state** held in the substrate's meaning-space (a decaying accumulation
   of attended meaning-clouds) that conditions the system's relaxation/retrieval, with the substrate's identity conserved across eviction by the
   latent-coordinate mechanism (§8) and persisted learned state (the self-state included) restorable from a checkpoint.

## 12. Reproducibility

All formulas above (face boundaries, the `10^-(i+1)` numeric encodings, the token superposition μ(x), the relation-
as-element composition, the clustering-coefficient role signal, the complement and archival rules) are sufficient to
implement the substrate. A reference implementation exists in the Genesis-Nova codebase:

- faces in `Core/FaceLayout.cs` and `Core/PlatonicFaceComposer.cs`;
- the substrate in `Cognition/Platonic/`: `DialecticalSpace.cs` (distributional cloud, `Merge`/`LearnFact`/
  `TryRecallFact` compositional fact memory, `IsFunctionLike` role recognition), `Element.cs`, `ElementStore.cs`;
- distributional acquisition of structure in `Train/PrebakeLanguageCurriculum.cs`;
- the field-cognition control path in `Infer/GenesisInferenceEngine.Field.cs` (including the meaning-space
  self-state); the controller in `Model/`.

Companion design documents: `PLATONIC_THEORY.md` (formal model), `PLATONIC_NUCLEUS.md` (the dual-face data model),
`PLATONIC_MIND.md` (the founding vision, held lightly), and `PLATONIC_CONSCIOUSNESS.md` (the self-state component,
whose mechanisms are real and whose "consciousness" language is aspirational; see the §10 framing note above).

---

*This disclosure is published defensively to establish prior art as of the date above. No claim is made herein as
to performance benchmarks; the contribution is the architecture and mechanisms enumerated in §11.*
