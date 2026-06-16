# Genesis-Nova — Peer-to-Peer Learning Network

> A protocol for many independent Genesis-Nova nodes — each trained at home on its owner's data and hardware —
> to **pool what they learn into a shared, growing platonic substrate**, without ever sharing raw data and
> without the weight-merging nightmare that makes federated learning hard for ordinary neural nets.

The thesis in one line: **because a Genesis-Nova substrate aligns by NAME and GEOMETRY rather than by opaque
weight coordinates, two nodes can MERGE what they know by union — not by averaging.** That single property is
what makes a real peer-to-peer learning network feasible here when it is impractical for transformers.

---

## 1. Why this architecture is uniquely suited to P2P

Federated learning on conventional models means **averaging weight tensors** across nodes. It is fragile:
weights are a dense, entangled, permutation-sensitive coordinate system; two models that learned "the same
thing" encode it in incompatible bases, so naive averaging destroys both. You need careful synchronization,
identical architectures, and trusted aggregation.

Genesis-Nova splits cleanly into two parts with very different merge properties:

| Component | What it is | Merge property |
|---|---|---|
| **Platonic substrate** | concepts (faces), relations (positioned elements), learned transforms | **Mergeable by union.** Aligns by *name* (concept "apple", relation "apple→fruit") and by *structured geometry* (poly/log/char/word face regions are canonical, not learned bases). Two nodes' substrates compose. |
| **GRU controller** | a *thin selector* that routes/decodes over the substrate | **Local.** Small, fast, cheap to retrain. Stays on-node; we do **not** average GRU weights by default. |

The config makes the decoupling explicit (`GenesisNovaConfig.FaceDimensionOverride`): *"the model has zero
face-dimension-sized parameters — the face space and the GRU hidden are bridged by concept→token-bias BY NAME,
not by vector alignment."* That is the enabling invariant for this whole document: **the valuable, hard-won
knowledge lives in a substrate that two strangers can combine, and the part that's hard to merge (the GRU) is
the cheap part you can just retrain locally against the merged substrate.**

---

## 2. Units of exchange — what actually flows on the wire

Nodes never ship raw training data. They publish small, content-addressed **learning artifacts**:

1. **Concept cards** — a named face: the concept's identity, its structured face vector (poly/log/char/word
   regions), provenance. Source: `PlatonicSpaceMemory.ExportSnapshot()` → `PlatonicMemorySnapshot`, filtered to
   a subset.
2. **Relation edges** — positioned elements: `(subjectName, relationName, objectName, strength)` where
   `strength = 1 − contradiction`. Mergeable; carry their own confidence.
3. **Transforms (skills)** — a learned function `T(f)` as a translation vector plus its **earned reliability**
   (`SuccessCount`/`AttemptCount`, the UCB reputation from the reliability-routing work). Source:
   `TransformAccumulator.ExportSnapshot()` → `TransformEntrySnapshot`. *This is the most exciting unit: a
   portable, verifiable atom of capability* — "this node learned a reliable `×3` / `pluralize` / `celsius→
   fahrenheit` transform; here it is."
4. **(Optional) GRU adapters** — small routing deltas for advanced federation; off by default (see §9 open
   problems).

Every artifact is **content-addressed** (hash of its canonical serialization), **signed** by the contributing
node's key, and tagged with provenance and a version. Hash = identity = automatic dedup across the network.

---

## 3. Merge semantics — union, not average

A receiving node folds an incoming artifact bundle into its own substrate with deterministic rules that
preserve Genesis-Nova's invariants:

- **Concepts** merge by canonical name. Same name + compatible structured face → reinforce. Same name +
  divergent free dims → keep both views, let the geometry/relations disambiguate (the identity dims — a
  number's poly/log value — are ground truth and never conflict).
- **Relations** merge by `(subject, relation, object)`. Conflicting strengths reconcile via the existing
  contradiction model: `strength = 1 − contradiction`, so a relation asserted by many trusted peers
  strengthens, one contradicted by trusted peers weakens. **Hard invariant preserved: numbers never form
  relation edges** — arithmetic is the homomorphism, not stored facts, so numeric edges are rejected at merge
  time (they pollute and erase prior lessons).
- **Transforms** merge by reliability voting. An incoming `T(f)` is **not trusted on faith** — it is admitted
  as a *candidate*, its reliability re-earned locally on the receiver's own probes (`ApplyImprovesOverIdentity`),
  and the peer's claimed reliability only *weights* exploration. A transform that proves itself locally
  graduates to Stable; one that doesn't is retired. The network shares *hypotheses*; each node keeps only what
  it can independently verify.
- **The GRU is then retrained locally** (cheaply) to *use* the enlarged substrate. The substrate grew by union;
  the selector re-learns to select over it. No weight averaging anywhere.

Result: a node that joins the network **inherits the union of everyone's verified concepts, relations, and
skills**, geometrically consistent, with no central server and no shared raw data.

---

## 4. Trust, reputation, and poisoning resistance

An open network invites bad actors. Defenses, mostly reusing primitives the engine already has:

- **Earn-don't-trust by default.** Every imported transform/relation is a *candidate* until it passes local
  verification. The reliability/UCB machinery (`TransformAccumulator.Reliability` / `ReliabilityUcb`) is exactly
  a trust score; the network just seeds exploration, it never grants authority.
- **Node reputation.** Each node accrues reputation from how often its published artifacts survive *other*
  nodes' local verification (a federated, sybil-resistant version of the same reliability signal). Low-rep
  contributions get less exploration budget.
- **Adversarial verification before promotion.** Before a high-impact merge, run the adversarial-verify pattern
  (N independent skeptics try to *refute* the artifact; promote only on majority survival) — the same harness
  used elsewhere in this project for findings.
- **Signed provenance + content addressing** make tampering detectable and contributions attributable.
- **Quarantine + rollback.** Merges land in a staging layer; the SpaceManager's eviction/pruning can roll a bad
  batch back out because artifacts are content-addressed and reversible.

The deep point: **poisoning is hard here because knowledge is verifiable.** A transform either improves
prediction over identity on held-out probes or it doesn't; a relation is either contradicted or it isn't. You
can't smuggle in a lie that survives independent local checking — unlike weight-poisoning, which is invisible.

---

## 5. Privacy & data sovereignty

- **Local-first.** Training happens on-device on the owner's data. Raw data never leaves.
- **Share derived structure, not examples.** Concept cards and transforms are *abstractions* (a `×3` vector, a
  `apple→fruit` edge), not the rows that taught them. Owners choose which subsets of their substrate to publish.
- **Opt-in, revocable, attributable.** Publish nothing, publish a curated slice, or publish broadly — per
  concept namespace. Signed artifacts mean you can attribute (and later retract) your contributions.
- This is the privacy story consumers actually want: *"my model learns from the collective without my files
  ever leaving my machine."*

---

## 6. Network & protocol architecture

```
   ┌─────────────┐        gossip / DHT         ┌─────────────┐
   │   Node A     │◀──────────────────────────▶│   Node B     │
   │ local data   │   content-addressed         │ local data   │
   │ local GRU    │   signed artifacts          │ local GRU    │
   │ substrate ───┼──► publish(concept/relation/─┼──► substrate │
   │              │      transform bundles)      │              │
   └─────┬────────┘                              └──────┬───────┘
         │            ┌───────────────────────┐         │
         └───────────▶│  Artifact registry /  │◀────────┘
                      │  DHT (hash-addressed,  │
                      │  no central authority) │
                      └───────────────────────┘
```

- **Transport:** a gossip overlay (libp2p-style) or a DHT for artifact discovery; no central server required.
  A light "tracker"/relay is optional for NAT traversal and bootstrap only.
- **Artifact store:** content-addressed (hash → bytes), so artifacts are immutable, dedupable, and cacheable by
  anyone. Think "IPFS for learning atoms."
- **Sync cycle (per node, on its own schedule):**
  1. *Publish* — export newly-Stable concepts/relations/transforms, sign, announce hashes.
  2. *Discover* — pull candidate artifacts relevant to the node's namespaces / interests.
  3. *Stage* — import into a quarantine substrate (`ImportSnapshot` into a scratch space).
  4. *Verify* — re-earn reliability locally; adversarially check high-impact items.
  5. *Merge* — union the survivors into the live substrate; retrain the thin GRU to use them.
  6. *Reputation* — report which artifacts survived, updating contributors' standing.
- **Versioning:** artifacts are versioned; the canonical `FaceLayout` (region offsets) is part of the protocol
  version so peers only merge geometry they agree on. Substrate width can differ per node (controller is
  decoupled), but the *face region semantics* must match — that's the wire contract.

---

## 7. The flywheel — why this is the billion-dollar shape

The distribution moat of local AI (llama.cpp, Ollama) is "runs on hardware you already own." Genesis-Nova adds
the thing those ecosystems *can't* easily do: **the local models get smarter together without a central trainer
and without sharing data.** That compounds:

```
 more users on the 6 GB minimum spec
        │  (cheap to run → huge install base)
        ▼
 more local training on diverse private data
        │  (each node a unique data source)
        ▼
 more verified concepts / relations / skills published
        │  (union-merge → the commons grows)
        ▼
 every node inherits a richer shared substrate
        │  (network effect: value ∝ participants)
        ▼
 the platform is more useful → more users  ──► (loop)
```

Defensibility is the **knowledge commons + the verification protocol**, not model scale. You don't have to
out-spend frontier labs on compute; you orchestrate a network that out-*collects* them on verified, structured,
privately-sourced knowledge. Monetization surfaces without breaking the commons: a **marketplace of verified
transforms/skill-packs**, hosted sync/relay and reputation services, private federations for enterprises
(same protocol, closed membership), and certified curated substrates. The free, open commons drives adoption;
the trust/verification/marketplace layer is the business.

---

## 8. MVP roadmap

- **Phase 0 — Manual federation (validate the merge).** Two nodes export substrate + transform snapshots to
  files; a third imports both and shows the union out-performs either alone on held-out probes. *This needs
  almost nothing new — `ExportSnapshot`/`ImportSnapshot` already exist on both the platonic space and the
  transform accumulator.* Proving union-merge beats solo is the whole hypothesis.
- **Phase 1 — Artifact format + signing.** Canonical, content-addressed, signed bundle format for
  concept/relation/transform cards. Local quarantine→verify→merge pipeline.
- **Phase 2 — Discovery overlay.** Gossip/DHT for announce + pull; bootstrap/relay node for NAT traversal.
- **Phase 3 — Reputation + adversarial promotion.** Federated reliability scoring; refute-before-promote on
  high-impact merges; rollback.
- **Phase 4 — Marketplace & private federations.** Skill-pack registry, curated substrates, closed enterprise
  networks on the same protocol.

---

## 9. Open problems (be honest)

- **GRU divergence.** Local selectors drift as substrates diverge. Mitigation: keep the GRU thin and retrain
  against the merged substrate after each merge cycle (cheap, by design). Federated GRU adapters are research,
  not MVP.
- **Merge at scale.** A million-object commons needs efficient namespace partitioning, interest-based pull (you
  don't sync the whole commons), and conflict reconciliation that stays cheap. Relations are positioned vectors
  → watch RAM; partition by namespace.
- **Sybil resistance.** Reputation must resist fake-node farms voting up bad artifacts. Verification-is-local
  helps (you can't fake *my* held-out check), but reputation aggregation needs sybil-hardening.
- **Geometry/version skew.** Peers must share `FaceLayout` semantics to merge. Protocol-versioned regions; refuse
  cross-version merges.
- **Eviction coordination.** Independent nodes evicting different concepts is fine (local autonomy), but the
  *commons* needs a notion of canonical retention so widely-verified knowledge isn't lost.

---

## 10. Mapping to what already exists in this repo

This is not greenfield — the load-bearing primitives are already in the engine:

| Need | Existing primitive |
|---|---|
| Export/import a substrate | `PlatonicSpaceMemory.ExportSnapshot` / `ImportSnapshot` (`PlatonicMemorySnapshot`) |
| Export/import skills + their reputation | `TransformAccumulator.ExportSnapshot` / `ImportSnapshot` (`TransformEntrySnapshot` with `SuccessCount`/`AttemptCount`) |
| Trust / verify a skill | `TransformAccumulator.Reliability` / `ReliabilityUcb` / `ApplyImprovesOverIdentity` (earn-don't-trust) |
| Conflict resolution for relations | relation `strength = 1 − contradiction` (positioned elements) |
| Merge-safety invariant | numbers never form relation edges (rejected at merge) |
| Controller decoupled from substrate | `GenesisNovaConfig.FaceDimensionOverride` (name-bridged, zero face-sized params) |
| Lifecycle / eviction / pruning | `SpaceManager` maintenance + `MaxPlatonicNodes` / `MaxPlatonicRelations` caps |

**Phase 0 is reachable now**: export two nodes' snapshots, import the union into a third, and measure. If the
union beats the parts, the peer-to-peer learning network is real — everything after that is plumbing and trust.
