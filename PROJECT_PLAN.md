# Genesis-Nova — Project Plan (living, honest)

> A working plan we check off **honestly**. The rule (see `nova-honest-live-acceptance`): a capability is **DONE**
> only when demonstrated in the **live engine** (`PredictAsync` / the running gym / the app) — never when a unit test
> alone passes. Every item is tagged **LIVE** / **BUILT-NOT-WIRED** / **NOT-DONE**.

## The goal
The model **reasons by the NN navigating the platonic space** — the address-space + navigator vision — learning
continuously, remembering, with a real self. The heuristic **route-ladder + cloud retrieval is the old way** we are
replacing.

## The role of the walk (the load-bearing insight)
**There is no perfect structure that answers every question.** The platonic space (faces, relations, clouds) gets the
walker into the **ballpark** — it does the bulk, the *unambiguous* cases, cheaply. The **learned walk, conditioned by
the self, attenuates the *ambiguous* / imperfect structure** — it navigates the cases where the structure is unclear,
where **heuristic routing hits its ceiling and fails.** The walk is the **last mile: from ballpark (~where heuristics
top out) to ~99%.** So the navigator's value is measured **on the ambiguous cases the old ladder gets wrong**, not the
easy ones it already nails. Heuristics own the dominant/clear branch; the walk owns the ambiguous branch.

## Honest audit — snapshot (2026-06-30)
| Capability | State | Note |
|---|---|---|
| Arithmetic (homomorphism) | ✅ LIVE | byte-identical |
| Fact recall / "remember my name" | ✅ LIVE | fixed; survives reload |
| De-hardcoded cues / grammar / function & number words | ✅ LIVE | incl. production dim |
| Address-space *substrate* (faces, orbital, identity) | ✅ LIVE | the model's faces |
| Function-word G6 conservation | ✅ LIVE | committed, proven |
| Persistence (model + nav + self) | ✅ LIVE | round-trip proven |
| **Navigator — reasoning as a walk** | ✅ **LIVE (ambiguous branch, ON in prod)** | M1 DONE: `TryFieldRelax` ambiguous-branch disambiguator (`DecisionPath="navigator-walk"`), `NavigatorDisambiguation` now ON in `WithProductionMechanisms`. LEARNED level cue (`∘gns/∘dom/∘rut`). Live `PredictAsync`, gym-trained-only navigator: **100% (18/18) vs one-shot 0%** on a natural-phrasing 2-hop ambiguous subset (incl. a nonce marker), clear case byte-identical, cold-safe. See `Tests/NavigatorM1LearnedCue.cs`. |
| Decode-from-the-void stack | 🟡 BUILT-NOT-WIRED | only the navigator touches it |
| `query → (anchor, cue)` from real text | ✅ LIVE | learned `∘gns/∘dom/∘rut` cue (`LearnNavLevelCue`); anchor via `ExtractSpecific` |
| Self-loop write side | 🟡 partial | gym-write reverted (pollution); inference fold not wired |
| Multi-hop / composition (lookahead) | ✅ **LIVE (composition + trap)** | M2: kind-conditioned walk does 2-hop CROSS-RELATION composition + clears a 1-hop same-kind LOOKAHEAD TRAP the heuristic can't. Held-out `PredictAsync`: **66.7% (4/6) vs one-shot 0%**, all 4 single-kind held-out persons clear the trap (baseline lands on the 1-hop city). Same-anchor multi-kind selection = partial (abstains). See `Tests/NavigatorM2MultiHop.cs`. |
| Overnight run that *improves real reasoning* | ❌ unproven | infra wired; no number-word feed; DAgger off |

**The gap in one sentence:** we built and proved the reasoning-by-navigation machinery, but the engine still reasons
the old way — **the navigator is a trained passenger, not the driver.**

## Milestones — acceptance = a *live* demonstration
- **M1 — Navigator drives inference, beating heuristics on ambiguity.** `query → anchor + learned ∘qst cue → walk →
  decode landing → answer; abstain if it never settles`, as a route in `GenerateFromField`. The old ladder keeps the
  **dominant/unambiguous** cases (the ballpark); the navigator owns the **ambiguous branch** (competing candidates, no
  dominant relation) where heuristics fail.
  **DONE =** on a real query set in the live `PredictAsync` runtime over the gym-trained space, the navigator route
  **improves accuracy on the ambiguous subset vs the old ladder** (toward ~99%), with honest abstention. Not a unit
  test on synthetic data.
  - M1.1 `query → (anchor, cue)` from a LEARNED cue (no stand-in enum). ✅ **LIVE** — the level cue (genus/domain/root)
    is now a LEARNED `∘gns`/`∘dom`/`∘rut` resolver (`GenesisInferenceEngine.LearnNavLevelCue`/`ResolveLearnedNavCue`,
    mirroring `∘qst`): the level comes from the answer's GRAPH DEPTH above the subject (no word list), and `DeriveNavCue`
    resolves it from the query tokens (highest-confidence wins, abstain on tie, default Genus). A reified marker is
    filtered as filler (`IsNavLevelCue` in `IsFiller`) so it can't steal the subject slot. Generalises to a NONCE marker
    ("zarni") the English keyword switch could never know. The anchor was already real (`ExtractSpecific`).
  - M1.2 navigator route in `GenerateFromField`, gated to the ambiguous branch first. ✅ LIVE (`TryFieldRelax`, after the
    dominant-relation check, before the one-shot `ds.Reason`; falls through on a non-confident/invalid landing).
  - M1.3 abstain wiring (no settle → fall through to one-shot, then `field-abstain`). ✅ LIVE (non-resolve falls through).
  - M1.4 a live eval harness: ambiguous query set, old-ladder vs navigator accuracy. ✅ LIVE (`Tests/NavigatorInferenceM1.cs`
    + the M1 close-out `Tests/NavigatorM1LearnedCue.cs` — learned cue, gym-trained-only navigator).
  - **CUTOVER DONE:** `NavigatorDisambiguation = true` in `WithProductionMechanisms` (R&D default-on); the gym sampler is
    DEEPENED (`SampleNavigatorQueries` emits a genus→root pair for genus HUBS too, chain ≥2, not just leaf members) so
    `TrainNavigatorCycle` ALONE trains the full cue range; a COLD navigator falls through safely (proven, 0 mis-emits);
    fast suite 251/0/47. **REMAINING:** the gym CURRICULUM doesn't yet emit level-cue text frames (the learner is wired
    into `ObserveLearningSignals` but only fires on a "what kind/broadly/ultimately is X→ancestor" frame), so an overnight
    run does not yet self-teach the cue without a warmup (M4); self-conditioning of the walk (M3).
- 2026-06-30 — **M2 LANDED (DONE-LIVE, composition + lookahead trap).** Added null-safe TARGET-KIND conditioning to the
  navigator (`NavQueryPolicyNet._kindEnc`, W_k·kindFace folded into the same un-saturated cue channel as self/cue → no
  head re-wiring, M1 byte-identical when kind absent), threaded through `QueryNavPolicy`/`NavQueryDaggerTrainer`/the
  `NavigatorDisambiguator` hook. Kind learned from the query as a high-degree category hub (`DeriveNavKind`, no word
  list). Deepened the gym sampler (`SampleCompositionQueries`: BFS ≥2-hop targets that belong to a category hub, Genus
  cue so the halt stops on the specific member not the hub, skipped when the degree-climb already reaches the target so a
  pure is-a taxonomy emits none → M1 gym training byte-stable, `NavigatorGymTrainTests` green). LIVE
  `Tests/NavigatorM2MultiHop.cs`: navigator **66.7% (4/6) vs one-shot 0%** held-out, 4/4 single-kind held-out clear a
  1-hop same-kind distractor trap, cold 0 mis-emits, clear case byte-identical. Fast suite **251/0/48**. Remaining =
  same-anchor multi-kind arbitration (M3-adjacent) + self-conditioning placed right (M3).
- **M2 — Multi-hop via composition / lookahead.** ✅ **DONE-LIVE (composition + lookahead trap).** The navigator now
  carries a **target-KIND** conditioning (`NavQueryPolicyNet._kindEnc`, null-safe → M1 byte-identical): the walk is
  seeded/per-hop-biased by the FACE of the answer's category (the "country" hub), learned as a high-degree category (no
  relation-name word list, `GenesisInferenceEngine.DeriveNavKind`). The gym sampler is deepened
  (`SampleCompositionQueries`): for each member, any concept ≥2 hops away that belongs to a category hub becomes a
  Genus+kind composition target, skipped when the cheap degree-climb already reaches it (so a pure is-a taxonomy emits
  none — M1 level training byte-stable). The flow-field oracle to the specific answer routes around the 1-hop distractor.
  **DONE =** held-out `PredictAsync` over `WithProductionMechanisms` (nav ON, trained by `TrainNavigatorCycle` ALONE):
  **navigator 66.7% (4/6) vs one-shot 0%**, 2-hop trajectories proven (`dave->madrid->spain`), all 4 single-kind
  held-out persons clear a 1-hop SAME-KIND distractor trap (baseline always lands on the 1-hop city/distractor),
  cold-safe 0 mis-emits, clear case byte-identical. **REMAINING:** same-anchor MULTI-kind selection (a person with BOTH
  a country and an industry chain) cross-contaminates and abstains — the kind face composes one chain well but doesn't
  yet cleanly arbitrate two equal-depth chains from one anchor.
- **M3 — Self-loop placed right.** Inference folds the *resolution* (not gym drills); re-enable DAgger safely.
  **DONE =** a live ablation shows the self changes the *ambiguous* answers in the running engine; prebake stable.
- **M4 — An overnight run that demonstrably improves reasoning.** number-word gym feed + the loop. **DONE =** the
  warm-history shows held-out navigator resolve% *climbing over the night* — not "it ran".
- **M5 — Cutover & cleanup.** Old ladder demoted to fallback; the 3 skipped routing tests re-earned *as navigation*;
  dead code gone.

## Working log (checked off honestly as we go)
- 2026-06-30 — plan created; dead-code retire pass landing; **M1 is next**, starting at M1.1.
- 2026-06-30 — **M1.2/M1.3/M1.4 LANDED.** Navigator wired into the live `TryFieldRelax` ambiguous branch via an optional
  engine hook (`GenesisInferenceEngine.NavigatorDisambiguator`), set by `GenesisRuntimeState` when `NavigatorDisambiguation`
  is on (default off → fast suite byte-identical, 251/0/46). LIVE `PredictAsync` head-to-head over `WithProductionMechanisms`
  at 1024/2048: navigator **100% (6/6)** vs one-shot **0% (6/6)** on a 2-hop ambiguous subset (`navigator-walk` on all 6),
  clear case `dog` byte-identical on/off. M1.1 (learned cue) still a keyword stand-in; cutover flag not yet flipped.
- 2026-06-30 — **M1.1 + CUTOVER LANDED (M1 DONE-LIVE).** Replaced the English keyword `DeriveNavCue` with a LEARNED level
  cue (`∘gns/∘dom/∘rut`, `LearnNavLevelCue` derives the level from the answer's graph depth — no word list; resolves
  highest-confidence, abstain on tie, default Genus). Reified markers filtered as filler (`IsNavLevelCue`→`IsFiller`) so
  they can't steal the subject slot — that was the one bug (markers like "ultimately"/"zarni" were being picked as the
  query subject). Flipped `NavigatorDisambiguation = true` in `WithProductionMechanisms`; deepened `SampleNavigatorQueries`
  so genus hubs get genus→root pairs (chain ≥2). LIVE demo (`Tests/NavigatorM1LearnedCue.cs`, `WithProductionMechanisms`,
  nav ON by default, policy trained by `TrainNavigatorCycle` ALONE): **navigator 100% (18/18) vs one-shot 0%** on the
  natural-phrasing ambiguous subset (incl. a NONCE marker "zarni" 6/6 — the keyword map could never resolve it), all via
  `navigator-walk`; COLD navigator 0 mis-emits (falls through, cutover-safe); clear case `dog` byte-identical. Fast suite
  **251/0/47** — no regression with the cutover ON. Remaining for a true overnight payoff: gym CURRICULUM level-cue frames
  (M4) + walk self-conditioning (M3).
