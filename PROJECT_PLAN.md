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
| **Navigator — reasoning as a walk** | 🟡 **BUILT-NOT-WIRED** | gym-trained + `/nav` only; NOT in `GenerateFromField`. The model does not yet reason by walking. |
| Decode-from-the-void stack | 🟡 BUILT-NOT-WIRED | only the navigator touches it |
| `query → (anchor, cue)` from real text | ❌ NOT-DONE | stand-in / explicit `/nav` |
| Self-loop write side | 🟡 partial | gym-write reverted (pollution); inference fold not wired |
| Multi-hop / composition (lookahead) | ❌ NOT-DONE | geometry is 1-hop; ceiling ~64–78% |
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
  - M1.1 `query → (anchor, cue)` from the learned `∘qst` machinery (no stand-in enum).
  - M1.2 navigator route in `GenerateFromField`, gated to the ambiguous branch first.
  - M1.3 abstain wiring (no settle → `platonic-abstain`).
  - M1.4 a live eval harness: ambiguous query set, old-ladder vs navigator accuracy.
- **M2 — Multi-hop via composition / lookahead.** **DONE =** held-out multi-hop climbs clearly above the 1-hop ceiling
  *through the live route*.
- **M3 — Self-loop placed right.** Inference folds the *resolution* (not gym drills); re-enable DAgger safely.
  **DONE =** a live ablation shows the self changes the *ambiguous* answers in the running engine; prebake stable.
- **M4 — An overnight run that demonstrably improves reasoning.** number-word gym feed + the loop. **DONE =** the
  warm-history shows held-out navigator resolve% *climbing over the night* — not "it ran".
- **M5 — Cutover & cleanup.** Old ladder demoted to fallback; the 3 skipped routing tests re-earned *as navigation*;
  dead code gone.

## Working log (checked off honestly as we go)
- 2026-06-30 — plan created; dead-code retire pass landing; **M1 is next**, starting at M1.1.
