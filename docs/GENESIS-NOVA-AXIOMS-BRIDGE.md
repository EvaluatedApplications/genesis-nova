# Genesis Nova — Axioms-First Bridge to Modern ML

This document is the reboot contract for Genesis Nova.

It treats the Genesis axioms as the source, and derives architecture/training/evaluation from them.
No legacy heuristic is allowed into the core unless it can be derived from an axiom or shown as a temporary bootstrap with an explicit removal plan.

## 1) Axioms (source layer)

1. **G1 Consciousness / Agentive selection** — the system chooses what to process next.
2. **G2 Non-contradiction** — incompatible beliefs cannot coexist unresolved.
3. **G3 Generative observation** — each observation can produce new structure.
4. **G4 Conservation** — every generated distinction has a balancing dual.
5. **G5 Recursive availability** — any represented concept can be observed again.
6. **G6 Irreversibility** — knowledge history is append-only (never silently deleted).

## 2) Modern ML interpretation (derived layer)

| Axiom | ML interpretation | Architecture consequence |
|---|---|---|
| G1 | Learnable control policy over compute | Router/policy head selects experts/tools/attention budget |
| G2 | Consistency constraints in latent and output space | Contradiction penalty + consistency checks in eval |
| G3 | Self-expanding predictive world model | Autoregressive decoding + optional latent world-state update |
| G4 | Balanced representation dynamics | Paired/anti-symmetric latent channels and conservation regularizer |
| G5 | Re-observable memory | Retrieval-ready memory tokens/state, revisitable by attention |
| G6 | Monotonic epistemic history | Append-only event/memory log; corrections are superseding events |

## 3) Core model contract (no hacks in core path)

Single trainable path:

1. Encode input + retrieved memory.
2. Predict route/expert distribution.
3. Decode output autoregressively.
4. Update memory/event log.

Single inference path:

1. Encode.
2. Route.
3. Decode to EOS.

No fallback ladders, no parallel handcrafted inference engines, no k-NN rescue in the default path.

## 4) Loss as axiom instrumentation

Use composite objective:

`L = λtok * Lce + λroute * Lroute + λconsistency * Lg2 + λconservation * Lg4 + λmemory * Lg6`

- **Lce**: next-token cross entropy.
- **Lroute**: expert/router supervision or distillation.
- **Lg2**: contradiction loss (same prompt/context should not support mutually exclusive claims).
- **Lg4**: conservation loss on paired latent channels.
- **Lg6**: memory monotonicity loss (updates as append/supersede, never erase).

## 5) Bootstrap policy (allowed, but temporary)

Bootstrap tricks are allowed only if all three are true:

1. Explicitly marked as bootstrap.
2. Measured against an axiom-derived metric.
3. Assigned a removal milestone.

Example: deterministic numeric encoding can be used to warm-start arithmetic, but must be retired once differentiable numeric representations pass parity.

## 6) Evaluation: prove first-principles behavior

Track three classes of metrics:

1. **Task metrics**: accuracy, exact match, perplexity.
2. **Axiom metrics**:
   - G2 contradiction rate
   - G4 conservation drift
   - G6 memory overwrite rate
3. **Generalization metrics**:
   - held-out templates
   - held-out numeric ranges
   - composition transfer tasks

Shipping criterion is not raw accuracy alone; it is accuracy with stable axiom metrics.

## 7) Immediate build order

1. Implement typed axiom map in code (axiom -> objective -> metric).
2. Add minimal differentiable trainer skeleton (token + route + axiom losses).
3. Add evaluation harness that reports task + axiom metrics together.
4. Only then add REPL/training UX.

This enforces the rule: **theory drives implementation, not vice versa**.

