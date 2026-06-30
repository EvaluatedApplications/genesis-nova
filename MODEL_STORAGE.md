# Genesis-Nova: Model Storage & Sharing

How a Genesis-Nova model is stored, shared via git, seeded, and promoted. A starter model lives in the repo;
people clone it, train their own fork locally in the in-app **gym**, and PR a better one back after tests verify it.

---

## 1. Two tiers: shared vs local

A checkpoint splits along the **shared / local** seam:

| Tier | Contents | Lives in | Git |
|---|---|---|---|
| **Shared starter** (what everyone forks) | vocab, embeddings, output/route/edit heads, GRU gates, query-op/operand + plan + trunk + role heads, learned spacing/casing/grammar, the **navigator** policy-net + its persistent **self**, **+ the platonic substrate** | `models/genesis-nova/` (+ pointer `models/genesis-nova.json`, substrate `models/genesis-nova.platonic/`) | committed |
| **Local working state** (per user) | the user's evolving fork as the in-app gym trains it, curriculum/mastery, metrics/logs | `%LocalAppData%/GenesisNova/gym/` | not committed |

The committed starter is a read-only baseline. The repo model changes only when someone promotes a fork and opens a PR.
The committed starter is **three tracked parts** (`.gitignore:10-12`): the pointer `models/genesis-nova.json`, the model
dir `models/genesis-nova/`, and the substrate dir `models/genesis-nova.platonic/`.

---

## 2. Binary weights (NN `double`, navigator `float32`)

The NN's big tensors are stored as raw little-endian **`double`** (8 bytes), not JSON text. The starter exists to be
*trained further*; the `double` format keeps a continued fork bit-faithful (no rounding drift down a fork chain) and is
~3× smaller and faster to parse than text. The one exception is the **navigator** policy-net: its params are native
**`float32`**, so they ride in a single concatenated `f32` shard (lossless for f32 params, half the bytes / autosave
I/O of the former f64 encoding — `GenesisShardedCheckpointStore.AddF32Raw`, `:262`). Small structured state is UTF-8 JSON.

---

## 3. Sharded, content-addressed

A model is a **directory of small binary shards + a text manifest**, not one blob:

```
models/genesis-nova/
  manifest.json              # text: the reviewable, PR-able unit
  shards/<sha256>.gnv        # small binary double shards, content-addressed
```

**Shard size: 32 MiB target** (`GenesisShardedCheckpointStore.TargetShardBytes`, `GenesisShardedCheckpointStore.cs:19`).
This sits under GitHub's 50 MB warning / 100 MB block per file, so a single ~100 MB GRU matrix (HiddenSize 2048) splits
into ~4 shards and a full ~460 MB model into ~15. It stays a flat `shards/` directory, not a sprawling tree.

**Content-addressed (hash-named):** a shard's filename is its SHA-256. Identical shards across forks/saves are stored
**once** (dedup); a "better model" PR is a **manifest diff** (text, reviewable) plus only the shards that changed.


**Sectioned**, so an unchanged tensor keeps its hash and isn't rewritten. Each section records its `kind`
(`f64` / `f32` / `json`), shape, and ordered shard hashes in the manifest. The big `f64` tensor sections are
`embeddings`, `outputWeights`, `routeWeights`, `editWeights`, `gruWih`, `gruWhh`, `queryOpWeights`,
`queryOperandWeights`, `planWeights`, `trunkWeights`; the navigator's concatenated weights are the one `f32` section
(`navigator`). Everything else — config, vocab, all biases, GRU gate biases, spacing/casing/grammar-role models,
curriculum, the navigator param *shapes* (values emptied), and the navigator's persistent **self** field
(`NavigatorSelfField`, ~608 doubles) — rides in a **single `meta` JSON section** (`GenesisShardedCheckpointStore.cs:42-61`),
not separate per-field JSON sections. The manifest stamps `FormatVersion` (`1`, `:20`) and the checkpoint schema
`ModelVersion` (currently `7` — `GenesisCheckpoint.CurrentVersion`, navigator + persistent self added at v7).

The **platonic substrate** is a *separate* sharded directory (`…/<name>.platonic/`, `GenesisShardedCheckpointStore.cs:28`)
so it can be reset (delete the dir) without disturbing the long-lived NN. This is the "wipe space, keep GRU" behaviour.
It serialises faithful dialectical elements (`DialecticalElementSnapshot`: orbital + kind + counters + the conserved
`FunctionEvidence` scalar) plus relations, chunks, op-tokens and the learned number-word lexicon, stamped with a
`LayoutVersion` (currently `2`, `PlatonicMemorySnapshot.CurrentLayoutVersion`) — an import stamped below it drops the
layout-dependent element orbitals and re-learns them, keeping only layout-independent state.

A per-save **generation** id is stamped on the model manifest, the substrate manifest, and the pointer marker, so a load
verifies all parts came from the same save and rejects a torn (crash-interrupted) checkpoint.

The loader reads this sharded format. A one-time legacy-JSON migration path also exists for older single-file checkpoints.

---

## 4. Seed & promote

**Seed** (`GenesisShardedCheckpointStore.SeedFromStarter`, `GenesisShardedCheckpointStore.cs:220`): if NO local gym
checkpoint exists yet and the committed starter is present and consistent, the app copies the starter's pointer + model +
substrate into the local gym dir, so a clone / fresh machine begins from the warmed model instead of an empty brain.
It is a **no-op** once a local checkpoint exists (the local fork then evolves independently and is never overwritten) or
if the starter is absent / torn. Called on startup from `MainWindow` (`MainWindow.cs:67`), before the runtime resumes.

**Promote** (`GenesisShardedCheckpointStore.PromoteToStarter`, `GenesisShardedCheckpointStore.cs:207`): copies a local
fork's pointer + model + substrate (mirroring the destination) into the starter location (`models/genesis-nova{.json,/,.platonic/}`)
so it can be committed and opened as a PR. It is **test-covered but not yet exposed as a UI/CLI action**: promotion is a
deliberate, manual step.

Both take POINTER paths (the `.json` marker); the model/substrate dirs derive from them.

---

## 5. Fork & contribute flow

1. `git clone` → `models/genesis-nova{.json,/,.platonic/}` is your starter.
2. **Seed on app start**: with no local gym checkpoint, the app copies the committed starter →
   `%LocalAppData%/GenesisNova/gym/`; the in-app **gym** trains *that* fork locally. The committed starter is untouched.
3. **Promote**: copy the local fork back into a fresh `models/genesis-nova{.json,/,.platonic/}` (new manifest + changed
   shards), commit, and open a PR.
4. **CI verifies** before merge: held-out generalisation, equal-budget "is it actually better?" (RaceBench), regime /
   retention suites. Manifest content hashes track provenance.

---

## 6. On-disk reference

```
models/genesis-nova.json             # SHARED (committed): the pointer marker
models/genesis-nova/                 #   the model dir
  manifest.json                      #     sections → shard hashes, dims, FormatVersion/ModelVersion, generation
  shards/<sha256>.gnv                #     ≤32 MiB content-addressed shards (f64 tensors, one f32 navigator blob, meta JSON)
models/genesis-nova.platonic/        #   substrate (separate → resettable), same shard layout; LayoutVersion-stamped

%LocalAppData%/GenesisNova/gym/      # LOCAL (not committed): the in-app gym working dir
  genesis-nova.autosave.checkpoint.json      # local pointer
  genesis-nova.autosave.checkpoint/          # the local fork (sharded, same format)
  genesis-nova.autosave.checkpoint.platonic/
  gym-<skill>-level.txt / personality-level.txt   # per-muscle curriculum levels
```
