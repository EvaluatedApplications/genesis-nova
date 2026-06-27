# Genesis-Nova — Model Storage & Sharing

How a Genesis-Nova model is stored, shared via git, seeded, and promoted. A starter model lives in the repo;
people clone it, train their own fork locally in the in-app **gym**, and PR a better one back after tests verify it.

---

## 1. Two tiers: shared vs local

A checkpoint splits along the **shared / local** seam:

| Tier | Contents | Lives in | Git |
|---|---|---|---|
| **Shared starter** — what everyone forks | vocab, embeddings, output/route/edit heads, GRU gates, learned spacing/casing, **+ the platonic substrate** | `models/genesis-nova/` (+ pointer `models/genesis-nova.json`, substrate `models/genesis-nova.platonic/`) | committed |
| **Local working state** — per user | the user's evolving fork as the in-app gym trains it, curriculum/mastery, metrics/logs | `%LocalAppData%/GenesisNova/gym/` | not committed |

The committed starter is a read-only baseline. The repo model changes only when someone promotes a fork and opens a PR.
The committed starter is **three tracked parts** (`.gitignore:10-12`): the pointer `models/genesis-nova.json`, the model
dir `models/genesis-nova/`, and the substrate dir `models/genesis-nova.platonic/`.

---

## 2. Binary, all-`double`

Weights are stored as raw little-endian **`double`** (8 bytes), not JSON text or `float32`. The starter exists to be
*trained further*; `double` keeps a continued fork bit-faithful (no rounding drift down a fork chain) and is ~3× smaller
and faster to parse than text.

---

## 3. Sharded, content-addressed

A model is a **directory of small binary shards + a text manifest**, not one blob:

```
models/genesis-nova/
  manifest.json              # text — the reviewable, PR-able unit
  shards/<sha256>.gnv        # small binary double shards, content-addressed
```

**Shard size: 32 MiB target** (`GenesisShardedCheckpointStore.TargetShardBytes`, `GenesisShardedCheckpointStore.cs:19`).
This sits under GitHub's 50 MB warning / 100 MB block per file, so a single ~100 MB GRU matrix (HiddenSize 2048) splits
into ~4 shards and a full ~460 MB model into ~15 — a flat `shards/` directory, not a sprawling tree.

**Content-addressed (hash-named):** a shard's filename is its SHA-256. Identical shards across forks/saves are stored
**once** (dedup); a "better model" PR is a **manifest diff** (text, reviewable) plus only the shards that changed.

**Sectioned**, so an unchanged tensor keeps its hash and isn't rewritten. Each section (`embeddings`, `outputWeights`,
`gruWih`, … plus JSON sections like `config`, `vocab`, `spacingModel`) records its `kind` (`f64` / `json`), shape, and
ordered shard hashes in the manifest. Big `f64` tensors are raw doubles; small structured state is UTF-8 JSON.

The **platonic substrate** is a *separate* sharded directory (`…/<name>.platonic/`, `GenesisShardedCheckpointStore.cs:28`)
so it can be reset (delete the dir) without disturbing the long-lived NN — the "wipe space, keep GRU" behaviour.

A per-save **generation** id is stamped on the model manifest, the substrate manifest, and the pointer marker, so a load
verifies all parts came from the same save and rejects a torn (crash-interrupted) checkpoint.

The loader reads this sharded format. A one-time legacy-JSON migration path also exists for older single-file checkpoints.

---

## 4. Seed & promote

**Seed** (`GenesisShardedCheckpointStore.SeedFromStarter`, `GenesisShardedCheckpointStore.cs:184`): if NO local gym
checkpoint exists yet and the committed starter is present and consistent, the app copies the starter's pointer + model +
substrate into the local gym dir, so a clone / fresh machine begins from the warmed model instead of an empty brain.
It is a **no-op** once a local checkpoint exists (the local fork then evolves independently and is never overwritten) or
if the starter is absent / torn. Called on startup from `MainWindow` (`MainWindow.cs:61`), before the runtime resumes.

**Promote** (`GenesisShardedCheckpointStore.PromoteToStarter`, `GenesisShardedCheckpointStore.cs:171`): copies a local
fork's pointer + model + substrate (mirroring the destination) into the starter location (`models/genesis-nova{.json,/,.platonic/}`)
so it can be committed and opened as a PR. It is **test-covered but not yet exposed as a UI/CLI action** — promotion is a
deliberate, manual step.

Both take POINTER paths (the `.json` marker); the model/substrate dirs derive from them.

---

## 5. Fork & contribute flow

1. `git clone` → `models/genesis-nova{.json,/,.platonic/}` is your starter.
2. **Seed on app start**: with no local gym checkpoint, the app copies the committed starter →
   `%LocalAppData%/GenesisNova/gym/`; the in-app **gym** trains *that* fork locally. The committed starter is untouched.
3. **Promote**: copy the local fork back into a fresh `models/genesis-nova{.json,/,.platonic/}` (new manifest + changed
   shards), commit, and open a PR.
4. **CI verifies** before merge — held-out generalisation, equal-budget "is it actually better?" (RaceBench), regime /
   retention suites. Manifest content hashes track provenance.

---

## 6. On-disk reference

```
models/genesis-nova.json             # SHARED (committed) — the pointer marker
models/genesis-nova/                 #   the model dir
  manifest.json                      #     sections → shard hashes, dims, generation
  shards/<sha256>.gnv                #     32 MiB content-addressed double shards
models/genesis-nova.platonic/        #   substrate (separate → resettable), same shard layout

%LocalAppData%/GenesisNova/gym/      # LOCAL (not committed) — the in-app gym working dir
  genesis-nova.autosave.checkpoint.json      # local pointer
  genesis-nova.autosave.checkpoint/          # the local fork (sharded, same format)
  genesis-nova.autosave.checkpoint.platonic/
  gym-<skill>-level.txt / personality-level.txt   # per-muscle curriculum levels
```
