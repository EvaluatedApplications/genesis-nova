# Genesis-Nova — Model Storage & Sharing

How a Genesis-Nova model is saved, sharded, shared via git, and forked. The goal: **a starter model people
clone from the repo, train their own fork locally, and PR a better one back — after tests verify it.**

---

## 1. Two tiers: shared vs local

A checkpoint mixes three things; we split them along the **shared / local** seam:

| Tier | Contents | Lives in | Git |
|---|---|---|---|
| **Shared model** — the starter everyone forks | vocab, embeddings, output/route/edit heads, **GRU gates**, learned spacing/casing, **+ the platonic substrate** | `models/genesis-nova/` (repo) | committed |
| **Local working state** — per user | the user's evolving fork as they train, curriculum/mastery (`TrainerLearningState`), autonomous history, conversation, metrics/queue/logs | `.claude-nova/` | gitignored |

The committed starter is a **read-only baseline**. On first run the daemon copies it into the local working
fork and trains *that*; the repo model only changes when someone **promotes** a fork and opens a PR.

---

## 2. Binary, all-`double`

Weights are stored as raw little-endian **`double`** (8 bytes), not JSON text or `float32`:
- **Exact resume.** The starter exists to be *trained further*; `float32` injects rounding the moment a forker
  continues, and that drift compounds down a fork chain. `double` keeps a continued fork bit-faithful.
- **Smaller + faster than JSON** anyway (~3× smaller, far quicker to parse than text `double[]`).

---

## 3. Sharded, content-addressed

A model is a **directory of small binary shards + a text manifest**, not one blob:

```
models/genesis-nova/
  manifest.json              # text — the reviewable, PR-able unit
  shards/<sha256>.gnv         # small binary double shards, content-addressed
```

**Shard size: 32 MiB target** (`GenesisShardedCheckpointStore.TargetShardBytes`). Why:
- GitHub **warns at 50 MB**, **blocks at 100 MB** per file. A single GRU matrix at HiddenSize 2048 is ~100 MB —
  already over the limit — so even one tensor *must* split.
- 32 MiB sits comfortably under the 50 MB warning (no GitHub nags), splits a ~100 MB tensor into ~4 shards and a
  full ~460 MB model into ~15 — a **flat, simple `shards/` directory**, not a sprawling tree (git stays sane).
- Power-of-two aligned; tunable in one constant.

**Content-addressed (hash-named):** a shard's filename is its SHA-256. So:
- identical shards across forks/saves are stored **once** (dedup),
- a "better model" PR is a **manifest diff** (text, reviewable) plus only the shards that actually changed,
- shards double as the **P2P artifacts** (see `P2P_LEARNING_NETWORK.md`).

**Sectioned**, so an unchanged tensor keeps its hash and isn't rewritten. Each section
(`embeddings`, `outputWeights`, `gruWih`, … plus JSON sections like `config`, `vocab`, `spacingModel`) records
its `kind` (`f64` / `json`), shape, and ordered shard hashes in the manifest. Big `f64` tensors are raw doubles;
small structured state is UTF-8 JSON — both chunked the same way.

The **platonic substrate** is a *separate* sharded directory (`…/<name>.platonic/`) so it can be reset (delete
the dir) without disturbing the long-lived NN — preserving the existing "wipe space, keep GRU" behaviour.

> When the committed total approaches ~1 GB, move the large shards to **Git LFS** or distribute them as a
> **GitHub Release asset / download-on-first-run**. The manifest stays the source of truth, so this switch does
> **not** change the fork/PR workflow — only where the bytes live.

---

## 4. Backwards compatibility + cleanup

Older checkpoints are the single JSON file `genesis-nova.autosave.checkpoint.json` (+ `*.platonic.json`
companion), `Version ≤ 3`.

- **Migrate on first load.** If the sharded model dir is absent but the legacy JSON exists, load it with the
  legacy reader, re-save in the sharded binary format, then **clean up** the legacy files (move them to
  `.claude-nova/.legacy-backup/` — out of the active directory so there's **no confusion about what's relevant**,
  but recoverable). One-time, idempotent.
- **Delayed hard delete.** The legacy-JSON reader + migration path is marked `// LEGACY: delete after <release>`
  to be removed once migration is proven in the wild.
- Your current 2048 / face-512 model migrates cleanly (dims, vocab, GRU, substrate all carry over).

---

## 5. Fork & contribute flow

1. `git clone` → `models/genesis-nova/` is your starter.
2. Daemon first run: copy starter → `.claude-nova/fork/`; train the fork locally (your custom regimes, data,
   training-set generators). The committed starter is untouched.
3. **Promote**: export your fork to a fresh `models/genesis-nova/` (new manifest + changed shards), open a PR.
4. **CI verifies** before merge — held-out generalisation, equal-budget "is it actually better?" (RaceBench),
   regime/retention suites. Manifest lineage + content hashes track provenance.

So "start from my work, train your own, PR a better one" falls straight out of the layout.

---

## 6. Roadmap

1. **Binary sharded store + manifest + migration + cleanup** — behind the existing Save/Load surface (this doc's
   core; daemon keeps working throughout). ← *implemented first*
2. Repo `models/` layout + manifest lineage/hash + copy-to-fork-on-first-run + `promote`/`export` command.
3. PR-with-tests CI verification gate.
4. Training-set generator packages (`ITrainingSetGenerator` — the "ability library").
5. P2P platonic-element coordination (reuses `ExportSnapshot`/`ImportSnapshot` + reliability/UCB +
   content-addressed shards from `P2P_LEARNING_NETWORK.md`).

---

## 7. On-disk reference

```
models/genesis-nova/                 # SHARED (committed)
  manifest.json                      #   sections → shard hashes, dims, lineage, tests-passed
  shards/<sha256>.gnv                #   32 MiB content-addressed double shards
models/genesis-nova.platonic/        #   substrate (separate → resettable), same shard layout

.claude-nova/                        # LOCAL (gitignored)
  genesis-nova.autosave.checkpoint/        # the local fork (sharded, same format)
  genesis-nova.autosave.checkpoint.platonic/
  training-state / metrics / queue / logs  # local runtime + curriculum
  .legacy-backup/                          # migrated-away legacy JSON (safe to delete)
```
