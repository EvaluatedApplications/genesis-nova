using System;

namespace GenesisNova.Core;

public sealed record GenesisNovaConfig(
    int HiddenSize = 512,
    double LearningRate = 0.1,
    int Seed = 42,
    bool EnableParallelMath = true,
    int MaxDegreeOfParallelism = 0,
    bool Deterministic = false,
    ComputeBackend Backend = ComputeBackend.Gpu,
    bool AutoPersist = true,
    bool AutoResume = false,
    // Reload the checkpoint mid-session when its file changes on disk. This existed for the RETIRED ClaudeMemory
    // daemon (a SEPARATE process that wrote checkpoints the REPL should pick up). The gym is now IN-PROCESS and the
    // sole writer, so watching makes every predict reload the runtime's OWN autosave — a lossy mid-training
    // teardown+rebuild that degrades the model. Default OFF; only enable if an external process writes the live
    // checkpoint. See [[nova-save-reload-lifecycle]].
    bool WatchExternalCheckpoint = false,
    bool AutoScaleVram = true,
    double TargetVramUtilization = 0.82,
    int ReserveVramMb = 1536,
    string? LocalStateDirectory = null,
    bool AutoManagePlatonicSpace = true,
    // Platonic-space capacity caps, sized for a SIMPLE-LLM-scale concept graph (vocab + learned
    // concepts + their relations). These are CAPS, not allocations — memory grows only as concepts
    // accumulate. The single source of truth: the runtime passes these into PlatonicSpaceMemory so its
    // hard eviction and the SpaceManager's maintenance pruning share the same limits.
    int MaxPlatonicNodes = 100_000,
    int MaxPlatonicRelations = 500_000,
    // HARD active-concept ceiling for the DialecticalSpace core (the default substrate). Relevance-decay alone cannot
    // bound a corpus stream (every re-observed word stays "recently seen" → never goes stale), so the space enforces
    // this cap by distributional VALUE: when the active concept count exceeds it, the lowest-utility excess concepts are
    // evicted down to the cap (∘-anchors / atoms / Functions / numbers / ▷-referenced components are protected). 0 =
    // unbounded (legacy). Sized for a vocab + learned-concept graph; the pure-overflow safety net for prebake bloat.
    int MaxActiveConcepts = 12_000,
    double L2RegularizationCoefficient = 0.0,
    // DECOUPLED platonic face width. 1024 (default) = the PRODUCTION substrate width: frozen address bands
    // [0,416) + a 608-dim learned orbital tail [416,1024) (see FaceLayout). Set 0 to track HiddenSize (legacy
    // behaviour); any other >0 pins a fixed width. The width is INDEPENDENT of the GRU controller width because
    // the model has zero face-dimension-sized parameters (face↔GRU bridge is by NAME, not vector alignment, see
    // FaceDimension), so the substrate (and its exact homomorphism) stays full-size while the controller
    // (HiddenSize) is chosen separately. This is the "fixed substrate, variable controller" knob.
    int FaceDimensionOverride = 1024,
    // EDGE-FOLLOWING RETRIEVAL at inference. true (default) = the full route ladder (relation-first edge +
    // multi-hop concept-chain walk) is available below proximity kNN. false = retrieval is proximity-kNN ONLY
    // (geometric "position IS identity"); relation edges are still observed/trained (attraction+repulsion shape
    // the geometry) but never FOLLOWED to answer a query. Experiment knob for "is the geometry enough to route?".
    bool EdgeRoutingEnabled = true,
    // RUNG 2 function gradient (PLATONIC_BACKPROP.md): descend softmax-CE so a query anchor retrieves a valid task
    // answer (pull target, push confusers, self-scaled). false (default) = built but dormant; flip true to A/B it
    // against Rung 1's gradient-free disruption (which is always on).
    bool FunctionGradientEnabled = false,
    // SUBSTRATE SELECTOR (PLATONIC_THEORY.md §11 rebuild). true (default, M4 2026-06-23) = the ground-up
    // Platonic.DialecticalSpace (born-neutral, per-aspect κ, composition hubs, recognition hierarchy) — validated:
    // GRU learns retrieval 100%/routed 100% through it, arithmetic exact, separation improves under long training
    // (anti-erosion). false = the legacy PlatonicSpaceMemory (kept as fallback). Both satisfy IPlatonicSpace.
    bool UseDialecticalCore = true,
    // LIVING SELF (PLATONIC_CONSCIOUSNESS.md). When true the GRU runs SELF-CONDITIONED: a persistent self threads
    // every thought (training folds into it, talking proceeds from it, it is checkpointed). false (default) = the
    // stateless contract, byte-identical — so unit tests are unaffected; the desktop app turns it on to wake alive.
    bool LivingSelf = false,
    // KEEP-CORE control path (PLATONIC_RECKONING.md). When true the substrate's OWN confidence drives the controller:
    // training labels/perception anchor on the discriminative cue (the seam fix), inference makes RELAXATION the
    // primary retrieval route, and a non-arithmetic query that nothing settles ABSTAINS instead of hallucinating via
    // the neural decoder. false (default) = the classifier-gated path, byte-identical; the desktop app turns it on.
    bool KeepCoreControl = false,
    // CONSCIOUS FIELD (PLATONIC_MIND.md / PLATONIC_CONSCIOUSNESS.md) — the real architecture. When true the model
    // thinks by the field RELAXING to a settled state (compute → relax → abstain) and the route/plan/op classifier
    // is bypassed ENTIRELY (no "neural vs platonic" head). false (default) keeps the legacy classifier path for
    // existing tests; the desktop app turns it on. This is the subtraction the reckoning pointed at, made real.
    bool ConsciousField = false,
    // DE-HARDCODED DISPATCH — the engine answers number↔word from the LEARNED lexicon (not the NumberWordVocabulary
    // codec) and routes compare / to-word / to-digit / question / retrieval by LEARNED cues (not the IsCompareCue /
    // IsToWordCue / IsToDigitCue / IsQuestionCue / IsRetrievalMarker word-lists). false (default) keeps the codec+lists
    // for existing/zero-shot tests; the production app turns it on so the model uses what it LEARNED (warm-start: the
    // gym bootstraps the lexicon/cues once this flips the codec off). See nova-learned-number-words / -op-cues.
    bool DeHardcodedDispatch = false,
    // GENERATIVE field routes (PLATONIC_MIND.md). FieldTicks = the genesis tick cascade (numeric + meaning) runs a
    // query as a frontier reduced over ticks, manufacturing intermediate elements. MeaningOps = generative ops in the
    // large/word face (compose two meanings, analogy by relation-vector). Default false (byte-identical); the app
    // turns them on so the field REASONS over the substrate instead of only retrieving from it.
    bool FieldTicks = false,
    bool MeaningOps = false,
    // Substrate perf: deferred batched-GPU cloud recompute (Cloud = A·T on CUDA) replacing per-observation scalar
    // RecomputeCloud. Default false (byte-identical scalar path); when on, observations defer + flush in batches.
    bool BatchedCloudGpu = false,
    // NAVIGATOR DISAMBIGUATION (M1, PLATONIC_NAVIGATOR.md). When true, GenesisRuntimeState attaches the trained shared
    // navigator (State.Navigator) to the inference engine as the AMBIGUOUS-BRANCH disambiguator: a query that reaches
    // the no-dominant-relation case is answered by WALKING the platonic space (multi-hop) instead of a single-shot
    // ds.Reason. false (default) ⇒ the hook is null and the ambiguous branch is byte-identical to the one-shot path
    // (so the fast suite is unaffected). ON in WithProductionMechanisms (the M1 cutover, R&D default-on); the field
    // default stays false so a bare config / the fast suite is byte-identical to the one-shot path.
    bool NavigatorDisambiguation = false,
    // DECODE-FROM-THE-VOID RECOVERY (G6 via the latent address). When true the substrate RE-MATERIALISES an evicted /
    // latent concept from its COORDINATE on demand (DialecticalSpace.RecoverFromCoordinate): the frozen identity bands
    // [0,416) are a deterministic invertible codec, so the IDENTITY survives eviction even though the learned orbital +
    // store slot were freed. The materialised space becomes a CACHE over the conserved decodable void — the navigator's
    // walk (TryLand) and reasoning (Reason) decode a missing concept back when they reach its address, instead of missing.
    // false (default) = the substrate misses on an evicted/latent coordinate, byte-identical. ON in WithProductionMechanisms.
    bool DecodeFromVoidRecovery = false,
    // SELF-HEAL MISROUTED CUES — the missing "learn from a WRONG ROUTE" signal in training. When true, a value-wrong
    // training probe whose TRUE answer is a NUMBER but whose produced answer is a compare WORD (an arithmetic query
    // hijacked to the predicate/compare route) CONTRADICTS the operator/cue that selected that route, so training
    // UNLEARNS the bad cue→∘cmp relation and the skill recovers. Without it a corpus-contaminated operator symbol
    // ("-"→∘cmp) is immortal and focused training can never fix it (see nova-subtract-stuck-compare-hijack). false
    // (default) = byte-identical (no disruption fires). ON in WithProductionMechanisms.
    bool SelfHealMisroutedCues = false)
{
    /// <summary>
    /// Platonic face (embedding) width. By default equals the GRU width (HiddenSize); when
    /// <see cref="FaceDimensionOverride"/> &gt; 0 it is pinned to that fixed value independent of the
    /// controller width. Independence is sound because the neural model has zero face-dimension-sized
    /// parameters — the face space and the GRU hidden are bridged by concept→token-bias BY NAME, not by
    /// vector alignment — so the substrate width can be chosen separately from the controller capacity.
    /// </summary>
    public int FaceDimension => Math.Max(4, FaceDimensionOverride > 0 ? FaceDimensionOverride : HiddenSize);

    /// <summary>
    /// THE PRODUCTION ARCHITECTURE — the single definition of "the mechanisms we actually ship", applied on top of
    /// whatever infrastructure (dims, dirs, backend, seed) the caller has set. Turn the conscious-field cognition on
    /// (DialecticalSpace substrate + field-relaxation, KeepCore seam, the meaning-space self) and the function
    /// gradient. The desktop app (MainWindow) AND the benchmark (RaceBench) both build from this, so the race always
    /// runs the same brain the app does — change a mechanism HERE, in one place, and both follow. (Mechanism toggles
    /// only; deployment infra stays on the caller's config.)
    /// </summary>
    public GenesisNovaConfig WithProductionMechanisms() => this with
    {
        UseDialecticalCore = true,    // the ground-up dialectical substrate (required by the conscious field)
        ConsciousField = true,        // think by field-relaxation; bypass the route/plan/op classifier
        KeepCoreControl = true,       // substrate-confidence control + discriminative-anchor seam + abstain
        DeHardcodedDispatch = true,   // number↔word + compare/format/question/retrieval cues are LEARNED, not hardcoded
        LivingSelf = true,            // the meaning-space self conditions ambiguous reasoning
        FunctionGradientEnabled = true, // Rung 2 — descend the function gradient alongside Rung 1 disruption
        EdgeRoutingEnabled = true,    // full retrieval ladder available (default, pinned here for clarity)
        FieldTicks = true,            // the genesis tick cascade (numeric + meaning) runs live
        MeaningOps = true,            // generative compose/analogy in the large face — GATED by the LEARNED director
                                      // (attached in ApplyTo with a conservative prior → defaults to retrieval, opens
                                      // only as it learns; no free-form misfire).
        NavigatorDisambiguation = true, // M1 CUTOVER — the trained navigator owns the AMBIGUOUS branch of TryFieldRelax
                                      // (multi-hop walk), gated to a CONFIDENT halt: an untrained/cold walk does not
                                      // confidently resolve → falls through to the one-shot reason (cold-safe, proven).
        DecodeFromVoidRecovery = true, // the materialised space is a CACHE over the conserved decodable void — the walk
                                      // (TryLand) and reasoning (Reason) DECODE + re-materialise an evicted/latent concept
                                      // from its coordinate on demand, guarded to a confident valid decode (no junk).
        SelfHealMisroutedCues = true, // training LEARNS FROM A WRONG ROUTE — a numeric example answered via the compare
                                      // route contradicts the operator/cue that hijacked it, so a corpus-contaminated
                                      // cue→∘cmp edge is UNLEARNED and the skill recovers (else it's immortal, stuck).
        BatchedCloudGpu = true,       // PERF: GPU-batched RecomputeCloud (Cloud = A·T via index_select/index_add_ on
                                      // CUDA) replacing the scalar per-observation hot loop that was ~90% of training
                                      // CPU. Fully wired (defer+flush; every read flushes via EnsureCloudsFresh); cloud
                                      // cos≈1.0 vs scalar (float32 write-back, semantically safe for gym retrieval).
    };
}
