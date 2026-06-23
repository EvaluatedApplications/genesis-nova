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
    double L2RegularizationCoefficient = 0.0,
    // DECOUPLED platonic face width. 0 (default) = track HiddenSize (legacy behaviour). When > 0, the
    // platonic substrate width is FIXED to this value INDEPENDENT of the GRU controller width — because
    // the model has zero face-dimension-sized parameters (face↔GRU bridge is by NAME, not vector
    // alignment, see FaceDimension), so the substrate (and its exact homomorphism) can stay full-size
    // while the controller (HiddenSize) shrinks. This is the "fixed substrate, variable controller" knob.
    int FaceDimensionOverride = 0,
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
    bool LivingSelf = false)
{
    /// <summary>
    /// Platonic face (embedding) width. By default equals the GRU width (HiddenSize); when
    /// <see cref="FaceDimensionOverride"/> &gt; 0 it is pinned to that fixed value independent of the
    /// controller width. Independence is sound because the neural model has zero face-dimension-sized
    /// parameters — the face space and the GRU hidden are bridged by concept→token-bias BY NAME, not by
    /// vector alignment — so the substrate width can be chosen separately from the controller capacity.
    /// </summary>
    public int FaceDimension => Math.Max(4, FaceDimensionOverride > 0 ? FaceDimensionOverride : HiddenSize);
}
