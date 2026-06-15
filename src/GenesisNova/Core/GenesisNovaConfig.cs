using System;

namespace GenesisNova.Core;

public sealed record GenesisNovaConfig(
    int HiddenSize = 512,
    int RouteCount = 8,
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
    int TrainingTickMultiplier = 16,
    // Upper bound on mid-generation platonic tool invocations for the platonic-assisted
    // reasoning route (route 2). Bounds the interleave so it cannot run away. Default 3.
    int MaxPlatonicAssistInvocations = 3,
    // DECOUPLED platonic face width. 0 (default) = track HiddenSize (legacy behaviour). When > 0, the
    // platonic substrate width is FIXED to this value INDEPENDENT of the GRU controller width — because
    // the model has zero face-dimension-sized parameters (face↔GRU bridge is by NAME, not vector
    // alignment, see FaceDimension), so the substrate (and its exact homomorphism) can stay full-size
    // while the controller (HiddenSize) shrinks. This is the "fixed substrate, variable controller" knob.
    int FaceDimensionOverride = 0)
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
