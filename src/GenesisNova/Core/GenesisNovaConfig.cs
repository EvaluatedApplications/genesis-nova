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
    int MaxPlatonicAssistInvocations = 3)
{
    /// <summary>
    /// THE single source of truth for the platonic face (embedding) width = the FULL model width.
    /// The old "/2" was a dangling experiment with no structural basis — the neural model has zero
    /// face-dimension-sized parameters (the face space and the GRU hidden are bridged by
    /// concept→token-bias by name, not by vector alignment), so halving only threw away platonic
    /// capacity (the word/free region, whose layout offsets are absolute). The face now uses the full
    /// width, giving the space its full capacity for semantic spread. Runtime (GenesisRuntimeState) and
    /// tests (ProductionDims) both derive from here, so this ratio is defined exactly once.
    /// </summary>
    public int FaceDimension => Math.Max(4, HiddenSize);
}
