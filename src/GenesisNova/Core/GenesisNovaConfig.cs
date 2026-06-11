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
    int MaxPlatonicNodes = 12_000,
    int MaxPlatonicRelations = 48_000,
    double L2RegularizationCoefficient = 0.0,
    int TrainingTickMultiplier = 16,
    // Upper bound on mid-generation platonic tool invocations for the platonic-assisted
    // reasoning route (route 2). Bounds the interleave so it cannot run away. Default 3.
    int MaxPlatonicAssistInvocations = 3);
