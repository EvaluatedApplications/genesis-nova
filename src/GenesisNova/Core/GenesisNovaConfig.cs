namespace GenesisNova.Core;

public sealed record GenesisNovaConfig(
    int HiddenSize = 8192,
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
    double TargetVramUtilization = 0.9,
    int ReserveVramMb = 512,
    string? LocalStateDirectory = null,
    bool AutoManagePlatonicSpace = true,
    int MaxPlatonicNodes = 12_000,
    int MaxPlatonicRelations = 48_000,
    double L2RegularizationCoefficient = 0.0);
