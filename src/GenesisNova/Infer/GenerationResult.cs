namespace GenesisNova.Infer;

public sealed record GenerationResult(
    string Output,
    int[] GeneratedTokens,
    bool UsedPlatonicQuery = false,
    bool UsedNeuralFallback = false,
    string DecisionPath = "neural-token",
    double PlatonicConfidence = 0.0,
    int AppliedBiasCount = 0,
    double AverageBiasMagnitude = 0.0,
    int ChunksGenerated = 1,
    int PlatonicHopCount = 0,
    string? RoutedTransform = null,
    string? TransformIntercept = null);
