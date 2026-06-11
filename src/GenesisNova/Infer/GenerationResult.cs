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
    string? TransformIntercept = null,
    IReadOnlyList<GenesisNova.Cognition.PlatonicEvidence>? Evidence = null,
    // Introspective (platonic-assisted reasoning) telemetry — additive, backward-compatible.
    // PlatonicAssistInvocations: number of mid-generation platonic tool calls attempted.
    // PlatonicAssistFired: number of those calls that produced an injected sub-result.
    int PlatonicAssistInvocations = 0,
    int PlatonicAssistFired = 0);
