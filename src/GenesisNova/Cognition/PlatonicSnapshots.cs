namespace GenesisNova.Cognition;

public sealed record PlatonicNodeSnapshot(
    string Name,
    double[] PositiveFace,
    double[] NegativeFace,
    int ObservationCount,
    int UseCount = 0,
    int SuccessCount = 0,
    int FailureCount = 0,
    long LastUsedStep = 0);

public sealed record PlatonicRelationSnapshot(
    string Left,
    string Right,
    double ThesisContradiction,
    double LastObservedContradiction,
    double SynthesisContradiction,
    int ObservationCount,
    int UseCount = 0,
    int SuccessCount = 0,
    int FailureCount = 0,
    long LastUsedStep = 0);

/// <summary>One mined scaffold chunk and how many graded-correct outputs reinforced it, grouped by tag.</summary>
public sealed record PlatonicChunkSnapshot(
    string Tag,
    string Chunk,
    int Count);

public sealed record PlatonicMemorySnapshot(
    int FaceDimension,
    PlatonicNodeSnapshot[] Nodes,
    PlatonicRelationSnapshot[] Relations,
    PlatonicChunkSnapshot[]? Chunks = null,
    // Op-tokens (framing/function words excluded from relation formation) are PART of the space's identity — the
    // relation graph was built excluding them, so the space must carry them so a reload stays consistent and the
    // coupling guard keeps working. Optional → backward-compatible with checkpoints written before this field.
    string[]? OperationTokens = null);
