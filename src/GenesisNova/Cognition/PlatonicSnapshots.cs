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

/// <summary>A DialecticalSpace element captured FAITHFULLY for an exact round-trip. The only mutable/learned state on
/// an element is its ORBITAL (the semantic-face slice); identity faces are codec-derived from the symbol and its
/// ▷-components are re-derived from the symbol, so we store just the orbital + kind + lifecycle counters. Unlike the
/// legacy node snapshot this INCLUDES atoms and records the kind + Archived flag, so reload reproduces the space
/// byte-for-byte instead of a partially re-synthesized approximation.</summary>
public sealed record DialecticalElementSnapshot(
    string Symbol,
    int Kind,
    double[] Orbital,
    long ObservationCount,
    bool Archived = false);

public sealed record PlatonicMemorySnapshot(
    int FaceDimension,
    PlatonicNodeSnapshot[] Nodes,
    PlatonicRelationSnapshot[] Relations,
    PlatonicChunkSnapshot[]? Chunks = null,
    // Op-tokens (framing/function words excluded from relation formation) are PART of the space's identity — the
    // relation graph was built excluding them, so the space must carry them so a reload stays consistent and the
    // coupling guard keeps working. Optional → backward-compatible with checkpoints written before this field.
    string[]? OperationTokens = null,
    // FAITHFUL DialecticalSpace elements (orbital + kind + counters, incl. atoms). When present, the dialectical core
    // restores from these for an EXACT round-trip; absent (legacy checkpoints) it falls back to Nodes. Optional →
    // backward-compatible.
    DialecticalElementSnapshot[]? Elements = null);
