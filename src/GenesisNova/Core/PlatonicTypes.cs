using System.Collections.Immutable;

namespace GenesisNova.Core;

/// <summary>
/// Element kind in the platonic space.
/// </summary>
public enum ElementKind
{
    /// <summary>Vocabulary/embedding basis</summary>
    Object = 1,
    
    /// <summary>Relation between objects</summary>
    Relation = 2,
    
    /// <summary>Composition (sum/product of elements)</summary>
    Composition = 3,
    
    /// <summary>Function/agent with local transform</summary>
    Function = 4
}

/// <summary>
/// Composition mode for function elements.
/// </summary>
public enum CompositionMode
{
    /// <summary>Sum embeddings</summary>
    Sum = 1,
    
    /// <summary>Element-wise product</summary>
    Product = 2,

    /// <summary>Element-wise difference</summary>
    Difference = 3,

    /// <summary>Segment-wise concatenation</summary>
    Concatenate = 4
}

/// <summary>
/// A single element in the platonic space.
/// </summary>
public record PlatonicElement(
    int Id,
    ElementKind Kind,
    double[] Embedding,
    string Symbol,
    int GeneratedAtTick,
    double NoveltyScore = 0.5,
    double BridgeConfidence = 0.5,
    ImmutableArray<int> RelatedTo = default,
    string? GenerationPath = null
);
// (R6/R7/R9-only fields — ComplementId, IsHypothesis, LocalTransform*, IsDevolved/Devolved* — were
// removed 2026-06-14 with the disconnected tick rules that set them; nothing read them. See TickExecutor.)

/// <summary>
/// The complete state of the platonic space at a moment in time.
/// </summary>
public record PlatonicState(
    ImmutableArray<PlatonicElement> Elements,
    int EmbeddingDimension,
    int NextId = 0,
    int CurrentTick = 0,
    Dictionary<string, double[]>? EmbeddingCache = null
)
{
    public Dictionary<string, double[]> EmbeddingCacheOrEmpty => EmbeddingCache ?? new();
}

/// <summary>
/// A sequence of tick actions to execute.
/// </summary>
public record TickSequence(
    ImmutableArray<TickAction> Actions,
    double[] ActionLogits,
    double Confidence
);
