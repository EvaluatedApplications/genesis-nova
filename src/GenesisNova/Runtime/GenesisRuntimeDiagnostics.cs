namespace GenesisNova.Runtime;

/// <summary>
/// A structured snapshot of WHAT the loaded model is — architecture, sizes, and the state of the platonic
/// substrate (concepts, relations, function-elements, learned operations, mined chunks) — for the diagnostic
/// CLI (tools/GenesisInspect). Produced by <see cref="GenesisEvalAppRuntime.Diagnose"/> off the live runtime
/// state, so it reflects exactly the model REPL/inference uses, not a re-loaded copy.
/// </summary>
public sealed record GenesisRuntimeDiagnostics(
    string CheckpointPath,
    bool CheckpointExists,
    DateTime? CheckpointWriteUtc,
    string Backend,
    int HiddenSize,
    int FaceDimension,
    int VocabularySize,
    long ParameterCount,
    int PlanKindCount,
    int NodeCount,
    int RelationCount,
    int FunctionElementCount,
    int LearnedTransformCount,
    int FoldPathCount,
    int LogLinearFitCount,
    int ChunkTagCount,
    int ChunkCount,
    int AutonomousRounds,
    int MaxNodes,
    int MaxRelations,
    bool SpaceManagerEnabled,
    int RelationBudget,
    RelationSummary[] TopRelations,
    FunctionElementSummary[] FunctionElements,
    TransformSummary[] LearnedTransforms,
    ChunkSummary[] Chunks);

public sealed record RelationSummary(string Left, string Right, long ObservationCount);

/// <summary>READ-ONLY single-concept diagnostics on the live dialectical space (produced by
/// <see cref="GenesisEvalAppRuntime.ProbeConcept"/>): whether the concept is classified function-like and the
/// neighbour-coherence reading (+ thresholds) that drives that, its strong-relation degree and total relation degree,
/// and the nearest neighbours — the headless equivalent of the inspect <c>probe</c>.</summary>
public sealed record ConceptProbe(
    string Concept,
    bool Exists,
    bool IsFunctionLike,
    double NeighbourCoherence,
    double CoherenceThreshold,
    double CoherenceFloor,
    double FunctionEvidence,
    int StrongRelationDegree,
    int RelationDegree,
    int ActiveConcepts,
    IReadOnlyList<ConceptProbeNeighbor> Nearest);

public sealed record ConceptProbeNeighbor(string Symbol, double Distance);
public sealed record FunctionElementSummary(string Symbol, string[] References);
public sealed record TransformSummary(string Name, int ObservationCount, double Confidence, string State);
public sealed record ChunkSummary(string Tag, string Chunk, int Count);
