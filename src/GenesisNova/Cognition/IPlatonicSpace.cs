namespace GenesisNova.Cognition;

/// <summary>
/// The stable CONTRACT at the Cognition boundary: the public surface that EXTERNAL consumers (inference,
/// training, runtime, config) depend on, so the substrate internals (<see cref="PlatonicSpaceMemory"/>)
/// can be reworked without breaking them. This declares exactly the public INSTANCE members of
/// <see cref="PlatonicSpaceMemory"/>; the static <c>IsReservedConcept</c>, the <c>SeqScaffoldTag</c> const,
/// and the nested record types stay on the concrete type (referenced here as qualified
/// <c>PlatonicSpaceMemory.*</c>).
/// </summary>
public interface IPlatonicSpace
{
    // Function / word elements (Snapshots partial).
    IReadOnlyList<Core.PlatonicElement> FunctionElements { get; }
    IReadOnlyList<Core.PlatonicElement> WordElements { get; }
    Core.PlatonicElement RegisterFunctionElement(string name, IReadOnlyList<string>? references = null);
    bool TryGetFunctionElement(string name, out Core.PlatonicElement element);
    Core.PlatonicElement RegisterWordElement(string concept);
    bool TryGetWordElement(string concept, out Core.PlatonicElement element);
    IReadOnlyList<Core.PlatonicElement> DecomposeWordElement(string concept);

    // Chunk mining (Snapshots partial).
    void MineChunk(string tag, string chunk);
    bool TryGetTopChunk(string tag, out string chunk);

    // Snapshots (Snapshots partial).
    PlatonicMemorySnapshot ExportSnapshot();
    void ImportSnapshot(PlatonicMemorySnapshot snapshot);

    // Relations (Relations partial).
    void ObserveContradiction(string left, string right, double observedContradiction);
    double GetContradiction(string left, string right);
    IReadOnlyList<RelationElement> GetRelationElements();
    bool TryRelationElementNeighbour(string concept, out string neighbour, out double strength);
    PlatonicSpaceMemory.PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops = 2,
        int beamWidth = 2);
    PlatonicSpaceMemory.PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops,
        int beamWidth,
        out IReadOnlyList<PlatonicEvidence> evidence);
    IReadOnlyList<PlatonicNeighbor> GetNeighbors(
        string concept,
        PlatonicNeighborhoodType type = PlatonicNeighborhoodType.Any,
        int maxNeighbors = 16,
        double minConfidence = 0.0);
    IReadOnlyList<(string Left, string Right, long ObservationCount)> GetAllRelations();
    void ReinforceEvidence(IReadOnlyList<PlatonicEvidence> evidence, bool success);
    int GetRelationDegree(string concept);

    // Geometry (Geometry partial).
    bool TryGetConceptFace(string concept, out double[] positiveFace);
    IReadOnlyList<(string Symbol, double Distance)> GetNearestConcepts(
        string concept,
        IReadOnlyCollection<string>? candidates = null,
        int maxNeighbors = 8,
        int maxCandidates = 96);
    PlatonicSpaceMemory.GeometrySummary SummarizePushPullGeometry(int maxConcepts = 600, int unrelatedPerNode = 8, int seed = 1234);
    IReadOnlyList<(string Symbol, double Distance)> GetNearestConceptsFresh(
        string concept, IReadOnlyCollection<string>? seeds = null, int maxNeighbors = 8);
    double[] ComputeRoutePerception(string anchor, double transformReliability = 0.0);
    void FineEditFromExample(
        IReadOnlyList<string> inputConcepts,
        IReadOnlyList<string> outputConcepts,
        bool isNegativeExample);
    void DisruptAssociation(string anchor, string answer);
    void FunctionGradientStep(string anchor, string target, IReadOnlyList<string> distractors, double rate = 0.05);
    double TotalCharge();

    // Core (PlatonicSpaceMemory.cs).
    bool UseInfoNceRepulsion { get; set; }
    bool DimensionalContradiction { get; set; }   // Phase 1 dialectic: per-dimension agreement/contradiction
    bool GenerativeAtoms { get => false; set { } } // token-as-atom + decompose/recognise (default off; only DialecticalSpace implements)
    bool BatchedCloudGpu { get => false; set { } } // deferred batched-GPU cloud recompute (default off; only DialecticalSpace implements)
    int NodeCount { get; }
    int RelationCount { get; }
    int ArchivedNodeCount { get; }
    int ArchivedRelationCount { get; }
    int FaceDimension { get; }
    int NumericDimensions { get; }
    bool ContainsConcept(string concept);
    void RegisterOperationToken(string token);
    bool IsOperationToken(string concept);

    // Maintenance (Maintenance partial).
    PlatonicSpaceMemory.SpaceMaintenanceResult ApplyMaintenance(PlatonicSpaceMemory.SpaceMaintenanceRequest request);
}
