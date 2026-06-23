using GenesisNova.Core;

namespace GenesisNova.Cognition;

public enum SpaceToolKind
{
    Observe = 1,
    Stabilize = 2,
    Expand = 3,
    Rebalance = 4,
    Reinforce = 5
}

public sealed record SpaceManagerSettings(
    bool Enabled = true,
    int MinNodes = 256,
    int MaxNodes = 12_000,
    int MinRelations = 1_024,
    int MaxRelations = 48_000,
    int TargetRelationsPerNode = 6,
    int NodeBuffer = 128,
    double NoiseThreshold = 0.65,
    double MinUtilityToKeep = 0.15,
    int MaxRelationPrunesPerCycle = 48,
    int MaxNodePrunesPerCycle = 6,
    int MaxNodeMergesPerCycle = 3,
    int MaxRebalancePrunesPerCycle = 48);

public sealed record SpaceManagementResult(
    bool Compacted,
    int NodesBefore,
    int NodesAfter,
    int RelationsBefore,
    int RelationsAfter,
    int NodesPruned,
    int RelationsPruned,
    double NoiseRatio,
    int RelationBudget,
    SpaceToolKind RecommendedTool = SpaceToolKind.Observe);

public sealed class SpaceManager
{
    private const int AnchorObservationProtectionThreshold = 8;
    private const int AnchorDegreeProtectionThreshold = 16;

    private readonly IPlatonicSpace _memory;
    private readonly SpaceManagerSettings _settings;

    public SpaceManager(IPlatonicSpace memory, SpaceManagerSettings? settings = null)
    {
        _memory = memory;
        _settings = settings ?? new SpaceManagerSettings();
    }

    public SpaceManagementResult Manage()
    {
        var snapshot = _memory.ExportSnapshot();
        var nodes = snapshot.Nodes.Length;
        var relations = snapshot.Relations.Length;
        var relationTuples = snapshot.Relations
            .Select(r => (r.Left, r.Right, ObservationCount: (long)r.ObservationCount))
            .ToArray();
        var quality = TransformQualityMetrics.GenerateReport(relationTuples);
        var relationBudget = ComputeTargetRelationBudget(nodes);

        return new SpaceManagementResult(
            Compacted: false,
            NodesBefore: nodes,
            NodesAfter: nodes,
            RelationsBefore: relations,
            RelationsAfter: relations,
            NodesPruned: 0,
            RelationsPruned: 0,
            NoiseRatio: quality.NoiseRatio,
            RelationBudget: relationBudget,
            RecommendedTool: SpaceToolKind.Observe);
    }

    /// <summary>
    /// ACTIVE maintenance pass — the LIVE eviction path, called once per training epoch via
    /// <see cref="GenesisNova.Train.GenesisTrainer.ManagePlatonicSpace"/>. It assesses the space and, when it is
    /// noisy or over its relation budget (<see cref="SelectBaselineTool"/> → Stabilize/Rebalance), runs the
    /// <see cref="PlatonicSpaceMemory.ApplyMaintenance"/> prune/merge/rebalance engine — protecting
    /// high-observation / high-degree ANCHORS (<see cref="BuildProtectedConcepts"/>) and never dropping below the
    /// Min floors — then returns the REAL pruned counts. A healthy, under-budget space is left untouched (the
    /// growth-oriented Expand/Reinforce recommendations are a no-op here; this pass only EVICTS). <see
    /// cref="Manage"/> stays a READ-ONLY assessment for diagnostics (GenesisInspect).
    /// </summary>
    public SpaceManagementResult Maintain(IReadOnlyCollection<string>? focusConcepts = null)
    {
        var before = Manage();
        if (!_settings.Enabled)
            return before;

        var action = SelectBaselineTool(before);
        if (action != SpaceToolKind.Stabilize && action != SpaceToolKind.Rebalance)
            return before with { RecommendedTool = action }; // under pressure threshold → observe only, no eviction

        var protectedConcepts = BuildProtectedConcepts(focusConcepts);
        var maintenance = _memory.ApplyMaintenance(action == SpaceToolKind.Stabilize
            ? BuildStabilizeRequest(before, protectedConcepts)
            : BuildRebalanceRequest(before, protectedConcepts));

        var after = Manage();
        var changed = maintenance.RelationsPruned > 0 || maintenance.NodesPruned > 0 || maintenance.NodesMerged > 0;
        return new SpaceManagementResult(
            Compacted: changed,
            NodesBefore: before.NodesAfter,
            NodesAfter: after.NodesAfter,
            RelationsBefore: before.RelationsAfter,
            RelationsAfter: after.RelationsAfter,
            NodesPruned: maintenance.NodesPruned,
            RelationsPruned: maintenance.RelationsPruned,
            NoiseRatio: after.NoiseRatio,
            RelationBudget: after.RelationBudget,
            RecommendedTool: action);
    }

    private SpaceToolKind SelectBaselineTool(SpaceManagementResult assessment)
    {
        var relationPressure = assessment.RelationBudget > 0
            ? (double)assessment.RelationsAfter / assessment.RelationBudget
            : 0.0;

        if (assessment.NoiseRatio >= Math.Max(0.55, _settings.NoiseThreshold * 0.9))
            return SpaceToolKind.Stabilize;

        if (relationPressure >= 1.12)
            return SpaceToolKind.Rebalance;

        if (relationPressure <= 0.70 && assessment.NoiseRatio <= 0.42)
            return SpaceToolKind.Expand;

        if (assessment.NoiseRatio <= 0.50)
            return SpaceToolKind.Reinforce;

        return SpaceToolKind.Observe;
    }

    // The two maintenance request shapes the live Maintain() pass uses (Stabilize = noisy, Rebalance = over
    // relation budget). Both protect anchors and floor at MinNodes/MinRelations; they differ in how aggressively
    // they cap per-node degree and the contradiction ceiling kept.
    private PlatonicSpaceMemory.SpaceMaintenanceRequest BuildStabilizeRequest(
        SpaceManagementResult assessment, IReadOnlyCollection<string> protectedConcepts)
        => new(
            MaxRelationPrunes: _settings.MaxRelationPrunesPerCycle,
            MaxNodePrunes: _settings.MaxNodePrunesPerCycle,
            MaxNodeMerges: _settings.MaxNodeMergesPerCycle,
            MaxRebalancePrunes: Math.Max(1, _settings.MaxRebalancePrunesPerCycle / 2),
            MaxRelationsPerNode: Math.Max(2, _settings.TargetRelationsPerNode + 2),
            TargetRelationCount: assessment.RelationBudget,
            MinRelationsToKeep: Math.Max(_settings.MinRelations, assessment.RelationBudget / 2),
            MinNodesToKeep: _settings.MinNodes,
            MaxSynthesisContradictionToKeep: 0.70,
            ProtectedConcepts: protectedConcepts);

    private PlatonicSpaceMemory.SpaceMaintenanceRequest BuildRebalanceRequest(
        SpaceManagementResult assessment, IReadOnlyCollection<string> protectedConcepts)
        => new(
            MaxRelationPrunes: Math.Max(1, _settings.MaxRelationPrunesPerCycle / 2),
            MaxNodePrunes: Math.Max(1, _settings.MaxNodePrunesPerCycle / 3),
            MaxNodeMerges: Math.Max(1, _settings.MaxNodeMergesPerCycle / 2),
            MaxRebalancePrunes: _settings.MaxRebalancePrunesPerCycle,
            MaxRelationsPerNode: Math.Max(1, _settings.TargetRelationsPerNode),
            TargetRelationCount: assessment.RelationBudget,
            MinRelationsToKeep: Math.Max(_settings.MinRelations, assessment.RelationBudget / 2),
            MinNodesToKeep: _settings.MinNodes,
            MaxSynthesisContradictionToKeep: 0.78,
            ProtectedConcepts: protectedConcepts);

    private int ComputeTargetRelationBudget(int nodes)
    {
        var dynamicBudget = (nodes * Math.Max(1, _settings.TargetRelationsPerNode)) + Math.Max(0, _settings.NodeBuffer);
        return Math.Clamp(dynamicBudget, _settings.MinRelations, _settings.MaxRelations);
    }

    private IReadOnlyCollection<string> BuildProtectedConcepts(IReadOnlyCollection<string>? focusConcepts)
    {
        var protectedConcepts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (focusConcepts is not null)
        {
            foreach (var concept in focusConcepts)
            {
                if (!string.IsNullOrWhiteSpace(concept))
                    protectedConcepts.Add(concept.Trim());
            }
        }

        var snapshot = _memory.ExportSnapshot();
        var degreeByConcept = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var relation in snapshot.Relations)
        {
            degreeByConcept[relation.Left] = degreeByConcept.TryGetValue(relation.Left, out var leftDegree) ? leftDegree + 1 : 1;
            degreeByConcept[relation.Right] = degreeByConcept.TryGetValue(relation.Right, out var rightDegree) ? rightDegree + 1 : 1;

            if (relation.ObservationCount >= AnchorObservationProtectionThreshold)
            {
                protectedConcepts.Add(relation.Left);
                protectedConcepts.Add(relation.Right);
            }
        }

        foreach (var node in snapshot.Nodes)
        {
            var degree = degreeByConcept.TryGetValue(node.Name, out var value) ? value : 0;
            if (node.ObservationCount >= AnchorObservationProtectionThreshold ||
                degree >= AnchorDegreeProtectionThreshold)
            {
                protectedConcepts.Add(node.Name);
            }
        }

        return protectedConcepts.ToArray();
    }
}
