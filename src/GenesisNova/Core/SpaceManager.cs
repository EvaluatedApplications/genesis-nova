using GenesisNova.Cognition;

namespace GenesisNova.Core;

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
    int MaxRelationPrunesPerCycle = 192,
    int MaxNodePrunesPerCycle = 24,
    int MaxNodeMergesPerCycle = 12,
    int MaxRebalancePrunesPerCycle = 192);

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
    private readonly PlatonicSpaceMemory _memory;
    private readonly SpaceManagerSettings _settings;

    public SpaceManager(PlatonicSpaceMemory memory, SpaceManagerSettings? settings = null)
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

    private int ComputeTargetRelationBudget(int nodes)
    {
        var dynamicBudget = (nodes * Math.Max(1, _settings.TargetRelationsPerNode)) + Math.Max(0, _settings.NodeBuffer);
        return Math.Clamp(dynamicBudget, _settings.MinRelations, _settings.MaxRelations);
    }

}
