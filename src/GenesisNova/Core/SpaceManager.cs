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
    double MinUtilityToKeep = 0.15);

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
    private static readonly HashSet<string> ProtectedConcepts = new(StringComparer.OrdinalIgnoreCase)
    {
        "+", "-", "*", "/", "x", "add", "sub", "mul", "div", "face:poly", "face:log"
    };

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
        var nodesBefore = snapshot.Nodes.Length;
        var relationsBefore = snapshot.Relations.Length;
        var tuples = snapshot.Relations
            .Select(r => (r.Left, r.Right, ObservationCount: (long)r.ObservationCount))
            .ToArray();
        var quality = TransformQualityMetrics.GenerateReport(tuples);
        var recommendedTool = RecommendTool(nodesBefore, relationsBefore, quality.NoiseRatio);
        return new SpaceManagementResult(
            Compacted: false,
            NodesBefore: nodesBefore,
            NodesAfter: nodesBefore,
            RelationsBefore: relationsBefore,
            RelationsAfter: relationsBefore,
            NodesPruned: 0,
            RelationsPruned: 0,
            NoiseRatio: quality.NoiseRatio,
            RelationBudget: relationsBefore,
            RecommendedTool: recommendedTool);
    }

    private static SpaceToolKind RecommendTool(int nodes, int relations, double noiseRatio)
    {
        if (noiseRatio >= 0.75)
            return SpaceToolKind.Stabilize;

        if (relations > Math.Max(1, nodes * 8))
            return SpaceToolKind.Rebalance;

        if (nodes < 128)
            return SpaceToolKind.Expand;

        if (noiseRatio <= 0.25)
            return SpaceToolKind.Reinforce;

        return SpaceToolKind.Observe;
    }
}
