using GenesisNova.Cognition;

namespace GenesisNova.Core;

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
    int RelationBudget);

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
        if (!_settings.Enabled || relationsBefore == 0)
        {
            return new SpaceManagementResult(
                Compacted: false,
                NodesBefore: nodesBefore,
                NodesAfter: nodesBefore,
                RelationsBefore: relationsBefore,
                RelationsAfter: relationsBefore,
                NodesPruned: 0,
                RelationsPruned: 0,
                NoiseRatio: 0.0,
                RelationBudget: relationsBefore);
        }

        var tuples = snapshot.Relations
            .Select(r => (r.Left, r.Right, ObservationCount: (long)r.ObservationCount))
            .ToArray();
        var quality = TransformQualityMetrics.GenerateReport(tuples);
        var relationBudget = Math.Clamp(
            Math.Max(_settings.MinRelations, nodesBefore * Math.Max(1, _settings.TargetRelationsPerNode)),
            _settings.MinRelations,
            Math.Max(_settings.MinRelations, _settings.MaxRelations));

        var shouldCompact = relationsBefore > relationBudget ||
                            nodesBefore > _settings.MaxNodes ||
                            quality.NoiseRatio >= _settings.NoiseThreshold;
        if (!shouldCompact)
        {
            return new SpaceManagementResult(
                Compacted: false,
                NodesBefore: nodesBefore,
                NodesAfter: nodesBefore,
                RelationsBefore: relationsBefore,
                RelationsAfter: relationsBefore,
                NodesPruned: 0,
                RelationsPruned: 0,
                NoiseRatio: quality.NoiseRatio,
                RelationBudget: relationBudget);
        }

        var maxObs = Math.Max(1, snapshot.Relations.Max(r => r.ObservationCount));
        var ranked = snapshot.Relations
            .Select(r =>
            {
                var utility = TransformQualityMetrics.ComputeRelationUtility(
                    (r.Left, r.Right, (long)r.ObservationCount), tuples);
                var obsScore = r.ObservationCount / (double)maxObs;
                var contradictionScore = 1.0 - Clamp01(r.SynthesisContradiction);
                var score = (utility * 0.65) + (obsScore * 0.25) + (contradictionScore * 0.10);
                return new RankedRelation(r, utility, score);
            })
            .OrderByDescending(x => x.Score)
            .ThenByDescending(x => x.Relation.ObservationCount)
            .ToArray();

        var keepCount = Math.Clamp(relationBudget, 1, ranked.Length);
        var guaranteed = ranked
            .Where(x => x.Utility >= _settings.MinUtilityToKeep)
            .Take(keepCount)
            .Select(x => x.Relation)
            .ToList();
        if (guaranteed.Count < keepCount)
        {
            var fill = ranked
                .Select(x => x.Relation)
                .Where(r => !guaranteed.Contains(r))
                .Take(keepCount - guaranteed.Count);
            guaranteed.AddRange(fill);
        }

        var referencedConcepts = new HashSet<string>(
            guaranteed.SelectMany(r => new[] { r.Left, r.Right }),
            StringComparer.OrdinalIgnoreCase);
        foreach (var concept in snapshot.Nodes.Where(n => IsProtectedConcept(n.Name)).Select(n => n.Name))
            referencedConcepts.Add(concept);

        var nodeBudget = Math.Clamp(
            Math.Max(_settings.MinNodes, referencedConcepts.Count + Math.Max(0, _settings.NodeBuffer)),
            _settings.MinNodes,
            Math.Max(_settings.MinNodes, _settings.MaxNodes));

        var keptNodes = snapshot.Nodes
            .Where(n => referencedConcepts.Contains(n.Name))
            .Concat(snapshot.Nodes
                .Where(n => !referencedConcepts.Contains(n.Name))
                .OrderByDescending(n => n.ObservationCount))
            .Take(nodeBudget)
            .ToArray();

        var keptNodeNames = new HashSet<string>(keptNodes.Select(n => n.Name), StringComparer.OrdinalIgnoreCase);
        var keptRelations = guaranteed
            .Where(r => keptNodeNames.Contains(r.Left) && keptNodeNames.Contains(r.Right))
            .ToArray();

        if (keptNodes.Length == nodesBefore && keptRelations.Length == relationsBefore)
        {
            return new SpaceManagementResult(
                Compacted: false,
                NodesBefore: nodesBefore,
                NodesAfter: nodesBefore,
                RelationsBefore: relationsBefore,
                RelationsAfter: relationsBefore,
                NodesPruned: 0,
                RelationsPruned: 0,
                NoiseRatio: quality.NoiseRatio,
                RelationBudget: relationBudget);
        }

        _memory.ImportSnapshot(new PlatonicMemorySnapshot(
            FaceDimension: snapshot.FaceDimension,
            Nodes: keptNodes,
            Relations: keptRelations));

        var nodesAfter = keptNodes.Length;
        var relationsAfter = keptRelations.Length;
        return new SpaceManagementResult(
            Compacted: true,
            NodesBefore: nodesBefore,
            NodesAfter: nodesAfter,
            RelationsBefore: relationsBefore,
            RelationsAfter: relationsAfter,
            NodesPruned: Math.Max(0, nodesBefore - nodesAfter),
            RelationsPruned: Math.Max(0, relationsBefore - relationsAfter),
            NoiseRatio: quality.NoiseRatio,
            RelationBudget: relationBudget);
    }

    private static bool IsProtectedConcept(string concept)
        => ProtectedConcepts.Contains(concept);

    private static double Clamp01(double value)
        => Math.Max(0.0, Math.Min(1.0, value));

    private readonly record struct RankedRelation(
        PlatonicRelationSnapshot Relation,
        double Utility,
        double Score);
}
