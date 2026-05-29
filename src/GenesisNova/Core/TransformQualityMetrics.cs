using System.Collections.Immutable;

namespace GenesisNova.Core;

/// <summary>
/// Analyzes quality of learned transforms and relations to drive high-quality learning.
/// 
/// Key metrics:
/// - Utility: based on observation count (frequent = good) and entropy (concentrated = good)
/// - Generalization: can this transform explain multiple different inputs?
/// - Noise ratio: what % of relations are sparse/low-quality?
/// 
/// These metrics feed back into training loss to:
/// - Penalize low-utility relations
/// - Reward high-generalization transforms
/// - Drive convergence to clean, abstract operators (e.g., single add transform)
/// </summary>
public static class TransformQualityMetrics
{
    /// <summary>
    /// Compute utility score for a relation [0..1].
    /// Higher = more useful.
    /// 
    /// Utility balances:
    /// - Observation count (more examples = higher utility)
    /// - Concentration (is the relation used consistently, or scattered?)
    /// </summary>
    public static double ComputeRelationUtility(
        (string Left, string Right, long ObservationCount) relation,
        IReadOnlyCollection<(string Left, string Right, long ObservationCount)> allRelations)
    {
        if (relation.ObservationCount == 0)
            return 0.0;

        // Normalize observation count: 0..1 where 1 = top relation count
        var maxCount = allRelations.Max(r => r.ObservationCount);
        var countScore = maxCount > 0 ? relation.ObservationCount / (double)maxCount : 0.0;

        // Penalize very rare relations (outliers)
        var threshold = allRelations.Average(r => r.ObservationCount) / 2.0;
        var rarity_penalty = relation.ObservationCount < threshold ? 0.3 : 1.0;

        // Higher count is good, but diminishing returns after ~1000 obs
        var saturated = Math.Min(1.0, relation.ObservationCount / 1000.0);
        
        // Utility = (normalized count) * (rarity adjustment) * (saturation curve)
        return countScore * rarity_penalty * saturated;
    }

    /// <summary>
    /// Compute entropy of targets for a given source operation.
    /// Higher entropy = scattered, low-quality learning.
    /// Lower entropy = concentrated, good generalization.
    /// 
    /// Example:
    /// - "add" → {2, 3, 5, 7, 8, 9, ...} = high entropy = BAD (scattered targets)
    /// - "add" → {specific numeric face} = low entropy = GOOD (concentrated)
    /// </summary>
    public static double ComputeOperationEntropy(
        string operation,
        IReadOnlyCollection<(string Left, string Right, long ObservationCount)> allRelations)
    {
        var targetsForOp = allRelations
            .Where(r => r.Left == operation)
            .GroupBy(r => r.Right)
            .ToList();

        if (targetsForOp.Count <= 1)
            return 0.0;  // Perfect concentration

        // Shannon entropy: -Σ(p_i * log(p_i))
        var totalCount = (double)targetsForOp.Sum(g => g.Sum(r => r.ObservationCount));
        var entropy = 0.0;

        foreach (var group in targetsForOp)
        {
            var groupCount = (double)group.Sum(r => r.ObservationCount);
            var p = groupCount / totalCount;
            if (p > 0)
                entropy -= p * Math.Log2(p);
        }

        // Normalize: max entropy for N targets = log2(N)
        var maxEntropy = Math.Log2(targetsForOp.Count);
        return maxEntropy > 0 ? entropy / maxEntropy : 0.0;  // [0..1]
    }

    /// <summary>
    /// Compute noise ratio: what % of relations are sparse/low-quality?
    /// 
    /// A relation is considered "noisy" if:
    /// - ObservationCount < 10 (very rare)
    /// - OR it's in the bottom 50% of observation counts AND has high entropy
    /// </summary>
    public static double ComputeNoiseRatio(
        IReadOnlyCollection<(string Left, string Right, long ObservationCount)> allRelations)
    {
        if (allRelations.Count == 0)
            return 0.0;

        var medianCount = allRelations
            .OrderBy(r => r.ObservationCount)
            .ElementAt(allRelations.Count / 2)
            .ObservationCount;

        var noisyCount = 0;
        foreach (var rel in allRelations)
        {
            var isVeryRare = rel.ObservationCount < 10;
            var isBelow_median = rel.ObservationCount < medianCount;
            
            if (isVeryRare || isBelow_median)
            {
                noisyCount++;
            }
        }

        return (double)noisyCount / allRelations.Count;
    }

    /// <summary>
    /// Compute generalization score: how well can this relation explain new examples?
    /// 
    /// A relation is generalizable if:
    /// - It has high observation count (seen many times = likely captures real pattern)
    /// - Its targets are semantically related (arithmetic: numeric-to-numeric, not scattered)
    /// 
    /// Returns [0..1] where 1 = perfect generalization, 0 = no generalization.
    /// </summary>
    public static double ComputeGeneralizationScore(
        string sourceOperation,
        IReadOnlyCollection<(string Left, string Right, long ObservationCount)> allRelations,
        IReadOnlyCollection<(string Target, int Count)> targetTypes)
    {
        // Find all relations from this source
        var relationsFromSource = allRelations
            .Where(r => r.Left == sourceOperation)
            .ToList();

        if (relationsFromSource.Count == 0)
            return 0.0;

        var totalObs = (double)relationsFromSource.Sum(r => r.ObservationCount);

        // Check: does this operation consistently target one target type?
        // Example: "add" should mostly target numeric nodes
        var targetCounts = relationsFromSource
            .GroupBy(r => ClassifyTarget(r.Right))
            .Select(g => (Type: g.Key, Count: (double)g.Sum(r => r.ObservationCount)))
            .OrderByDescending(x => x.Count)
            .ToList();

        if (targetCounts.Count == 0)
            return 0.0;

        // If top target type accounts for >80% of observations, it's well-generalized
        var topTargetRatio = targetCounts[0].Count / totalObs;
        var concentration = Math.Min(1.0, topTargetRatio * 1.25);  // Reward >80%

        // Also reward high total observation count (seen many times = reliable)
        var frequency_reward = Math.Min(1.0, totalObs / 500.0);

        return concentration * 0.7 + frequency_reward * 0.3;
    }

    /// <summary>
    /// Classify a target symbol (numeric, face, operation, etc).
    /// </summary>
    private static string ClassifyTarget(string target)
    {
        if (target.StartsWith("face:"))
            return "face";
        if (int.TryParse(target, out _))
            return "numeric";
        if (target is "add" or "sub" or "mul" or "div" or "+" or "-" or "*" or "/")
            return "operation";
        return "other";
    }

    /// <summary>
    /// Compute quality-adjusted loss term for a newly learned relation.
    /// 
    /// If the relation is high-quality, loss penalty is low (model is rewarded).
    /// If the relation is low-quality/noisy, loss penalty is high (model is penalized).
    /// </summary>
    public static double ComputeQualityLossPenalty(
        string operation,
        string target,
        long relationObservationCount,
        IReadOnlyCollection<(string Left, string Right, long ObservationCount)> allRelations,
        double utilityThreshold = 0.3)
    {
        var relation = (Left: operation, Right: target, ObservationCount: relationObservationCount);
        var utility = ComputeRelationUtility(relation, allRelations);
        var entropy = ComputeOperationEntropy(operation, allRelations);
        var noiseRatio = ComputeNoiseRatio(allRelations);

        // Quality loss combines:
        // 1. Low utility penalty: (1 - utility)
        // 2. High entropy penalty: entropy * 0.5
        // 3. High noise ratio penalty: noiseRatio * 0.2

        var qualityLoss = (1.0 - Math.Max(utility, utilityThreshold)) * 0.6 +
                         entropy * 0.3 +
                         noiseRatio * 0.1;

        return Math.Clamp(qualityLoss, 0.0, 1.0);
    }

    /// <summary>
    /// Generate a quality report for the current state of learned relations.
    /// Useful for debugging and understanding what the model is learning.
    /// </summary>
    public static QualityReport GenerateReport(
        IReadOnlyCollection<(string Left, string Right, long ObservationCount)> allRelations)
    {
        if (allRelations.Count == 0)
            return new(
                TotalRelations: 0,
                AvgUtility: 0.0,
                NoiseRatio: 0.0,
                TopOperations: [],
                LowestUtilityRelations: []);

        var utilities = allRelations
            .Select(r => (Relation: r, Utility: ComputeRelationUtility(r, allRelations)))
            .OrderByDescending(x => x.Utility)
            .ToList();

        var avgUtility = utilities.Count > 0 ? utilities.Average(x => x.Utility) : 0.0;
        var noiseRatio = ComputeNoiseRatio(allRelations);

        var topOps = allRelations
            .GroupBy(r => r.Left)
            .Select(g => (Operation: g.Key, Count: g.Count(), AvgObs: g.Average(r => r.ObservationCount)))
            .OrderByDescending(x => x.Count)
            .Take(10)
            .ToList();

        var lowestUtility = utilities
            .Where(x => x.Utility < 0.5)
            .Take(20)
            .Select(x => $"{x.Relation.Left}->{x.Relation.Right} ({x.Relation.ObservationCount} obs, U={x.Utility:F3})")
            .ToList();

        return new(
            TotalRelations: allRelations.Count,
            AvgUtility: avgUtility,
            NoiseRatio: noiseRatio,
            TopOperations: topOps.Select(x => $"{x.Operation}: {x.Count} targets, avg {x.AvgObs:F1} obs").ToList(),
            LowestUtilityRelations: lowestUtility);
    }
}

/// <summary>
/// Quality report snapshot for analysis.
/// </summary>
public record QualityReport(
    int TotalRelations,
    double AvgUtility,
    double NoiseRatio,
    IReadOnlyList<string> TopOperations,
    IReadOnlyList<string> LowestUtilityRelations);
