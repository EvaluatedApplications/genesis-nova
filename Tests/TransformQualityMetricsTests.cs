using GenesisNova.Core;
using System.Text.Json;

namespace GenesisNova.Tests;

/// <summary>
/// Tests the quality metrics system to verify it can identify good vs bad learning.
/// Also demonstrates how quality signals drive better convergence to clean transforms.
/// </summary>
public sealed class TransformQualityMetricsTests
{
    [Fact]
    public void WhenRelationIsHighUtility_ThenUtilityScoreIsHigh()
    {
        // Arrange: a high-observation relation
        var relation = ("add", "3", ObservationCount: 500L);
        var allRelations = new List<(string, string, long)>
        {
            ("add", "1", 100L),
            ("add", "2", 200L),
            ("add", "3", 500L),  // High count
            ("add", "4", 150L),
        };

        // Act
        var utility = TransformQualityMetrics.ComputeRelationUtility(relation, allRelations);

        // Assert
        // Utility is normalized by max (500) and adjusted for rarity
        // With this data, utility should be in range [0.3, 0.8]
        Assert.True(utility > 0.3, $"High-observation relation should have reasonable utility, got {utility}");
    }

    [Fact]
    public void WhenRelationIsVeryRare_ThenUtilityScoreIsLow()
    {
        // Arrange: a rare relation
        var relation = ("mul", "xyz", ObservationCount: 1L);
        var allRelations = new List<(string, string, long)>
        {
            ("add", "1", 500L),
            ("add", "2", 480L),
            ("mul", "xyz", 1L),  // Very rare
        };

        // Act
        var utility = TransformQualityMetrics.ComputeRelationUtility(relation, allRelations);

        // Assert
        Assert.True(utility < 0.3, $"Very rare relation should have low utility, got {utility}");
    }

    [Fact]
    public void WhenOperationHasConcentratedTargets_ThenEntropyIsLow()
    {
        // Arrange: "add" mostly targets numeric nodes (concentrated)
        var allRelations = new List<(string, string, long)>
        {
            ("add", "1", 200L),
            ("add", "2", 180L),
            ("add", "3", 190L),
            ("add", "4", 210L),
            // vs random noise:
            ("add", "foo", 5L),
            ("add", "bar", 3L),
        };

        // Act
        var entropy = TransformQualityMetrics.ComputeOperationEntropy("add", allRelations);

        // Assert
        // Most observations (780/803 ≈ 97%) go to numeric targets
        // With 6 different targets, max entropy = log2(6) = 2.58
        // Actual entropy will be lower due to concentration
        Assert.True(entropy > 0.0, $"Should compute non-zero entropy, got {entropy}");
        Assert.True(entropy < 1.0, $"Concentrated operation should have entropy <1.0, got {entropy}");
    }

    [Fact]
    public void WhenOperationHasScatteredTargets_ThenEntropyIsHigh()
    {
        // Arrange: operation scattered across many different targets
        var allRelations = new List<(string, string, long)>
        {
            ("junk", "A", 10L),
            ("junk", "B", 12L),
            ("junk", "C", 11L),
            ("junk", "D", 9L),
            ("junk", "E", 13L),
            // All roughly equal = high entropy
        };

        // Act
        var entropy = TransformQualityMetrics.ComputeOperationEntropy("junk", allRelations);

        // Assert
        Assert.True(entropy > 0.8, $"Scattered operation should have high entropy, got {entropy}");
    }

    [Fact]
    public void WhenManyRelationsAreSparse_ThenNoiseRatioIsHigh()
    {
        // Arrange: mostly sparse relations
        var allRelations = new List<(string, string, long)>
        {
            ("op1", "t1", 500L),  // Good
            ("op2", "t2", 3L),    // Sparse
            ("op3", "t3", 2L),    // Sparse
            ("op4", "t4", 1L),    // Sparse
            ("op5", "t5", 2L),    // Sparse
        };

        // Act
        var noiseRatio = TransformQualityMetrics.ComputeNoiseRatio(allRelations);

        // Assert
        // 4 out of 5 relations are sparse (~80%)
        Assert.True(noiseRatio > 0.7, $"Should detect high noise ratio, got {noiseRatio}");
    }

    [Fact]
    public void WhenOperationIsGeneralizable_ThenGeneralizationScoreIsHigh()
    {
        // Arrange: "add" mostly targets numeric nodes consistently
        var allRelations = new List<(string, string, long)>
        {
            ("add", "1", 300L),
            ("add", "2", 280L),
            ("add", "3", 290L),
            ("add", "4", 310L),
            ("add", "5", 320L),  // All numeric
            ("add", "face:poly", 50L),  // Slight deviation
        };
        var targetTypes = new List<(string, int)>
        {
            ("numeric", 1500),
            ("face", 50),
        };

        // Act
        var score = TransformQualityMetrics.ComputeGeneralizationScore("add", allRelations, targetTypes);

        // Assert
        Assert.True(score > 0.5, $"Well-generalized operation should score >0.5, got {score}");
    }

    [Fact]
    public void WhenCheckpointIsLoaded_ThenQualityReportShowsMetrics()
    {
        // Arrange
        var checkpointPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova", "state", "genesis-nova.autosave.checkpoint.json");

        if (!File.Exists(checkpointPath))
        {
            Console.WriteLine("[SKIPPED] Checkpoint not found");
            return;
        }

        var json = JsonDocument.Parse(File.ReadAllText(checkpointPath));
        var root = json.RootElement;
        var relationsElem = root.GetProperty("PlatonicSpace").GetProperty("Relations");

        // Extract relations from checkpoint
        var relations = new List<(string, string, long)>();
        foreach (var rel in relationsElem.EnumerateArray())
        {
            var left = rel.GetProperty("Left").GetString() ?? "";
            var right = rel.GetProperty("Right").GetString() ?? "";
            var obs = rel.GetProperty("ObservationCount").GetInt64();
            relations.Add((left, right, obs));
        }

        if (relations.Count == 0)
        {
            Console.WriteLine("[SKIPPED] Checkpoint has no relations");
            return;
        }

        // Act
        var report = TransformQualityMetrics.GenerateReport(relations);

        // Assert & Report
        Console.WriteLine("\n=== QUALITY METRICS REPORT ===");
        Console.WriteLine($"Total relations: {report.TotalRelations}");
        Console.WriteLine($"Average utility: {report.AvgUtility:F3} (target: >0.5)");
        Console.WriteLine($"Noise ratio: {report.NoiseRatio:F3} (target: <0.4)");
        
        Console.WriteLine($"\nTop operations:");
        foreach (var op in report.TopOperations.Take(5))
        {
            Console.WriteLine($"  {op}");
        }
        
        if (report.LowestUtilityRelations.Count > 0)
        {
            Console.WriteLine($"\nLowest utility relations (first 5):");
            foreach (var rel in report.LowestUtilityRelations.Take(5))
            {
                Console.WriteLine($"  {rel}");
            }
        }

        Console.WriteLine($"\n=== INTERPRETATION ===");
        if (report.AvgUtility > 0.5 && report.NoiseRatio < 0.4)
        {
            Console.WriteLine("✓ Good learning: high-utility relations dominate");
        }
        else if (report.NoiseRatio > 0.6)
        {
            Console.WriteLine("✗ Poor learning: too many sparse/noisy relations");
            Console.WriteLine("  Action: Increase training on high-value examples");
        }
        else
        {
            Console.WriteLine("~ Moderate learning: some noise present");
            Console.WriteLine("  Action: Apply quality penalty to sparse relations");
        }

        Assert.True(report.TotalRelations > 0, "Should have learned some relations");
    }

    [Fact]
    public void WhenQualityLossIsComputed_ThenLowUtilityRelationsHavePenalty()
    {
        // Arrange
        var allRelations = new List<(string, string, long)>
        {
            ("add", "1", 500L),
            ("add", "2", 480L),
            ("add", "xyz_noise", 2L),  // Low utility target
        };

        // Act: compute penalty for the low-utility relation
        var penalty = TransformQualityMetrics.ComputeQualityLossPenalty(
            operation: "add",
            target: "xyz_noise",
            relationObservationCount: 1L,
            allRelations: allRelations,
            utilityThreshold: 0.3);

        // Assert
        Assert.True(penalty > 0.3, $"Low-utility relation should have penalty >0.3, got {penalty}");
    }

    [Fact]
    public void WhenQualityLossIsComputed_ThenHighUtilityRelationsHaveLowPenalty()
    {
        // Arrange: good concentration of high-value relations
        var allRelations = new List<(string, string, long)>
        {
            ("add", "1", 1000L),  // High utility
            ("add", "2", 980L),   // High utility
            ("add", "3", 1010L),  // High utility (matches others)
        };

        // Act: compute penalty for a high-utility relation
        var penalty = TransformQualityMetrics.ComputeQualityLossPenalty(
            operation: "add",
            target: "3",
            relationObservationCount: 1010L,
            allRelations: allRelations,
            utilityThreshold: 0.3);

        // Assert
        // With concentrated high-obs relations, penalty should be moderate
        Assert.True(penalty < 0.5, $"High-utility relation should have lower penalty, got {penalty}");
    }
}
