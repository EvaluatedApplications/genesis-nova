using GenesisNova.Cognition;
using GenesisNova.Core;

namespace GenesisNova.Tests;

public sealed class SpaceManagerTests
{
    [Fact]
    public void WhenSpaceIsNoisyAndOverBudget_Then_PrunesLowUtilityRelations()
    {
        var memory = new PlatonicSpaceMemory(faceDimension: 16, seed: 7);

        // Keep these as high-utility anchors.
        for (var i = 0; i < 120; i++)
            memory.ObserveContradiction("add", "2", 0.05);

        for (var i = 0; i < 80; i++)
            memory.ObserveContradiction("sub", "1", 0.05);

        // Inject noise: many one-off sparse relations.
        for (var i = 0; i < 140; i++)
            memory.ObserveContradiction($"noise-left-{i}", $"noise-right-{i}", 0.8);

        var manager = new SpaceManager(memory, new SpaceManagerSettings(
            Enabled: true,
            MinNodes: 32,
            MaxNodes: 96,
            MinRelations: 32,
            MaxRelations: 96,
            TargetRelationsPerNode: 2,
            NoiseThreshold: 0.35,
            MinUtilityToKeep: 0.12));

        var result = manager.Manage();

        Assert.True(result.Compacted);
        Assert.True(result.RelationsPruned > 0);
        Assert.True(result.NodesAfter <= 96);
        Assert.True(result.RelationsAfter <= 96);

        var remaining = memory.GetAllRelations();
        Assert.Contains(remaining, r =>
            (r.Left.Equals("add", StringComparison.OrdinalIgnoreCase) && r.Right.Equals("2", StringComparison.OrdinalIgnoreCase)) ||
            (r.Right.Equals("add", StringComparison.OrdinalIgnoreCase) && r.Left.Equals("2", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void WhenSpaceIsWithinLimits_Then_NoCompactionOccurs()
    {
        var memory = new PlatonicSpaceMemory(faceDimension: 12, seed: 11);
        for (var i = 0; i < 20; i++)
            memory.ObserveContradiction("cat", "animal", 0.1);

        var manager = new SpaceManager(memory, new SpaceManagerSettings(
            Enabled: true,
            MinNodes: 16,
            MaxNodes: 512,
            MinRelations: 16,
            MaxRelations: 512,
            TargetRelationsPerNode: 20,
            NoiseThreshold: 0.95));

        var result = manager.Manage();

        Assert.False(result.Compacted);
        Assert.Equal(0, result.RelationsPruned);
        Assert.Equal(0, result.NodesPruned);
    }
}
