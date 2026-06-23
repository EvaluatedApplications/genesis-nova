using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// The space's eviction engine (<see cref="PlatonicSpaceMemory.ApplyMaintenance"/>) is now WIRED to the live
/// per-epoch path via <see cref="SpaceManager.Maintain"/> — previously it was implemented but stranded behind
/// the never-called ExecuteTool, so the space never pruned (Manage() hardcoded NodesPruned=0). These tests pin
/// the new behaviour: (1) under relation-pressure the live pass actually EVICTS, while protecting
/// high-observation anchors and respecting the Min floors; (2) a healthy, under-budget space is left untouched.
/// </summary>
public sealed class SpaceMaintenanceWiringTests
{
    private readonly ITestOutputHelper _out;
    public SpaceMaintenanceWiringTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Maintain_UnderPressure_Evicts_ProtectsAnchors_RespectsFloors()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 11);

        // A high-observation ANCHOR relation (obs >> protection threshold, strong = low contradiction): must survive.
        for (var i = 0; i < 40; i++)
        {
            memory.ObserveContradiction("anchorleft", "anchorright", 0.03);
            memory.ObserveContradiction("anchorright", "anchorleft", 0.03);
        }

        // Low-utility JUNK as a sparse RING (each node degree 4, observed once → ObservationCount 4): deliberately
        // BELOW the anchor-protection thresholds (obs<8, degree<16) so it is NOT treated as an anchor, never
        // retrieved (UseCount 0 → utility 0), high contradiction = low strength. Many relations vs a tight budget
        // → over-pressure → the live pass evicts. (A dense clique would make every node look like an anchor.)
        const int n = 40;
        for (var i = 0; i < n; i++)
        {
            memory.ObserveContradiction("jk" + i, "jk" + ((i + 1) % n), 0.92);
            memory.ObserveContradiction("jk" + i, "jk" + ((i + 2) % n), 0.92);
        }

        // Tight budget so the junk clique is clearly over-pressure (default MinRelations=1024 would hide it).
        var settings = new SpaceManagerSettings(
            Enabled: true, MinNodes: 2, MaxNodes: 12_000,
            MinRelations: 8, MaxRelations: 48_000,
            TargetRelationsPerNode: 1, NodeBuffer: 0);
        var manager = new SpaceManager(memory, settings);

        var relationsBefore = memory.ExportSnapshot().Relations.Length;
        var result = manager.Maintain();
        _out.WriteLine($"action={result.RecommendedTool} relPruned={result.RelationsPruned} nodePruned={result.NodesPruned} " +
                       $"compacted={result.Compacted} relBefore={relationsBefore} relAfter={result.RelationsAfter}");

        Assert.True(result.Compacted, "live maintenance should compact a clearly over-pressure space");
        Assert.True(result.RelationsPruned > 0, "expected relations to be evicted under pressure");
        Assert.True(result.RelationsAfter < relationsBefore, "relation count should drop");
        Assert.True(result.NodesAfter >= settings.MinNodes, "must not drop below the node floor");

        // The protected anchor relation must survive eviction.
        Assert.True(memory.TryRelationElementNeighbour("anchorleft", out var nbr, out _), "anchor relation was evicted");
        Assert.Equal("anchorright", nbr, ignoreCase: true);
    }

    [Fact]
    public void Maintain_HealthySpace_EvictsNothing()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 13);
        foreach (var (a, b) in new[] { ("apple", "fruit"), ("dog", "animal"), ("red", "color") })
            for (var i = 0; i < 20; i++)
                memory.ObserveContradiction(a, b, 0.05);

        // Default settings: relation budget floors at 1024 >> the handful of relations here → no pressure → no eviction.
        var manager = new SpaceManager(memory, new SpaceManagerSettings());
        var result = manager.Maintain();

        Assert.False(result.Compacted, "a healthy under-budget space must not be evicted");
        Assert.Equal(0, result.RelationsPruned);
        Assert.Equal(0, result.NodesPruned);
        Assert.True(memory.TryRelationElementNeighbour("apple", out var nbr, out _));
        Assert.Equal("fruit", nbr, ignoreCase: true);
    }
}
