using GenesisNova.Cognition;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// AXIOM G6 (Irreversibility) — "once a distinction has been made it cannot be unmade; the platonic space only
/// expands." Eviction must therefore ARCHIVE (dormant), never DESTROY: active + archived retains every distinction
/// ever made, and a re-observed archived concept reactivates intact. Production face dimension (512).
/// </summary>
public sealed class PlatonicArchivalTests
{
    // maxNodes is floored at 256 by the ctor, so create well past that to force capacity eviction.
    private static PlatonicSpaceMemory OverflowedSpace(int pairs)
    {
        var m = new PlatonicSpaceMemory(faceDimension: 512, seed: 7, maxNodes: 6, maxRelations: 100_000);
        for (var i = 0; i < pairs; i++)
            m.ObserveContradiction($"c{i}", $"d{i}", 0.1);
        return m;
    }

    [Fact]
    public void Eviction_Archives_NeverDestroys_NoDistinctionLost()
    {
        const int pairs = 200;            // 400 distinct concepts, well over the 256 active cap
        var m = OverflowedSpace(pairs);

        Assert.True(m.NodeCount <= 256, $"active space must stay within the cap (was {m.NodeCount})");
        Assert.True(m.ArchivedNodeCount > 0, "evicted nodes must be ARCHIVED, not destroyed (G6)");
        Assert.Equal(pairs * 2, m.NodeCount + m.ArchivedNodeCount); // every distinction retained (active or dormant)
    }

    [Fact]
    public void ReObservingArchivedConcept_ReactivatesItIntact()
    {
        const int pairs = 200;
        var m = OverflowedSpace(pairs);

        // Find a concept that was evicted (no longer active → it's in the archive).
        string? evicted = null;
        for (var i = 0; i < pairs && evicted is null; i++)
            if (!m.ContainsConcept($"c{i}")) evicted = $"c{i}";
        Assert.NotNull(evicted); // capacity pressure must have archived at least one

        var totalBefore = m.NodeCount + m.ArchivedNodeCount;
        m.ObserveContradiction(evicted!, "reactivation-probe", 0.1); // re-observe → reactivate from archive

        Assert.True(m.ContainsConcept(evicted!), "a re-observed archived concept must be reactivated (active again)");
        // Reactivation moves the concept from archive to active; the probe adds one new distinction.
        Assert.Equal(totalBefore + 1, m.NodeCount + m.ArchivedNodeCount); // still nothing destroyed
    }
}
