using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// M3 — TYPED EDGES (▷ part-of vs κ contradiction kept DISTINCT) + the RECOGNITION HIERARCHY (PLATONIC_THEORY.md §4
/// + §6 recognize). ▷ points DOWN to reused components; κ points SIDEWAYS to meaning — retrieval must never conflate
/// them (the framing-word-hub collapse). Recognition hits the WHOLE first, decomposes on a miss, and composes-and-
/// stores novelty so it is remembered next time. EMPIRICAL probes, production dims.
/// </summary>
public sealed class DialecticalSpaceM3Tests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceM3Tests(ITestOutputHelper o) => _out = o;

    [Fact] // ▷ and κ are distinct and never conflated in retrieval: a word's κ-neighbours are its MEANING relations,
           // never its char atoms; its ▷ components are its char atoms, never its κ relations.
    public void TypedEdges_PartOf_And_Contradiction_NotConflated()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        for (var i = 0; i < 20; i++) space.ObserveContradiction("cat", "animal", 0.0); // κ semantic edge

        var kappa = space.GetNeighbors("cat").Select(n => n.Concept).ToArray();
        var partOf = space.ComponentSymbolsOf("cat");
        _out.WriteLine($"cat κ-neighbours: [{string.Join(",", kappa)}]   ▷ components: [{string.Join(",", partOf)}]");

        Assert.Contains("animal", kappa);                                  // κ surfaces the meaning relation
        Assert.DoesNotContain(kappa, c => c.StartsWith("atom:"));          // ▷ atoms never leak into κ retrieval
        Assert.Equal(new[] { "atom:c", "atom:a", "atom:t" }, partOf);       // ▷ is the reused char components
        Assert.DoesNotContain("animal", partOf);                            // κ never leaks into ▷ structure
    }

    [Fact] // recognize-highest-first: an existing concept is remembered WHOLE (no decomposition); a number via the homomorphism.
    public void Recognize_WholeFirst()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        space.ObserveContradiction("cat", "animal", 0.0);

        var r = space.Recognize("cat");
        _out.WriteLine($"Recognize(cat): wholeHit={r.WholeHit} conf={r.Confidence:F2}");
        Assert.True(r.WholeHit);
        Assert.Equal(1.0, r.Confidence, 6);

        var n = space.Recognize("42");
        Assert.True(n.WholeHit); // numbers recognized via the homomorphism
    }

    [Fact] // miss → decompose → compose-and-store: a novel sentence over KNOWN words is composed (conf 1.0 — all
           // parts known) and STORED, so the SECOND recognition is a whole hit. The accumulating, scaling path.
    public void Recognize_DecomposeComposeStore_ThenWholeHit()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        space.ObserveContradiction("cat", "animal", 0.0);
        space.ObserveContradiction("dog", "animal", 0.0); // "cat","dog" now known words

        var first = space.Recognize("cat dog");   // novel sentence, both words known
        _out.WriteLine($"first Recognize(cat dog): wholeHit={first.WholeHit} known={first.KnownComponents}/{first.TotalComponents} conf={first.Confidence:F2}");
        Assert.False(first.WholeHit);
        Assert.Equal(2, first.KnownComponents);
        Assert.Equal(2, first.TotalComponents);
        Assert.Equal(1.0, first.Confidence, 6);

        var second = space.Recognize("cat dog");   // now stored → remembered whole
        Assert.True(second.WholeHit);
    }

    [Fact] // partial coverage: a sentence with one unknown word recognizes the known part, composes-stores the rest.
    public void Recognize_PartialCoverage()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        space.ObserveContradiction("cat", "animal", 0.0); // only "cat","animal" known

        var r = space.Recognize("cat zzz");  // "cat" known, "zzz" novel
        _out.WriteLine($"Recognize(cat zzz): known={r.KnownComponents}/{r.TotalComponents} conf={r.Confidence:F2}");
        Assert.False(r.WholeHit);
        Assert.Equal(1, r.KnownComponents);
        Assert.Equal(2, r.TotalComponents);
        Assert.True(space.Recognize("cat zzz").WholeHit); // stored after first sight
    }
}
