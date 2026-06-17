using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// PUSH/PULL FORCE BALANCE: the contrastive dynamics (attraction via ObserveContradiction→MessagePassUpdate,
/// repulsion via ApplyContrastiveRepulsionPass) must move RELATED concepts clearly CLOSER than UNRELATED ones
/// in the semantic face — otherwise geometric retrieval has no signal (the space collapses into a cone). This
/// test builds synthetic CLIQUE clusters (intra-cluster pairs attract; cross-cluster pairs are only ever
/// repelled) and asserts a clearly POSITIVE separation. It is the pass/fail gate for tuning the forces.
/// Fast, no NN — just the space dynamics, at production face dimension.
/// </summary>
public sealed class PlatonicGeometryDynamicsTests
{
    private readonly ITestOutputHelper _out;
    public PlatonicGeometryDynamicsTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void PushPull_SeparatesRelatedFromUnrelated()
    {
        var space = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);

        // C clusters, each a CLIQUE of K items. Intra-cluster pairs are observed as AGREEING (contradiction 0
        // → attract). Cross-cluster pairs are never observed → only the repulsion pass touches them.
        const int C = 4, K = 5;
        var clusters = Enumerable.Range(0, C)
            .Select(c => Enumerable.Range(0, K).Select(i => $"c{c}item{i}").ToArray())
            .ToArray();

        for (var epoch = 0; epoch < 40; epoch++)
            foreach (var cl in clusters)
                for (var i = 0; i < K; i++)
                    for (var j = i + 1; j < K; j++)
                        space.ObserveContradiction(cl[i], cl[j], 0.0); // agree → attract

        var g = space.SummarizePushPullGeometry();
        _out.WriteLine($"related(pull) mean {g.RelatedMean:F3}  unrelated(push) mean {g.UnrelatedMean:F3}  separation {g.Separation:F3}");
        _out.WriteLine($"  related [{g.RelatedMin:F3}..{g.RelatedMax:F3}]  unrelated [{g.UnrelatedMin:F3}..{g.UnrelatedMax:F3}]");

        Assert.True(g.RelatedPairs > 0 && g.UnrelatedPairs > 0, "need both related and unrelated pairs sampled");
        Assert.True(g.Separation > 0.15,
            $"push/pull must pull related clearly closer than unrelated; separation={g.Separation:F3} " +
            $"(related {g.RelatedMean:F3}, unrelated {g.UnrelatedMean:F3})");
    }

    [Fact] // STAR/HUB structure — the gym's actual shape (many items → one class hub, item→class edges only).
           // This is what collapsed in the live model; the forces must keep cross-cluster concepts apart.
    public void PushPull_StarHubs_DoNotCollapse()
    {
        var space = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        const int C = 4, K = 6;
        for (var epoch = 0; epoch < 40; epoch++)
            for (var c = 0; c < C; c++)
                for (var i = 0; i < K; i++)
                    space.ObserveContradiction($"c{c}item{i}", $"class{c}", 0.0); // item → its class hub

        var g = space.SummarizePushPullGeometry();
        _out.WriteLine($"[hub] related(pull) mean {g.RelatedMean:F3}  unrelated(push) mean {g.UnrelatedMean:F3}  separation {g.Separation:F3}");
        Assert.True(g.Separation > 0.10,
            $"star-hub clusters must not collapse; separation={g.Separation:F3} (related {g.RelatedMean:F3}, unrelated {g.UnrelatedMean:F3})");
    }

    [Fact] // AT SCALE — mirrors the live gym (hundreds of sparsely-related concepts). With repulsion sampled at
           // a FIXED count and run LESS often than attraction, the space collapses at this size even though the
           // tiny 20-node tests pass. This is the gate that actually reflects the running model.
    public void PushPull_AtScale_ManySparseConcepts_DoNotCollapse()
    {
        var space = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        // ~240 mutable concepts, mostly degree-1 (nonce synonym pairs) + a handful of small clusters — the
        // gym's regime: many concepts, sparse relations. Attraction (every obs) must not out-pull repulsion.
        const int Pairs = 120;
        const int Numbers = 400; // FROZEN-number dilution — mirrors the live gym, where numbers are the majority
                                 // of nodes and used to starve the repulsion sample pool (the real collapse cause).
        // WEAK attraction (contradiction 0.4 → affinity 0.2, BELOW MinAttractAffinity) — mirrors the live state
        // where the edit-head magnitude m is timid. Without the attraction FLOOR, related would not cluster
        // below the (now-effective) repulsion and separation stays flat; the floor is what makes it pass.
        const double weak = 0.4;
        for (var epoch = 0; epoch < 25; epoch++)
        {
            for (var i = 0; i < Pairs; i++)
            {
                space.ObserveContradiction($"eqa{i}", $"eqb{i}", weak);   // synonym (attract), both directions
                space.ObserveContradiction($"eqb{i}", $"eqa{i}", weak);
            }
            // mutable word ↔ FROZEN number (like digit↔word): adds many frozen numbers to the node pool that
            // must NOT dilute the text-vs-text repulsion.
            for (var i = 0; i < Numbers; i++)
                space.ObserveContradiction($"w{i}", $"{1000 + i}", weak);
        }

        var g = space.SummarizePushPullGeometry();
        _out.WriteLine($"[scale] concepts mutable {g.MutableConcepts}  related(pull) {g.RelatedMean:F3}  unrelated(push) {g.UnrelatedMean:F3}  separation {g.Separation:F3}");
        Assert.True(g.Separation > 0.15,
            $"at scale the space must still separate related from unrelated; separation={g.Separation:F3} " +
            $"(related {g.RelatedMean:F3}, unrelated {g.UnrelatedMean:F3}, {g.MutableConcepts} concepts)");
    }
}
