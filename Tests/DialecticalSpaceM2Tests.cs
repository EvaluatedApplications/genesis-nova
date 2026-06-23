using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// M2 — EMERGENT POSITIONING (PLATONIC_THEORY.md Law D, Law M). A concept's place is NOT stamped; it SETTLES as the
/// synthesis of its per-aspect agreements (pull) and contradictions (the bounded hinge spread). Two probes prove it
/// empirically: (1) a concept settles near its relational CONTEXT and far from a disjoint one — meaning is
/// differential (Law M); (2) under LONG training the separation stays stable and does NOT erode or invert — the
/// failure mode that plagued the old store. Production dims, no NN.
/// </summary>
public sealed class DialecticalSpaceM2Tests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceM2Tests(ITestOutputHelper o) => _out = o;

    private static double SemDist(IPlatonicSpace s, string x, string y)
    {
        Assert.True(s.TryGetConceptFace(x, out var fx));
        Assert.True(s.TryGetConceptFace(y, out var fy));
        var semStart = FaceLayout.WordFaceStart(s.FaceDimension);
        var sum = 0.0;
        for (var i = semStart; i < fx.Length; i++) { var d = fx[i] - fy[i]; sum += d * d; }
        return Math.Sqrt(sum);
    }

    [Fact] // Law D / Law M — position EMERGES from the κ-context: a concept ends up near what it is related to and
           // far from a disjoint neighbourhood it was NEVER observed with. Identity by difference, not by stamp.
    public void Position_EmergesFromRelationalContext()
    {
        IPlatonicSpace space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        var clusterA = new[] { "cat", "kitten", "feline", "huba" };
        var clusterB = new[] { "dog", "puppy", "canine", "hubb" };
        // cat is NEVER observed with anything in B; its only context is A.
        for (var epoch = 0; epoch < 30; epoch++)
        {
            foreach (var cl in new[] { clusterA, clusterB })
                for (var i = 0; i < cl.Length; i++)
                    for (var j = i + 1; j < cl.Length; j++)
                        space.ObserveContradiction(cl[i], cl[j], 0.0);
        }

        var withinA = SemDist(space, "cat", "kitten");
        var acrossToB = SemDist(space, "cat", "dog");
        var hubSame = SemDist(space, "cat", "huba");
        var hubOther = SemDist(space, "cat", "hubb");
        _out.WriteLine($"cat→kitten {withinA:F3}  cat→dog {acrossToB:F3}   cat→huba {hubSame:F3}  cat→hubb {hubOther:F3}");
        Assert.True(withinA < acrossToB, $"cat must settle nearer its context (kitten) than a disjoint one (dog): {withinA:F3} vs {acrossToB:F3}");
        Assert.True(hubSame < hubOther, $"cat must settle nearer its own hub than the other: {hubSame:F3} vs {hubOther:F3}");
    }

    [Fact] // Anti-erosion — the failure that killed the old store. Under LONG training the separation must stay
           // stable (related clearly closer than unrelated) and NOT collapse or invert.
    public void Separation_StableUnderLongTraining()
    {
        IPlatonicSpace space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        const int C = 4, K = 5;
        var clusters = Enumerable.Range(0, C).Select(c => Enumerable.Range(0, K).Select(i => $"c{c}item{i}").ToArray()).ToArray();

        double sepAt(int epochs)
        {
            for (var e = 0; e < epochs; e++)
                foreach (var cl in clusters)
                    for (var i = 0; i < K; i++)
                        for (var j = i + 1; j < K; j++)
                            space.ObserveContradiction(cl[i], cl[j], 0.0);
            return space.SummarizePushPullGeometry().Separation;
        }

        var early = sepAt(20);
        var late = sepAt(200); // 200 more epochs — the regime where the old store eroded
        _out.WriteLine($"separation early(20)={early:F3}  late(+200)={late:F3}");
        Assert.True(early > 0.1, $"must separate early; {early:F3}");
        Assert.True(late > 0.1, $"must STAY separated under long training (no erosion/inversion); {late:F3}");
        Assert.True(late > 0.7 * early, $"separation must not decay materially; early={early:F3} late={late:F3}");
    }
}
