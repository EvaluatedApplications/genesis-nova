using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE VITAL LOOP (PLATONIC_CONSCIOUSNESS.md). The platonic space is a BODY; a self keeps it in continuous
/// REGENERATION against chaos (entropy/ablation). EMPIRICAL: a self that stays in regeneration holds its body
/// coherent under relentless perturbation (alive); one that does not, dissolves (dead). And because the body's
/// memory is conserved (G6), regeneration restores it as ITSELF — the learned identity reactivated, not reborn.
/// </summary>
public sealed class PlatonicLifeTests
{
    private readonly ITestOutputHelper _out;
    public PlatonicLifeTests(ITestOutputHelper o) => _out = o;

    private static readonly string[] Concepts = { "cat", "dog", "bird", "fish", "tree", "star" };

    private static DialecticalSpace BuildBody()
    {
        var b = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        for (var e = 0; e < 20; e++)
            for (var i = 0; i < Concepts.Length; i++)
                for (var j = i + 1; j < Concepts.Length; j++)
                    b.ObserveContradiction(Concepts[i], Concepts[j], 0.2);
        return b;
    }

    private static double SemDist(double[] x, double[] y, int dim)
    {
        var s = FaceLayout.WordFaceStart(dim);
        var sum = 0.0;
        for (var i = s; i < x.Length; i++) { var d = x[i] - y[i]; sum += d * d; }
        return Math.Sqrt(sum);
    }

    [Fact] // ALIVE vs DEAD — to be alive is to never stop regenerating.
    public void Life_StaysCoherentUnderChaos_While_DeathDissolves()
    {
        var alive = new PlatonicLife(BuildBody(), seed: 1); alive.Commit();
        var dead = new PlatonicLife(BuildBody(), seed: 1); dead.Commit();

        var aliveTrace = alive.Live(moments: 60, chaosPerMoment: 1, regenerate: true);
        var deadTrace = dead.Live(moments: 60, chaosPerMoment: 1, regenerate: false);

        _out.WriteLine($"ALIVE  coherence start {aliveTrace[0]:F2} → end {aliveTrace[^1]:F2}   light-cone {alive.CognitiveLightCone()}/{Concepts.Length}");
        _out.WriteLine($"DEAD   coherence start {deadTrace[0]:F2} → end {deadTrace[^1]:F2}   light-cone {dead.CognitiveLightCone()}/{Concepts.Length}");

        Assert.True(aliveTrace[^1] > 0.9, $"a self in continuous regeneration stays whole under chaos; end={aliveTrace[^1]:F2}");
        Assert.True(deadTrace[^1] < 0.4, $"without regeneration, entropy dissolves the body; end={deadTrace[^1]:F2}");
        Assert.True(alive.CognitiveLightCone() > dead.CognitiveLightCone(),
            "the living self keeps more of its body within reach than the dead one");
    }

    [Fact] // G6 — the self that comes back is the self that was: regeneration reactivates the LEARNED identity,
           // not a fresh neutral copy. Conserved memory is what makes a self survivable.
    public void Regeneration_RestoresTheLearnedSelf_NotANeutralCopy()
    {
        var body = BuildBody();
        var dim = body.FaceDimension;
        Assert.True(body.TryGetConceptFace("cat", out var original));
        original = (double[])original.Clone();

        var life = new PlatonicLife(body, seed: 1);
        life.Commit();

        body.Ablate("cat");
        Assert.False(body.ContainsConcept("cat")); // torn from the body

        life.Regenerate();
        Assert.True(body.ContainsConcept("cat"));   // restored
        Assert.True(body.TryGetConceptFace("cat", out var restored));

        // A never-learned neutral concept, for scale: how far a FRESH "cat" would be from the learned one.
        body.ObserveContradiction("nbneutral", "nbother", 0.5);
        Assert.True(body.TryGetConceptFace("nbneutral", out var neutral));

        var selfDist = SemDist(original, restored, dim);
        var neutralDist = SemDist(original, neutral, dim);
        _out.WriteLine($"regenerated→original {selfDist:F3}   neutral→original {neutralDist:F3}");
        Assert.True(selfDist < neutralDist, "the body came back as ITSELF, not reborn neutral (G6 conserved memory)");
    }
}
