using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// The space bounds itself by RELEVANCE-DECAY, not an arbitrary size cap. An arbitrary cap "contracts" the space — it
// evicts whatever is lowest-value the instant a counter crosses a line, which churns and disturbs REINFORCED structure
// (it broke function-word warming). Instead the space DISCHARGES noise: a concept barely observed AND gone stale (not
// re-seen for a window) is released; anything reinforced or recently active is always kept. So the space holds as much
// relevant structure as exists, and reinforced signals are never touched. These tests pin both halves: noise discharges
// (bounded), and — critically — the function-word geometry SURVIVES a flood of noise (the regression the cap caused).
public sealed class SpaceEvictionTests
{
    private readonly ITestOutputHelper _out;
    public SpaceEvictionTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Discharge_releases_stale_noise_but_keeps_reinforced()
    {
        var space = new DialecticalSpace(faceDimension: 64)
        { DischargeStalenessWindow = 1_000, DischargeInterval = 400 };

        // A few RECURRING concepts (reinforced every step) plus a stream of unique HAPAX (each seen exactly once).
        string[] keep = { "alpha", "beta", "gamma", "delta", "epsilon" };
        const int steps = 8_000;
        for (var i = 0; i < steps; i++)
        {
            space.ObserveContradiction(keep[i % keep.Length], keep[(i + 1) % keep.Length], 0.2); // reinforce the keepers
            space.ObserveContradiction("hapax" + i, keep[i % keep.Length], 0.2);                 // a once-seen noise token
        }

        _out.WriteLine($"created≈{steps + keep.Length}  active={space.NodeCount}");

        // KEPT: the reinforced concepts survive — they are never candidates (observed far more than the threshold).
        foreach (var k in keep) Assert.True(space.ContainsConcept(k), $"reinforced concept '{k}' was wrongly discharged");

        // DISCHARGED: the early hapax (seen once, long stale) are released as noise.
        var earlyGone = Enumerable.Range(0, 400).Count(i => !space.ContainsConcept("hapax" + i));
        Assert.True(earlyGone > 350, $"stale noise not discharged — only {earlyGone}/400 released");

        // BOUNDED BY RELEVANCE (no cap): the active set is a small multiple of the staleness window, FAR below the 8k created.
        Assert.True(space.NodeCount < 3_500, $"space not bounded by relevance: {space.NodeCount} active of {steps} created");
    }

    [Fact]
    public void Function_word_geometry_survives_a_flood_of_noise()
    {
        // THE REGRESSION TEST: an arbitrary cap churned the space and collapsed function-word separation to 0. Relevance-
        // decay must NOT — function/content words are reinforced every step, so their geometry is never disturbed while the
        // hapax flood is discharged around them.
        var space = new DialecticalSpace(faceDimension: 96)
        { DischargeStalenessWindow = 1_500, DischargeInterval = 500 };

        string[][] clusters =
        {
            new[]{"cat","dog","cow","pig","hen","fox","owl","bat","ant","elk"},
            new[]{"red","blue","green","pink","gray","black","white","brown","gold","teal"},
            new[]{"bob","sam","joe","amy","tom","kim","dan","liz","ben","eve"},
            new[]{"name","color","age","city","job","pet","car","book","song","food"},
        };
        string[] func = { "the", "of", "in", "to", "and", "is", "a", "on", "for", "with" };
        var rng = new Random(13);
        for (var step = 0; step < 12_000; step++)
        {
            var cl = clusters[rng.Next(clusters.Length)];
            var w1 = cl[rng.Next(cl.Length)]; var w2 = cl[rng.Next(cl.Length)];
            if (w1 != w2) space.ObserveContradiction(w1, w2, 0.15);                         // content clusters with its kind
            space.ObserveContradiction(func[rng.Next(func.Length)], clusters[rng.Next(clusters.Length)][rng.Next(10)], 0.2); // glue bridges
            space.ObserveContradiction("noise" + step, clusters[rng.Next(clusters.Length)][rng.Next(10)], 0.2);            // hapax flood
        }

        double Frac(string[] ws) => ws.Count(space.IsFunctionLike) / (double)ws.Length;
        var fFrac = Frac(func);
        var cFrac = Frac(clusters.SelectMany(c => c).ToArray());
        _out.WriteLine($"active={space.NodeCount}  function-like: func={fFrac:F2} content={cFrac:F2}");

        // SEPARATION SURVIVES the noise + discharge (this is exactly what the arbitrary cap destroyed).
        Assert.True(fFrac >= 0.7, $"function words lost their separation under noise (func fn-like {fFrac:F2})");
        Assert.True(fFrac - cFrac >= 0.4, $"function/content separation collapsed (func {fFrac:F2} content {cFrac:F2})");
        // …and the space stayed bounded despite 12k hapax flooding through it.
        Assert.True(space.NodeCount < 5_000, $"space not bounded under the noise flood: {space.NodeCount} active");
    }

    [Fact]
    public void Contribution_to_answers_outlives_mere_observation()
    {
        // THE STRONGER SIGNAL: "contributed to correct answers" should keep a concept alive longer than "was merely seen".
        // Two concepts observed IDENTICALLY (3× each, then abandoned); the only difference is one's edge gets CREDITED for
        // producing right answers (what ReinforceEvidence does). After the clock runs far past both their bare grace, the
        // credited one survives (utility bought it grace) and the merely-observed one decays out.
        var space = new DialecticalSpace(faceDimension: 64) { DischargeStalenessWindow = 500, DischargeInterval = 200 };
        for (var i = 0; i < 3; i++)
        {
            space.ObserveContradiction("alpha", "useful", 0.2);   // will be credited
            space.ObserveContradiction("beta", "idle", 0.2);      // observed the same, never credited
        }
        // Credit the alpha–useful edge for contributing to CORRECT answers (the live ReinforceEvidence path).
        for (var i = 0; i < 12; i++)
            space.ReinforceEvidence(new[] { new PlatonicEvidence("alpha", "useful", 1.0, 1) }, success: true);

        // Advance the clock FAR past both bare graces WITHOUT touching either pair (unrelated noise).
        for (var step = 0; step < 5_000; step++) space.ObserveContradiction("n" + step, "filler", 0.2);

        Assert.True(space.ContainsConcept("useful"), "a concept whose edge EARNED correct answers was wrongly discharged");
        Assert.False(space.ContainsConcept("idle"), "a merely-observed concept (same observation, no contribution) was not discharged");
        Assert.True(space.GetRelationDegree("useful") > 0, "the credited edge was dropped");
        Assert.Equal(0, space.GetRelationDegree("idle"));     // the un-credited edge decayed away
    }
}
