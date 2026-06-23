using System;
using System.Linq;
using GenesisNova.Cognition;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Rung 1 of PLATONIC_BACKPROP.md — TASK-OUTCOME disruption: a value-wrong answer must REPEL the produced concept
/// away from the query anchor in the geometry (the function signal reaching the space), while NEVER scattering a
/// frozen number. Production face dimension (no shrinking for speed).
/// </summary>
public sealed class FunctionDisruptionTests
{
    private static PlatonicSpaceMemory NewSpace() =>
        new(faceDimension: 512, seed: 7, maxNodes: 1000, maxRelations: 5000);

    private static double DistTo(PlatonicSpaceMemory m, string anchor, string target)
    {
        foreach (var (sym, dist) in m.GetNearestConceptsFresh(anchor, seeds: null, maxNeighbors: 64))
            if (string.Equals(sym, target, StringComparison.OrdinalIgnoreCase))
                return dist;
        return double.NaN;
    }

    [Fact]
    public void DisruptAssociation_PushesWrongAnswerAwayFromAnchor()
    {
        var m = NewSpace();
        // Pull apple↔dog together (a learned-but-WRONG association the geometry made retrievable).
        for (var i = 0; i < 40; i++) m.ObserveContradiction("apple", "dog", 0.05);
        var before = DistTo(m, "apple", "dog");
        Assert.False(double.IsNaN(before), "dog should be a measurable neighbour of apple after attraction");

        for (var i = 0; i < 25; i++) m.DisruptAssociation("apple", "dog"); // value-wrong outcome → repel

        var after = DistTo(m, "apple", "dog");
        Assert.False(double.IsNaN(after));
        Assert.True(after > before, $"disruption must push dog away from apple: before {before:F3} after {after:F3}");
    }

    [Fact]
    public void FunctionGradient_MakesTaskTarget_TheNearestNeighbour_OverAConfuser()
    {
        var m = NewSpace();
        // Start WRONG: apple↔dog pulled tight (the confuser), while fruit lives elsewhere (near plant, not apple).
        for (var i = 0; i < 40; i++) m.ObserveContradiction("apple", "dog", 0.05);
        for (var i = 0; i < 8; i++) m.ObserveContradiction("fruit", "plant", 0.05);
        var dogBefore = DistTo(m, "apple", "dog");
        var fruitBefore = DistTo(m, "apple", "fruit");
        Assert.True(dogBefore < fruitBefore, $"setup: dog should start nearer than fruit ({dogBefore:F3} < {fruitBefore:F3})");

        // Descend the function gradient toward the TASK target (fruit), with dog as the confuser.
        for (var i = 0; i < 80; i++) m.FunctionGradientStep("apple", "fruit", new[] { "dog" });

        var dogAfter = DistTo(m, "apple", "dog");
        var fruitAfter = DistTo(m, "apple", "fruit");
        Assert.True(fruitAfter < dogAfter,
            $"function gradient must make the target win: fruit {fruitAfter:F3} should be < dog {dogAfter:F3}");
        Assert.True(fruitAfter < fruitBefore, "target should have been pulled closer to the anchor");
    }

    [Fact]
    public void DisruptAssociation_FrozenNumber_IsSafeNoOp()
    {
        var m = NewSpace();
        for (var i = 0; i < 40; i++) m.ObserveContradiction("apple", "fruit", 0.05);
        var before = DistTo(m, "apple", "fruit");

        m.DisruptAssociation("apple", "29"); // "29" is a frozen number — must not scatter, must not move apple
        m.DisruptAssociation("apple", "apple"); // self — no-op

        var after = DistTo(m, "apple", "fruit");
        Assert.Equal(before, after, 6); // apple's real neighbour is untouched by the frozen-number disruption
    }
}
