using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// RELIABILITY-WEIGHTED ROUTING — "bubble up transform successes to the route head" (2026-06-16).
/// Observation that motivated it: the platonic space builds a working transform T(f) long BEFORE the route
/// head learns to pick it. So each transform now tracks its EARNED downstream success; that reliability —
/// UCB-shaped, with an exploration bonus so an unproven transform still gets tried instead of frozen out —
/// is surfaced into the route perception's spare channel (index 4). The route head therefore LEARNS to trust
/// the function/platonic route when its transforms are proven and distrust it when they're noisy.
/// These tests pin the mechanism end to end (earned success → UCB reliability → perception channel, persisted
/// across checkpoints) and that the route head can actually learn to route from that channel alone.
/// </summary>
public sealed class TransformReliabilityRoutingTests
{
    private readonly ITestOutputHelper _out;
    public TransformReliabilityRoutingTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Reliability_EarnedSuccess_RanksProvenAboveNoisy()
    {
        var acc = new TransformAccumulator(8);
        var inp = new double[8];
        var outp = new double[8]; outp[0] = 1.0;
        acc.Learn("proven", inp, outp);
        acc.Learn("noisy", inp, outp);

        for (var i = 0; i < 20; i++) acc.RecordOutcome("proven", correct: true);
        for (var i = 0; i < 20; i++) acc.RecordOutcome("noisy", correct: false);

        var proven = acc.Reliability("proven");
        var noisy = acc.Reliability("noisy");
        _out.WriteLine($"proven={proven:F3} noisy={noisy:F3}");
        Assert.True(proven > 0.85, $"proven should be high, was {proven}");
        Assert.True(noisy < 0.15, $"noisy should be low, was {noisy}");
        Assert.True(proven > noisy);
        Assert.Equal(0.0, acc.Reliability("unknown")); // an unknown transform earns nothing
    }

    [Fact]
    public void ReliabilityUcb_ExplorationBonus_KeepsUnprovenInPlay_AndBestReflectsProven()
    {
        var acc = new TransformAccumulator(8);
        var inp = new double[8]; var outp = new double[8]; outp[0] = 1.0;
        acc.Learn("proven", inp, outp);
        acc.Learn("fresh", inp, outp);
        for (var i = 0; i < 30; i++) acc.RecordOutcome("proven", true);

        // 'fresh' has ~0 attempts of its own while total attempts are high → its UCB carries a large
        // exploration bonus above its raw 0.5 rate, so the router won't freeze it out before it's tried.
        var freshRaw = acc.Reliability("fresh");
        var freshUcb = acc.ReliabilityUcb("fresh");
        Assert.True(freshUcb > freshRaw, $"UCB ({freshUcb:F3}) should exceed raw rate ({freshRaw:F3}) for an under-tried transform");

        Assert.True(acc.BestReliabilityUcb() >= acc.ReliabilityUcb("proven") - 1e-9);
        Assert.Equal(0.0, new TransformAccumulator(8).BestReliabilityUcb()); // no transforms → no signal
    }

    [Fact]
    public void ApplyImprovesOverIdentity_TracksConsistentTranslation()
    {
        const int dim = 8;
        var acc = new TransformAccumulator(dim);
        // Teach +3 on dim 0 from several consistent examples → a clean constant translation.
        for (var x = 0; x < 6; x++)
        {
            var i = new double[dim]; i[0] = x;
            var o = new double[dim]; o[0] = x + 3;
            acc.Learn("plus3", i, o);
        }
        // Held-out: applying +3 lands closer to the target than identity (doing nothing) does → success.
        var hi = new double[dim]; hi[0] = 42;
        var ho = new double[dim]; ho[0] = 45;
        Assert.True(acc.ApplyImprovesOverIdentity("plus3", hi, ho));
        // Target inconsistent with the learned translation (== input) → identity is exact, transform overshoots.
        var bad = new double[dim]; bad[0] = 42;
        Assert.False(acc.ApplyImprovesOverIdentity("plus3", hi, bad));
    }

    [Fact]
    public void RoutePerception_CarriesTransformReliability_InSpareChannel()
    {
        var space = new PlatonicSpaceMemory(faceDimension: 64, seed: 7);
        var withRel = space.ComputeRoutePerception("anything", transformReliability: 0.73);
        Assert.Equal(GenesisNeuralModel.EditPerceptionDim, withRel.Length);
        Assert.Equal(0.73, withRel[4], 3); // reliability lands in the spare channel (index 4)
        Assert.Equal(1.0, withRel[5]);      // bias channel unchanged
        Assert.Equal(1.0, space.ComputeRoutePerception("anything", 5.0)[4]); // clamped to [0,1]
        Assert.Equal(0.0, space.ComputeRoutePerception("anything")[4]);      // default 0 when omitted (feature off)
    }

    [Fact]
    public void ReliabilityCounts_SurviveCheckpointRoundTrip()
    {
        var acc = new TransformAccumulator(8);
        var inp = new double[8]; var outp = new double[8]; outp[0] = 1.0;
        acc.Learn("f", inp, outp);
        for (var i = 0; i < 7; i++) acc.RecordOutcome("f", true);
        for (var i = 0; i < 3; i++) acc.RecordOutcome("f", false);
        var before = acc.Reliability("f");

        var restored = new TransformAccumulator(8);
        restored.ImportSnapshot(acc.ExportSnapshot());
        Assert.Equal(before, restored.Reliability("f"), 6); // success/attempt counts persist through the checkpoint
    }

    [SlowFact]
    public void RouteHead_LearnsToTrustFunctionRoute_FromReliabilityChannel()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config); // PerceptionRouting + TransformReliabilityRouting default ON
        var tokens = tokenizer.Encode("apply flonk to 7");
        model.EnsureVocabularySize(tokenizer.VocabularySize);

        // Two perception vectors that differ ONLY in the reliability channel (index 4); bias (5) is shared,
        // all geometric channels (0–3) are zero — so any learned discrimination is attributable to reliability.
        static double[] HighRel() => new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
        static double[] LowRel() => new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

        // Teach: a PROVEN transform (high reliability) → route platonic (1); unproven (low) → neural (0).
        for (var i = 0; i < 400; i++)
        {
            model.ReinforceRouteHead(tokens, HighRel(), routeId: 1, reward: 1.0);
            model.ReinforceRouteHead(tokens, LowRel(), routeId: 0, reward: 1.0);
        }

        var high = model.PredictRoute(tokens, HighRel());
        var low = model.PredictRoute(tokens, LowRel());
        _out.WriteLine($"high-reliability → route {high.RouteId} (conf {high.Confidence:F2}); low → route {low.RouteId} (conf {low.Confidence:F2})");

        // Identical tokens → the reliability channel is the ONLY difference, so this isolates that the route
        // head LEARNED to route platonic from earned transform reliability — the success bubbled up.
        Assert.Equal(1, high.RouteId);
        Assert.NotEqual(1, low.RouteId);
    }
}
