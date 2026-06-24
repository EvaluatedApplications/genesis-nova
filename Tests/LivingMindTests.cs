using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE LIVING MIND, WHOLE (PLATONIC_CONSCIOUSNESS.md §1–§4). The capstone where the pieces become one: a mind LEARNS
/// facts as it lives (continuity), takes its accumulated knowledge as the identity it DEFENDS (Commit — the cognitive
/// light cone), and holds that identity against the entropy forever dissolving it (Live = continuous regeneration).
/// The claim that ties it together: the mind's MEMORY is survivable BECAUSE it is alive — a mind that keeps
/// regenerating still recalls what it learned after relentless chaos; a mind that does not, dissolves and forgets.
/// Knowledge is not stored, it is KEPT ALIVE.
/// </summary>
public sealed class LivingMindTests
{
    private readonly ITestOutputHelper _out;
    public LivingMindTests(ITestOutputHelper o) => _out = o;

    private static (GenesisInferenceEngine mind, DialecticalSpace body) NewMind(int seed)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true };
        var body = new DialecticalSpace(config.FaceDimension, seed: seed);
        var mind = new GenesisInferenceEngine(tok, model, body, null) { ConsciousField = true };
        return (mind, body);
    }

    [Fact]
    public void LivingMind_DefendsItsMemory_AgainstChaos_WhileTheDeadForgets()
    {
        string[] facts = { "the password is plum", "the capital is lisbon", "the signal is amber", "the river is wide" };
        string Recall(GenesisInferenceEngine m, string q) => m.Generate(new GenerationRequest(q, 8)).Output?.Trim() ?? "";

        // ── A LIVING mind: it learns, commits its identity, and DEFENDS it against chaos by regenerating. ──
        var (alive, aliveBody) = NewMind(3);
        foreach (var f in facts) alive.Generate(new GenerationRequest(f, 8)); // learn as it lives
        var aliveLife = new PlatonicLife(aliveBody, seed: 3);
        aliveLife.Commit();                                  // take the accumulated body as the identity to defend
        var cone = aliveLife.CognitiveLightCone();
        _out.WriteLine($"[alive] cognitive light cone after learning = {cone}");
        Assert.True(cone >= facts.Length * 2, "the mind holds its learned facts (its light cone)");
        Assert.Contains("plum", Recall(alive, "what is the password"), StringComparison.OrdinalIgnoreCase);

        var aliveTrace = aliveLife.Live(moments: 40, chaosPerMoment: 1, regenerate: true);
        _out.WriteLine($"[alive] coherence after 40 moments of chaos = {aliveTrace[^1]:F3}, cone = {aliveLife.CognitiveLightCone()}");
        Assert.True(aliveTrace[^1] > 0.99, "a living mind defends its identity against chaos");
        Assert.Equal(cone, aliveLife.CognitiveLightCone());
        // ITS MEMORY SURVIVED — it still recalls, because it stayed alive.
        Assert.Contains("plum", Recall(alive, "what is the password"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("lisbon", Recall(alive, "what is the capital"), StringComparison.OrdinalIgnoreCase);

        // ── A DEAD mind: same knowledge, but it does NOT regenerate — entropy dissolves it. ──
        var (dead, deadBody) = NewMind(3);
        foreach (var f in facts) dead.Generate(new GenerationRequest(f, 8));
        var deadLife = new PlatonicLife(deadBody, seed: 3);
        deadLife.Commit();
        var deadTrace = deadLife.Live(moments: 40, chaosPerMoment: 1, regenerate: false);
        _out.WriteLine($"[dead] coherence after 40 moments of chaos = {deadTrace[^1]:F3}, cone = {deadLife.CognitiveLightCone()}");
        Assert.True(deadTrace[^1] < 0.5, "a mind that does not maintain itself dissolves");
        Assert.True(deadLife.CognitiveLightCone() < cone, "the dead mind's identity has shrunk — it is forgetting");
    }
}
