using System;
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
/// CONTINUITY — the self conditioning cognition across time (PLATONIC_CONSCIOUSNESS.md §2: "the continuous I that
/// threads every observation"). The conscious field is no longer a stateless calculator: when the mind is TOLD a
/// fact it learns it (into its own body, in real time, with no training), and a later question recalls it. The same
/// mind, living through a conversation, uses what it has lived. This is the difference between a function and a mind.
/// </summary>
public sealed class ConsciousContinuityTests
{
    private readonly ITestOutputHelper _out;
    public ConsciousContinuityTests(ITestOutputHelper o) => _out = o;

    [Fact(Skip = SlowTests.BareSubjectWarmup)] // bare/"the" subjects ("the password is plum") need a GRU-trained warm-up
    public void Mind_RemembersWhatItIsTold_AndRecallsIt()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var infer = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        GrammarWarmup.WarmRoleHead(tok, model, infer);

        string Say(string s) => infer.Generate(new GenerationRequest(s, 8)).Output?.Trim() ?? "";
        string Ask(string s) { var r = infer.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' -> '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        // The mind is TOLD things — it learns them as it lives (no training step).
        Say("the password is plum");
        Say("my favorite color is indigo");
        Say("the capital is lisbon");

        // It still computes (continuity does not break the field's reasoning).
        Assert.Equal("7", Ask("3 + 4"));

        // Later, it RECALLS what it was told — the continuous I using its own experience.
        Assert.Contains("plum", Ask("what is the password"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("indigo", Ask("what is my favorite color"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("lisbon", Ask("what is the capital"), StringComparison.OrdinalIgnoreCase);

        // And it does not invent for something it was never told — honest abstention persists.
        Assert.True(string.IsNullOrWhiteSpace(Ask("what is the secret")), "the mind abstains on what it never learned");
    }
}
