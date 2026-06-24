using System;
using System.Collections.Generic;
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
/// REASONING FROM THE SELF (PLATONIC_MIND.md §6, disambiguation-by-relaxation). The mind reasons FROM WHO IT IS:
/// the same ambiguous question relaxes into the basin consistent with what the mind has been attending to. Two minds
/// over the same body — one whose focus is the river, one whose focus is money — answer "what is a bank" differently,
/// each from its own context. The self conditions the reasoning itself, and (proven separately) without disturbing a
/// query that is already clear.
/// </summary>
public sealed class ConsciousReasoningFromSelfTests
{
    private readonly ITestOutputHelper _out;
    public ConsciousReasoningFromSelfTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Mind_DisambiguatesAnAmbiguousQuery_FromItsContext()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        void Rel(string a, string b) { for (var i = 0; i < 3; i++) space.FineEditFromExample(new[] { a }, new[] { b }, false); }
        // "bank" is genuinely AMBIGUOUS — near a river-sense cluster AND a money-sense cluster.
        Rel("bank", "river"); Rel("bank", "money");
        Rel("river", "water"); Rel("river", "stream"); Rel("water", "stream");   // river sense
        Rel("money", "cash"); Rel("money", "coin"); Rel("cash", "coin");          // money sense

        var riverSense = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "river", "water", "stream" };
        var moneySense = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "money", "cash", "coin" };

        // Mind A has been attending to the RIVER.
        var mindA = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        mindA.Generate(new GenerationRequest("the flowing river", 6));
        var a = mindA.Generate(new GenerationRequest("what is a bank", 6));
        _out.WriteLine($"  river-mind: 'what is a bank' -> '{a.Output?.Trim()}' [{a.DecisionPath}]");

        // Mind B has been attending to MONEY.
        var mindB = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        mindB.Generate(new GenerationRequest("the cash and money", 6));
        var b = mindB.Generate(new GenerationRequest("what is a bank", 6));
        _out.WriteLine($"  money-mind: 'what is a bank' -> '{b.Output?.Trim()}' [{b.DecisionPath}]");

        Assert.Contains(a.Output?.Trim() ?? "", riverSense);
        Assert.Contains(b.Output?.Trim() ?? "", moneySense);
        Assert.NotEqual(a.Output?.Trim(), b.Output?.Trim()); // the same query, different self → different reasoning
        Assert.True(a.DecisionPath == "field-relax-self" || b.DecisionPath == "field-relax-self",
            "at least one answer was tipped by the self's context");
    }
}
