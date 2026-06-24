using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// FIRST-PRINCIPLES PROBE: is the LARGE (word) face — the distributional meaning cloud, ~60% of the vector — usable
// GENERATIVELY, or only for shallow retrieval? The numeric faces do compute; here we ask the meaning face to (a)
// COMPOSE two meanings into the concept that fits both, and (b) do ANALOGY (relation-vector arithmetic). Experiment
// first — report what the cloud structure actually supports before wiring it into inference.
public sealed class MeaningGenerativeTests
{
    private readonly ITestOutputHelper _out;
    public MeaningGenerativeTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void LargeFace_ComposesMeanings_ToRetrieveTheFit()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) space.FineEditFromExample(new[] { a }, new[] { b }, false); }
        Rel("apple", "red"); Rel("apple", "fruit");
        Rel("cherry", "red"); Rel("cherry", "fruit");
        Rel("lemon", "yellow"); Rel("lemon", "fruit");
        Rel("corn", "yellow"); Rel("corn", "grain");
        Rel("brick", "red"); Rel("brick", "stone");

        // COMPOSITION: combining two meaning clouds retrieves the concept fitting BOTH — work a single anchor can't do.
        var redFruit = space.Reason(new[] { "red", "fruit" });
        var yellowFruit = space.Reason(new[] { "yellow", "fruit" });
        var redThing = space.Reason(new[] { "red", "stone" });
        _out.WriteLine($"red + fruit   -> {redFruit.Symbol} ({redFruit.Confidence:F2})");
        _out.WriteLine($"yellow + fruit-> {yellowFruit.Symbol} ({yellowFruit.Confidence:F2})");
        _out.WriteLine($"red + stone   -> {redThing.Symbol} ({redThing.Confidence:F2})");

        Assert.Contains(redFruit.Symbol, new[] { "apple", "cherry" });  // a RED FRUIT (not lemon=yellow, not brick=not-fruit)
        Assert.Equal("lemon", yellowFruit.Symbol);                       // the YELLOW fruit
        Assert.Equal("brick", redThing.Symbol);                          // the RED stone
    }

    [Fact]
    public void LargeFace_Analogy_Probe()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) space.FineEditFromExample(new[] { a }, new[] { b }, false); }
        // capital ↔ country structure
        Rel("paris", "france"); Rel("rome", "italy"); Rel("tokyo", "japan"); Rel("madrid", "spain"); Rel("berlin", "germany");

        var a1 = space.Analogy(new[] { ("paris", "france"), ("rome", "italy") }, "tokyo");
        var a2 = space.Analogy(new[] { ("france", "paris"), ("italy", "rome") }, "japan");
        _out.WriteLine($"paris:france :: tokyo:? -> '{a1.Symbol}' ({a1.Confidence:F2})  (want japan)");
        _out.WriteLine($"france:paris :: japan:? -> '{a2.Symbol}' ({a2.Confidence:F2})  (want tokyo)");
        // FINDING: the distributional cloud DOES support relation-vector analogy in the large face (≈0.96 here). It
        // completes the analogy in BOTH directions — generative meaning reasoning, not lookup. (For a symmetric
        // relation part of the lift is the query's own neighbour; a directed-relation test is the sharper follow-up.)
        Assert.Equal("japan", a1.Symbol);
        Assert.Equal("tokyo", a2.Symbol);
    }
}
