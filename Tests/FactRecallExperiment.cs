using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// GENERAL fact memory — NO hardcoding. Teach a fact with an ordinary assertion ("my name is X") and recall it with an
// ordinary question ("what is my name") through the substrate's native learn (TryFieldLearn) + retrieve, exactly the
// same path as any other association. If this works, "remember my name" is just a special case of general fact memory,
// learned, not a coded name routine.
public sealed class FactRecallExperiment
{
    private readonly ITestOutputHelper _out;
    public FactRecallExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_LearnsThenRecalls_AnAssertedFact()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true };
        string Say(string s) { var r = mind.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' → '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        Say("my name is stephen");        // learn it (general assertion path)
        Say("my dog is rex");
        var name = Say("what is my name"); // recall it (general retrieval)
        var dog = Say("what is my dog");

        Assert.Contains("stephen", name, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("rex", dog, StringComparison.OrdinalIgnoreCase);
    }
}
