using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// The op is LEARNED, not hardcoded — "op classified from context", done in the field (relations + self), not a synonym
// list and not the GRU's stateless head. From examples nova infers which operation reproduces the answer and relates
// the cue (word OR a coined symbol) to it; at inference it resolves the op by that learned relation and computes via
// the homomorphism. Generalises to held-out operands and to ANY notation it has been shown.
public sealed class FieldOpLearningTests
{
    private readonly ITestOutputHelper _out;
    public FieldOpLearningTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_LearnsArithmeticCues_FromExamples_NoneHardcoded()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        DialecticalSpace Space() => new(config.FaceDimension, seed: 7);

        var space = Space();
        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true };

        // "result" (multiply), "ratio" (divide), and a COINED operator "glorp" (add) — none are in any hardcoded list.
        // nova infers the op from the answer and relates the cue to it.
        foreach (var (i, o) in new[] { ("the result of 2 and 3", "6"), ("the result of 4 and 5", "20"), ("the result of 3 and 6", "18") })
            mind.LearnArithmeticCue(i, o);
        foreach (var (i, o) in new[] { ("the ratio of 6 and 2", "3"), ("the ratio of 20 and 5", "4"), ("the ratio of 9 and 3", "3") })
            mind.LearnArithmeticCue(i, o);
        foreach (var (i, o) in new[] { ("5 glorp 3", "8"), ("4 glorp 6", "10"), ("7 glorp 1", "8") })
            mind.LearnArithmeticCue(i, o);

        void Check(string q, string want)
        {
            var r = mind.Generate(new GenerationRequest(q, 8));
            _out.WriteLine($"{q,-26} -> '{r.Output?.Trim()}' [{r.DecisionPath}]");
            Assert.Equal(want, r.Output?.Trim());
        }
        Check("the result of 6 and 8", "48");   // learned result = ×, on held-out operands
        Check("the ratio of 24 and 6", "4");    // learned ratio = ÷
        Check("9 glorp 2", "11");               // learned coined operator = +

        // It is genuinely LEARNED, not hardcoded: an UNTAUGHT mind doesn't know "result" and abstains.
        var fresh = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), Space(), null) { ConsciousField = true };
        var blank = fresh.Generate(new GenerationRequest("the result of 6 and 8", 8));
        _out.WriteLine($"untaught 'the result of 6 and 8' -> '{blank.Output?.Trim()}' [{blank.DecisionPath}]");
        Assert.NotEqual("48", blank.Output?.Trim());
    }
}
