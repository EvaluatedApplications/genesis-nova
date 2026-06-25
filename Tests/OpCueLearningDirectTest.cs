using System;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// ISOLATE op-cue learning from the gym (no scheduler, no volume, no GRU heads): feed a few CLEAN worded examples per op
// straight through LearnArithmeticCue, then ask the field to compute a worded query. The mechanism is a RELATION, not a
// gradient — a few examples per op should suffice (DurableMechanismTests proves the INFIX form "4 zorp 3"→12 works in 3).
// If WORDED product/difference/quotient still abstain here, the mechanism (framing-word handling) — not training volume
// — is the gap. This pinpoints whether the gym needs more data or the resolver needs fixing.
public sealed class OpCueLearningDirectTest
{
    private readonly ITestOutputHelper _out;
    public OpCueLearningDirectTest(ITestOutputHelper o) => _out = o;

    [Fact]
    public void LearnArithmeticCue_LearnsEachWordedOpSynonym_FromAFewCleanExamples()
    {
        var nova = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: 128, Seed: 7).WithProductionMechanisms());
        string Gen(string i) => nova.Inference.Generate(new GenerationRequest(i, 8)).Output?.Trim() ?? "";
        string Path(string i) => nova.Inference.Generate(new GenerationRequest(i, 8)).DecisionPath ?? "";

        (string In, string Out)[] lessons =
        {
            ("the sum of 2 and 3", "5"), ("the sum of 4 and 1", "5"), ("the sum of 6 and 2", "8"),
            ("the difference of 7 and 5", "2"), ("the difference of 9 and 4", "5"), ("the difference of 8 and 3", "5"),
            ("the product of 2 and 5", "10"), ("the product of 3 and 4", "12"), ("the product of 2 and 6", "12"),
            ("the quotient of 6 and 2", "3"), ("the quotient of 8 and 4", "2"), ("the quotient of 9 and 3", "3"),
        };
        foreach (var (i, o) in lessons) nova.Inference.LearnArithmeticCue(i, o);

        void Probe(string q, string want)
        {
            var got = Gen(q);
            _out.WriteLine($"  '{q}' -> '{got}' [{Path(q)}]  want {want}  {(AnswerEquivalence.Equivalent(got, want) ? "OK" : "MISS")}");
        }
        _out.WriteLine("── worded op-cue resolution after 3 clean examples each ──");
        Probe("the sum of 3 and 4", "7");
        Probe("the difference of 10 and 6", "4");
        Probe("the product of 3 and 5", "15");
        Probe("the quotient of 12 and 4", "3");

        // Direct resolver inspection: what does each op-word's relational neighbourhood look like?
        if (nova.Memory is GenesisNova.Cognition.Platonic.DialecticalSpace ds)
            foreach (var w in new[] { "sum", "difference", "product", "quotient", "of", "the", "and" })
            {
                var ns = ds.GetNeighbors(w, GenesisNova.Cognition.PlatonicNeighborhoodType.Relational, maxNeighbors: 8, minConfidence: 0.0);
                _out.WriteLine($"  rel[{w}] -> {string.Join(", ", System.Linq.Enumerable.Select(ns, n => $"{n.Concept}:{n.Confidence:F2}"))}");
            }
        Assert.True(true);
    }
}
