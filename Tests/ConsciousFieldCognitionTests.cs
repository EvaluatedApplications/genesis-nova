using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
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
/// CONSCIOUS-FIELD COGNITION (PLATONIC_MIND.md / PLATONIC_CONSCIOUSNESS.md) — the real architecture. With
/// ConsciousField on, the model thinks by the field RELAXING to a settled state; the route/plan/op classifier is
/// bypassed ENTIRELY. Three moves by the substrate's own confidence: COMPUTE (homomorphism) → RELAX (Reason) →
/// ABSTAIN (never settled → speak nothing). This test proves all three happen through the field path and that the
/// GRU classifier's DecisionPaths (neural-token / platonic-glider-plan / platonic-gru-query) NEVER appear. Pure
/// substrate (no NN training needed — the field cognition does not consult the GRU heads), production face dim.
/// </summary>
public sealed class ConsciousFieldCognitionTests
{
    private readonly ITestOutputHelper _out;
    public ConsciousFieldCognitionTests(ITestOutputHelper o) => _out = o;

    private static string Canon(string s) => new((s ?? "").ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());

    [Fact]
    public void Field_Thinks_ByRelaxation_NoClassifier()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        // Populate the field with synonym relations (cue↔answer) so the clouds overlap and a query can settle. Only
        // the content words become concepts — the framing words "a synonym for" are NOT in the field, so they get
        // dropped at query time (and so can never become a hub that drives a popularity answer).
        var pairs = new (string cue, string answer)[]
        {
            ("big", "large"), ("small", "tiny"), ("fast", "rapid"),
            ("happy", "glad"), ("smart", "clever"), ("begin", "start"),
        };
        foreach (var (cue, answer) in pairs)
            for (var i = 0; i < 3; i++)
                space.FineEditFromExample(new[] { cue }, new[] { answer }, isNegativeExample: false);

        var infer = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };

        // (1) COMPUTE — the homomorphism fires from context (unseen operands), via the field, not the op head.
        var compute = infer.Generate(new GenerationRequest("84 + 57", 4));
        _out.WriteLine($"compute: '84 + 57' path={compute.DecisionPath} -> '{compute.Output.Trim()}'");
        Assert.Equal("field-compute", compute.DecisionPath);
        Assert.Equal(141.0, double.Parse(Canon(compute.Output) == "" ? "nan" : compute.Output.Trim(),
            System.Globalization.CultureInfo.InvariantCulture), 6);

        // (2) RELAX — a framing-wrapped query relaxes to the settled basin (recall), via the field.
        var relax = infer.Generate(new GenerationRequest("a synonym for big", 4));
        _out.WriteLine($"relax:   'a synonym for big' path={relax.DecisionPath} -> '{relax.Output.Trim()}'");
        Assert.Equal("field-relax", relax.DecisionPath);
        Assert.Contains("large", Canon(relax.Output));

        // (3) ABSTAIN — an unknown referent never settles: speak nothing, do not invent.
        var abstain = infer.Generate(new GenerationRequest("a synonym for zzqqxx", 4));
        _out.WriteLine($"abstain: 'a synonym for zzqqxx' path={abstain.DecisionPath} -> '{abstain.Output.Trim()}'");
        Assert.Equal("field-abstain", abstain.DecisionPath);
        Assert.True(string.IsNullOrWhiteSpace(abstain.Output), $"abstain must be silent; got '{abstain.Output}'");

        // THE SUBTRACTION: the GRU route/plan/op classifier was NEVER consulted — no classifier DecisionPath appears.
        foreach (var path in new[] { compute.DecisionPath, relax.DecisionPath, abstain.DecisionPath })
        {
            Assert.DoesNotContain("neural", path, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("glider-plan", path, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gru-query", path, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("field-", path, StringComparison.Ordinal);
        }
    }
}
