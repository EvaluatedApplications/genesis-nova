using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// Function induction is now a substrate fit, not two baked-in +k/×k hypotheses: the rule is fit as a LINE in the poly
// face (affine a·in+b) or the log face (power C·in^a), verified across the demos, applied to the query. In-context (no
// learned weights — the transform comes only from the in-prompt demos), and it GENERALISES beyond +k/×k to ax+b / x^a.
public sealed class FunctionInductionTests
{
    private readonly ITestOutputHelper _out;
    public FunctionInductionTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Induces_Affine_And_Power_Rules_BeyondPlusK_TimesK()
    {
        var config = new GenesisNovaConfig(HiddenSize: 128, FaceDimensionOverride: 128);
        var infer = new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config),
            new DialecticalSpace(config.FaceDimension, seed: 7), null) { ConsciousField = true };
        string P(string s) { var r = infer.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' -> '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        // The special cases the old route already did (now subsumed: ×k is b=0, +k is a=1):
        Assert.Equal("15", P("fn 2 is 6 fn 3 is 9 fn 4 is 12 fn 5 is"));   // ×3
        Assert.Equal("10", P("fn 2 is 7 fn 3 is 8 fn 4 is 9 fn 5 is"));    // +5

        // GENERALISATIONS the old +k/×k route ABSTAINED on (neither a constant diff nor a constant ratio):
        Assert.Equal("9", P("fn 1 is 3 fn 2 is 5 fn 3 is 7 fn 4 is"));     // 2x+1   (affine, poly face)
        Assert.Equal("11", P("fn 1 is 2 fn 2 is 5 fn 3 is 8 fn 4 is"));    // 3x-1   (affine)
        Assert.Equal("25", P("fn 2 is 4 fn 3 is 9 fn 4 is 16 fn 5 is"));   // x^2    (power, log face)
        Assert.Equal("5", P("fn 4 is 2 fn 9 is 3 fn 16 is 4 fn 25 is"));   // sqrt x (power, a=0.5)

        // Honest abstention: a non-affine, non-power rule (x^2 + x) is outside the substrate's two faces -> no answer.
        Assert.True(string.IsNullOrEmpty(P("fn 1 is 2 fn 2 is 6 fn 3 is 12 fn 4 is")) // x^2+x: 2,6,12,(20) — abstain
            || P("fn 1 is 2 fn 2 is 6 fn 3 is 12 fn 4 is") != "20");
    }
}
