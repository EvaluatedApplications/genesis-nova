using System.Globalization;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// SELF-HEAL A CUE MISROUTE — proves the training regimen can RECOVER FROM A WRONG ROUTE, the gap that left
/// subtraction stuck (see nova-subtract-stuck-compare-hijack). Corpus contamination relates the operator symbol "-"
/// to the compare anchor ∘cmp, so the route ladder (predicate before arithmetic) hijacks "12 - 1" → "greater". That
/// edge is otherwise IMMORTAL: routing has no trainable parameter, the subtract curriculum never touches the edge,
/// and a weakened cue still wins. This test reproduces the hijack, drives the SAME signal the gym now applies on a
/// value-wrong probe (HealMisroutedCue), and asserts the skill recovers — while a genuine comparison still works and
/// the default-off path stays byte-identical (no disruption).
/// </summary>
public sealed class SelfHealMisroutedCueTests
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    private readonly ITestOutputHelper _out;
    public SelfHealMisroutedCueTests(ITestOutputHelper o) => _out = o;

    // A space contaminated EXACTLY like production: a genuine compare cue + outcome vocabulary, PLUS the corpus
    // contamination "-" → ∘cmp (a hyphenated number range answered by a word trains this in LearnIntentCue).
    private static DialecticalSpace ContaminatedSpace(int dim)
    {
        var space = new DialecticalSpace(dim, seed: 7);
        for (var i = 0; i < 3; i++)
        {
            space.FineEditFromExample(new[] { "compared" }, new[] { "∘cmp" }, isNegativeExample: false); // genuine compare cue
            space.FineEditFromExample(new[] { "greater" }, new[] { "∘gt" }, isNegativeExample: false);    // outcome vocabulary
            space.FineEditFromExample(new[] { "less" }, new[] { "∘lt" }, isNegativeExample: false);
            space.FineEditFromExample(new[] { "equal" }, new[] { "∘eq" }, isNegativeExample: false);
            space.FineEditFromExample(new[] { "-" }, new[] { "∘cmp" }, isNegativeExample: false);          // THE CONTAMINATION
        }
        return space;
    }

    private static GenesisInferenceEngine Engine(DialecticalSpace space, bool selfHeal)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        return new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config), space, null)
        {
            ConsciousField = true,        // think by the field ladder (predicate → arithmetic …)
            LearnedCuesOnly = true,       // CompareCue resolves via the LEARNED ∘cmp relation (production path)
            SelfHealMisroutedCues = selfHeal,
        };
    }

    private static readonly (string Q, string A)[] SubProbes =
    {
        ("12 - 1", "11"), ("9 - 4", "5"), ("15 - 6", "9"), ("8 - 3", "5"),
        ("20 - 7", "13"), ("11 - 2", "9"), ("14 - 5", "9"), ("7 - 1", "6"),
    };

    [Fact]
    public void FocusedTraining_HealsTheCompareHijack_AndKeepsGenuineCompare()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = ContaminatedSpace(config.FaceDimension);
        var eng = Engine(space, selfHeal: true);

        // REPRODUCE THE HIJACK: the subtraction symbol routes to the compare path and answers "greater".
        var hijack = eng.Generate(new GenerationRequest("12 - 1", 4));
        _out.WriteLine($"before: '12 - 1' path={hijack.DecisionPath} -> '{hijack.Output.Trim()}'");
        Assert.Equal("field-predicate", hijack.DecisionPath);
        Assert.Equal("greater", hijack.Output.Trim());

        // FOCUSED SUBTRACT TRAINING, exactly as the gym now does it: predict each probe; on a value-WRONG answer,
        // apply the self-heal signal (HealMisroutedCue). No gradient on the homomorphism is needed — the fix is
        // unlearning the bad routing cue.
        for (var pass = 0; pass < 2; pass++)
            foreach (var (q, a) in SubProbes)
            {
                var r = eng.Generate(new GenerationRequest(q, 4));
                if (r.Output.Trim() != a)
                    eng.HealMisroutedCue(q, new[] { a }, r.Output);
            }

        // HEALED: the same query now computes the difference via the arithmetic route.
        var healed = eng.Generate(new GenerationRequest("12 - 1", 4));
        _out.WriteLine($"after:  '12 - 1' path={healed.DecisionPath} -> '{healed.Output.Trim()}'");
        Assert.Equal("field-compute", healed.DecisionPath);
        Assert.Equal(11.0, double.Parse(healed.Output.Trim(), NumberStyles.Float, Inv), 6);

        // SURGICAL: a GENUINE comparison still routes to predicate and answers the outcome word (only "-" was unlearned).
        var cmp = eng.Generate(new GenerationRequest("12 compared to 1", 4));
        _out.WriteLine($"compare:'12 compared to 1' path={cmp.DecisionPath} -> '{cmp.Output.Trim()}'");
        Assert.Equal("field-predicate", cmp.DecisionPath);
        Assert.Equal("greater", cmp.Output.Trim());
    }

    [Fact]
    public void Default_Off_IsByteIdentical_NoHealing()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = ContaminatedSpace(config.FaceDimension);
        var eng = Engine(space, selfHeal: false); // gate OFF

        Assert.Equal("greater", eng.Generate(new GenerationRequest("12 - 1", 4)).Output.Trim());

        // Run the SAME focused loop; with the gate off HealMisroutedCue is a no-op, so nothing is unlearned …
        for (var pass = 0; pass < 2; pass++)
            foreach (var (q, a) in SubProbes)
            {
                var r = eng.Generate(new GenerationRequest(q, 4));
                if (r.Output.Trim() != a) eng.HealMisroutedCue(q, new[] { a }, r.Output);
            }

        // … and the hijack persists (the regimen can NOT recover — the very gap this feature closes).
        var still = eng.Generate(new GenerationRequest("12 - 1", 4));
        _out.WriteLine($"gate-off after loop: '12 - 1' path={still.DecisionPath} -> '{still.Output.Trim()}'");
        Assert.Equal("field-predicate", still.DecisionPath);
        Assert.Equal("greater", still.Output.Trim());
    }
}
