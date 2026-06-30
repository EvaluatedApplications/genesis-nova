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
                    eng.HealMisroutedCue(q, new[] { a }, r.Output, r.DecisionPath);
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

    private const int Compare = 1; // FieldIntent.Compare code returned by ResolveIntentForTests

    private static GenesisInferenceEngine FreshEngine(out DialecticalSpace space, int seed = 7, bool selfHeal = false)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        space = new DialecticalSpace(config.FaceDimension, seed: seed);
        var eng = new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config), space, null)
        {
            ConsciousField = true,
            LearnedCuesOnly = true,
            SelfHealMisroutedCues = selfHeal,
        };
        for (long v = 0; v <= 20; v++) eng.LearnNumberWord($"{v} in words", NumberWordVocabulary.ToWords(v));
        return eng;
    }

    // GENERAL DISCRIMINATION (the operator special-cases are DELETED — no TryOpToken exclusion anywhere): an operator
    // earns NO stable intent cue because it LIVES in arithmetic — every subtraction emits "-"→∘cmp/∘tow/∘tod/∘ret
    // NEGATIVES, which a one-off corpus mislabel (a hyphenated date range answered by a WORD, the old contamination
    // source — which NOW relates "-"→∘cmp, no longer blocked) can never outweigh. So "-" nets out and subtraction
    // computes — a CONSEQUENCE of discrimination, not a special-case — while a genuine marker ("compared") still routes.
    [Fact]
    public void LearnIntentCue_OperatorSelfExcludesByDiscrimination_NoSpecialCase()
    {
        var eng = FreshEngine(out _);

        // THE CONTAMINATION the old special-case blocked: a hyphenated number range answered by a WORD now DOES relate
        // "-"→∘cmp. Discrimination must net it out — not a blacklist.
        for (var i = 0; i < 4; i++) eng.LearnIntentCue("2008 - 2009", "span");
        // ARITHMETIC the operator genuinely lives in — each subtraction emits "-" NEGATIVE to every intent. THIS is what
        // makes "-" self-exclude. (In the gym every example flows through LearnIntentCue, arithmetic included.)
        foreach (var (a, b) in new[] { (12, 1), (9, 4), (15, 6), (8, 3), (20, 7), (11, 2) })
            for (var i = 0; i < 3; i++)
                eng.LearnIntentCue($"{a} - {b}", (a - b).ToString());
        // GENUINE comparison: the discriminative cue + outcome vocabulary still learn and route.
        foreach (var (a, b) in new[] { (8, 3), (2, 9), (5, 1), (3, 7) })
            for (var i = 0; i < 3; i++)
                eng.LearnIntentCue($"{a} compared to {b}", a > b ? "greater" : "less");

        // "-" nets out ⇒ resolves to NO intent ⇒ subtraction computes (never hijacked to compare). NO special-case did this.
        _out.WriteLine($"intent '-'={eng.ResolveIntentForTests("-")}, 'compared'={eng.ResolveIntentForTests("compared")}");
        Assert.NotEqual(Compare, eng.ResolveIntentForTests("-"));
        var sub = eng.Generate(new GenerationRequest("12 - 1", 4));
        _out.WriteLine($"'12 - 1' path={sub.DecisionPath} -> '{sub.Output.Trim()}'");
        Assert.Equal("field-compute", sub.DecisionPath);
        Assert.Equal(11.0, double.Parse(sub.Output.Trim(), NumberStyles.Float, Inv), 6);

        // …a GENUINE comparison still routes to the compare path (discrimination is surgical, not a blanket disable).
        Assert.Equal(Compare, eng.ResolveIntentForTests("compared"));
        var cmp = eng.Generate(new GenerationRequest("8 compared to 3", 4));
        _out.WriteLine($"'8 compared to 3' path={cmp.DecisionPath} -> '{cmp.Output.Trim()}'");
        Assert.Equal("field-predicate", cmp.DecisionPath);
        Assert.Equal("greater", cmp.Output.Trim());
    }

    // DISCRIMINATION on ARBITRARY tokens (a non-operator case, so the "-" result is clearly a CONSEQUENCE, not a patch):
    // a PROMISCUOUS nonce that co-occurs with compare in a few examples but ALSO lives across arithmetic NETS OUT (never
    // a stable cue), while a nonce that CONSISTENTLY marks compare sticks.
    [Fact]
    public void Discrimination_PromiscuousNonce_NetsOut_While_ConsistentNonce_Sticks()
    {
        var eng = FreshEngine(out _, seed: 11);
        var pairs = new[] { (8, 3), (2, 9), (5, 1), (3, 7), (6, 2), (4, 9) };

        // CONSISTENT marker "qwib" — only ever a compare frame.
        foreach (var (a, b) in pairs)
            for (var i = 0; i < 3; i++)
                eng.LearnIntentCue($"{a} qwib {b}", a > b ? "greater" : "less");
        // PROMISCUOUS nonce "zlorp" — a few compare mislabels, but it ALSO lives across many arithmetic examples.
        foreach (var (a, b) in pairs)
            eng.LearnIntentCue($"{a} zlorp {b}", a > b ? "greater" : "less");
        foreach (var (a, b) in pairs)
            for (var i = 0; i < 3; i++)
                eng.LearnIntentCue($"zlorp {a} + {b}", (a + b).ToString());

        _out.WriteLine($"qwib -> intent {eng.ResolveIntentForTests("qwib")}, zlorp -> intent {eng.ResolveIntentForTests("zlorp")}");
        Assert.Equal(Compare, eng.ResolveIntentForTests("qwib"));      // consistent marker sticks
        Assert.NotEqual(Compare, eng.ResolveIntentForTests("zlorp"));  // promiscuous token nets out
    }

    // GENERAL SELF-HEAL on a DIFFERENT intent than compare: a to-WORD hijack self-corrects through the SAME structural
    // path. A contaminated cue "grok"→∘tow routes "grok 5" to the number-word route (spelling the digit → "five"); the
    // TRUE answer is a NUMBER, whose structure does not fit what to-word PRODUCES (a number-word) → the cue is disrupted.
    // Proves the self-heal is general (DecisionPath + true-answer structure), not an arithmetic→compare hardcode.
    [Fact]
    public void GeneralSelfHeal_HealsAToWordHijack_ThroughTheSamePath()
    {
        var eng = FreshEngine(out var space, seed: 5, selfHeal: true);
        for (var i = 0; i < 3; i++)
        {
            space.FineEditFromExample(new[] { "words" }, new[] { "∘tow" }, isNegativeExample: false); // genuine to-word cue
            space.FineEditFromExample(new[] { "grok" }, new[] { "∘tow" }, isNegativeExample: false);   // THE CONTAMINATION
        }

        // REPRODUCE THE HIJACK: "grok" routes "grok 5" to the to-word route, spelling the digit instead of the wanted number.
        var hijack = eng.Generate(new GenerationRequest("grok 5", 4));
        _out.WriteLine($"before: 'grok 5' path={hijack.DecisionPath} -> '{hijack.Output.Trim()}'");
        Assert.Equal("field-numberword", hijack.DecisionPath);
        Assert.Equal("five", hijack.Output.Trim());

        // SELF-HEAL: the SAME general signal — the true answer ("8", a NUMBER) does not fit to-word's produced shape.
        for (var pass = 0; pass < 4; pass++)
        {
            var r = eng.Generate(new GenerationRequest("grok 5", 4));
            if (r.Output.Trim() != "8") eng.HealMisroutedCue("grok 5", new[] { "8" }, r.Output, r.DecisionPath);
        }

        // HEALED: "grok" no longer selects the to-word route (the cue was unlearned by the general path).
        var healed = eng.Generate(new GenerationRequest("grok 5", 4));
        _out.WriteLine($"after:  'grok 5' path={healed.DecisionPath} -> '{healed.Output.Trim()}'");
        Assert.NotEqual("field-numberword", healed.DecisionPath);
        Assert.NotEqual("five", healed.Output.Trim());

        // SURGICAL: a GENUINE to-word query still routes and spells (only "grok" was unlearned, not "words").
        var genuine = eng.Generate(new GenerationRequest("5 in words", 4));
        _out.WriteLine($"genuine:'5 in words' path={genuine.DecisionPath} -> '{genuine.Output.Trim()}'");
        Assert.Equal("field-numberword", genuine.DecisionPath);
        Assert.Equal("five", genuine.Output.Trim());
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
                if (r.Output.Trim() != a) eng.HealMisroutedCue(q, new[] { a }, r.Output, r.DecisionPath);
            }

        // … and the hijack persists (the regimen can NOT recover — the very gap this feature closes).
        var still = eng.Generate(new GenerationRequest("12 - 1", 4));
        _out.WriteLine($"gate-off after loop: '12 - 1' path={still.DecisionPath} -> '{still.Output.Trim()}'");
        Assert.Equal("field-predicate", still.DecisionPath);
        Assert.Equal("greater", still.Output.Trim());
    }
}
