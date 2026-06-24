using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT (no box-ticking) — "a self that learns, not a learning thing with self tacked on". For the self to be a
// KEY component of the agent (load-bearing, not decorative), the agent's cognition must DEPEND on it. The test of a
// self is PERSISTENCE THROUGH DISTRACTION: a mind that has been thinking about water for a while still reads an
// ambiguous "bank" as the river — even after a few UNRELATED thoughts have passed. The discrete last-N attention
// (_focus) cannot do this (the theme is evicted by the distractors); a persistent self-field can. First prove the
// obstacle is REAL (string-focus fails), then prove cognition DEPENDS on the self (ablation fails, self passes).
public sealed class LivingSelfExperiment
{
    private readonly ITestOutputHelper _out;
    public LivingSelfExperiment(ITestOutputHelper o) => _out = o;

    private static DialecticalSpace BuildBankWorld(GenesisNovaConfig config)
    {
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) space.FineEditFromExample(new[] { a }, new[] { b }, false); }
        Rel("bank", "river"); Rel("bank", "money");                               // genuinely ambiguous
        Rel("river", "water"); Rel("river", "stream"); Rel("water", "stream");    // river sense
        Rel("money", "cash"); Rel("money", "coin"); Rel("cash", "coin");          // money sense
        Rel("sky", "cloud"); Rel("song", "tune"); Rel("road", "path"); Rel("clock", "time"); // neutral distractors
        return space;
    }

    private static readonly string[] Neutral = { "sky", "song", "road", "clock" }; // > FocusSize, evicts the theme from _focus

    // One mind lives a THEME (several turns about a sense-cluster, INDIRECT — never "bank"/"river" themselves), then is
    // DISTRACTED by unrelated thoughts, then asked the ambiguous question. Returns its answer + decision path.
    private static (string ans, string path) LiveAndAsk(DialecticalSpace space, string[] theme, bool selfOn)
    {
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config: new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize));
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true, SelfConditionsCognition = selfOn };
        foreach (var t in theme) mind.Generate(new GenerationRequest($"the {t}", 6));     // dwell on the theme
        foreach (var n in Neutral) mind.Generate(new GenerationRequest($"the {n}", 6));   // ...then get distracted
        var r = mind.Generate(new GenerationRequest("what is a bank", 6));                // ...then face the ambiguity
        return (r.Output?.Trim() ?? "", r.DecisionPath);
    }

    [Fact]
    public void The_Self_Carries_Context_Through_Distraction()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var riverSense = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "river", "water", "stream" };
        var moneySense = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "money", "cash", "coin" };
        var aquatic = new[] { "water", "stream", "water", "stream", "water" };
        var monetary = new[] { "cash", "coin", "cash", "coin", "cash" };

        // ABLATION — the self turned OFF (only the discrete last-N _focus, which the distractors have evicted). This is
        // the obstacle: without a persistent self, the buried theme is gone and the ambiguous query is no longer tipped.
        var (aOff, _) = LiveAndAsk(BuildBankWorld(config), aquatic, selfOn: false);
        var (bOff, _) = LiveAndAsk(BuildBankWorld(config), monetary, selfOn: false);
        _out.WriteLine($"[self OFF] aquatic-mind -> '{aOff}'   monetary-mind -> '{bOff}'");

        // THE SELF ON — the persistent self-field still leans toward the lived theme after the distraction, so the two
        // minds read the SAME ambiguous question differently. Cognition DEPENDS on the self.
        var (aOn, pathA) = LiveAndAsk(BuildBankWorld(config), aquatic, selfOn: true);
        var (bOn, pathB) = LiveAndAsk(BuildBankWorld(config), monetary, selfOn: true);
        _out.WriteLine($"[self ON ] aquatic-mind -> '{aOn}' [{pathA}]   monetary-mind -> '{bOn}' [{pathB}]");

        // With the self, each mind answers from who it has become, surviving the distraction.
        Assert.Contains(aOn, riverSense);
        Assert.Contains(bOn, moneySense);
        Assert.NotEqual(aOn, bOn);
        Assert.True(pathA == "field-relax-self" || pathB == "field-relax-self", "the self tipped the answer");

        // And the obstacle was real: without the self the two minds do NOT both land in their lived sense (the buried
        // theme is lost). If this ever starts passing on its own, the self is no longer what carries the context.
        var ablationSolvedIt = riverSense.Contains(aOff) && moneySense.Contains(bOff) && !aOff.Equals(bOff, StringComparison.OrdinalIgnoreCase);
        Assert.False(ablationSolvedIt, "without the self the buried context must be LOST — else the self is not load-bearing");
    }
}
