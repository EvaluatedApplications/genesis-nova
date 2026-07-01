using System;
using System.Collections.Generic;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// SEAM B — the self is now REINFORCED BY OUTCOME, not just passively accumulated (the neglected-backprop fix). The
/// audit found every _selfField write was outcome-blind: a right and a wrong conclusion shaped the self identically. Now
/// the grade stage pulls the self TOWARD a correct answer (TrainRetrievalToward) and PUSHES it away from a wrong one
/// (DisruptWrongAnswer). This proves it is LOAD-BEARING: with the geometry gradients OFF (so the SPACE is byte-identical
/// between runs), the ONLY thing that differs is the outcome-reinforced self — and it must FLIP an ambiguous query
/// toward whichever sense the outcomes reinforced. Ablate (SelfReinforcement off) → no reinforcement → no flip.
/// </summary>
public sealed class SelfReinforcementTests
{
    private readonly ITestOutputHelper _out;
    public SelfReinforcementTests(ITestOutputHelper o) => _out = o;

    private static DialecticalSpace BuildBankWorld(GenesisNovaConfig config)
    {
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) space.FineEditFromExample(new[] { a }, new[] { b }, false); }
        Rel("bank", "river"); Rel("bank", "money");                            // genuinely ambiguous
        Rel("river", "water"); Rel("river", "stream"); Rel("water", "stream"); // river sense
        Rel("money", "cash"); Rel("money", "coin"); Rel("cash", "coin");       // money sense
        return space;
    }

    // Shape the self by OUTCOMES ONLY (geometry gradients off → the space never changes), then ask the ambiguous query.
    // reinforceTarget is the CORRECT answer fed to TrainRetrievalToward; wrongTarget is fed to DisruptWrongAnswer.
    private static (string ans, string path) ReinforceAndAsk(
        DialecticalSpace space, string reinforceTarget, string wrongTarget, bool reinforceOn)
    {
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize));
        var mind = new GenesisInferenceEngine(tok, model, space, null)
        {
            ConsciousField = true, SelfConditionsCognition = true, SelfReinforcement = reinforceOn,
            FunctionGradientEnabled = false, FunctionDisruptionEnabled = false, // OUTCOME touches ONLY the self, never the space
        };
        mind.PerceiveSelf("bank"); // birth the self neutrally on the ambiguous anchor (ReinforceSelf no-ops before first perception)
        for (var i = 0; i < 6; i++)
        {
            mind.TrainRetrievalToward("what is a bank", new[] { reinforceTarget }); // correct → pull self toward this sense
            mind.DisruptWrongAnswer("what is a bank", wrongTarget);                 // wrong → push self away from the other
        }
        var r = mind.Generate(new GenerationRequest("what is a bank", 6));
        return (r.Output?.Trim() ?? "", r.DecisionPath);
    }

    [Fact]
    public void OutcomeReinforcedSelf_FlipsAmbiguousQuery_AndIsLoadBearing()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var riverSense = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "river", "water", "stream" };
        var moneySense = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "money", "cash", "coin" };

        // ON — same starting space, opposite outcomes. Reinforced toward river (money wrong) vs toward money (river wrong).
        var (riverAns, pathR) = ReinforceAndAsk(BuildBankWorld(config), "water", "cash", reinforceOn: true);
        var (moneyAns, pathM) = ReinforceAndAsk(BuildBankWorld(config), "cash", "water", reinforceOn: true);
        _out.WriteLine($"[reinforce ON ] river-outcomes -> '{riverAns}' [{pathR}]   money-outcomes -> '{moneyAns}' [{pathM}]");

        // OFF (ablation) — identical outcome calls, but SelfReinforcement off → the self is never updated by them.
        var (riverOff, _) = ReinforceAndAsk(BuildBankWorld(config), "water", "cash", reinforceOn: false);
        var (moneyOff, _) = ReinforceAndAsk(BuildBankWorld(config), "cash", "water", reinforceOn: false);
        _out.WriteLine($"[reinforce OFF] river-outcomes -> '{riverOff}'   money-outcomes -> '{moneyOff}'");

        // THE PROOF: with reinforcement, the SAME ambiguous query resolves to whichever sense the OUTCOMES reinforced.
        Assert.Contains(riverAns, riverSense);
        Assert.Contains(moneyAns, moneySense);
        Assert.NotEqual(riverAns, moneyAns);
        Assert.True(pathR == "field-relax-self" || pathM == "field-relax-self", "the reinforced self tipped the answer");

        // LOAD-BEARING: ablated, the identical outcome calls do NOT drive the flip (the self stayed the neutral birth).
        var ablationFlipped = riverSense.Contains(riverOff) && moneySense.Contains(moneyOff)
                              && !riverOff.Equals(moneyOff, StringComparison.OrdinalIgnoreCase);
        Assert.False(ablationFlipped, "without SelfReinforcement the outcomes must NOT flip the query — else the self is not what learned");
    }
}
