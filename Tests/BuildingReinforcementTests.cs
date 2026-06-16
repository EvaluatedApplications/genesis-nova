using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Emergence guard for the SPACE-BUILDING reinforcement (2026-06-16). The edit head is now rewarded by the
/// WITHIN-STEP, CONTRASTIVE, BIDIRECTIONAL retrievability DELTA its write causes (replacing a per-string
/// baseline the answer-order shuffling defeated). This trains a few plain relational associations and asserts
/// they (a) retrieve their target and (b) at least one ROUTES via the PLATONIC substrate (not neural fallback)
/// — i.e. the building tools actually construct a retrievable, routed relation rather than memorizing in the
/// decoder. [SlowFact] (opt-in RUN_SLOW=1) because it trains.
/// </summary>
public sealed class BuildingReinforcementTests
{
    private readonly ITestOutputHelper _out;
    public BuildingReinforcementTests(ITestOutputHelper output) => _out = output;

    [SlowFact]
    public void EditHeadReward_BuildsRetrievableRoutedAssociations()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 11);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null);
        trainer.SetInferencePolicy(inference);

        // Plain, non-arithmetic associations — the relational/retrieval modality the daemon trains.
        var pairs = new[]
        {
            ("apple", "fruit"),
            ("sparrow", "bird"),
            ("copper", "metal"),
            ("violet", "color"),
        };
        var examples = pairs.Select(p => new GenesisExample(p.Item1, p.Item2)).ToList();

        var rng = new Random(11);
        for (var epoch = 0; epoch < 60; epoch++)
        {
            for (var i = examples.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (examples[i], examples[j]) = (examples[j], examples[i]); }
            foreach (var ex in examples) trainer.TrainStep(ex);
        }

        var retrieved = 0;
        var routedPlatonic = 0;
        foreach (var (input, expected) in pairs)
        {
            var r = inference.Generate(new GenerationRequest(input, 4));
            var ok = AnswerEquivalence.Equivalent(r.Output, expected);
            var platonic = r.UsedPlatonicQuery && !r.UsedNeuralFallback;
            if (ok) retrieved++;
            if (ok && platonic) routedPlatonic++;
            _out.WriteLine($"{input} -> '{r.Output}' (want '{expected}') ok={ok} platonic={platonic} path={r.DecisionPath}");
        }

        // Building works: a majority retrieve, and at least one is genuinely ROUTED through the substrate
        // (not just memorized by the neural decoder) — the capability the new edit-head reward is meant to grow.
        Assert.True(retrieved >= 3, $"only {retrieved}/4 associations retrieved their target");
        Assert.True(routedPlatonic >= 1, $"no association routed via the platonic path ({routedPlatonic}/4)");
    }
}
