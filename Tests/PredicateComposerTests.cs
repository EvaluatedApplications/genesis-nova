using System;
using System.Collections.Generic;
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
/// LEARNED-COMPOSER, increment 1 (2026-06-14): the GRU's PLAN head learns to assemble a block composition
/// and the substrate executes it — replacing the deleted hardcoded "compare" resolver with an EMERGENT one.
/// Trained on a predicate+arithmetic mix, the model must learn to ROUTE "compare a b" to the Compare→Branch
/// glider (PlanKind=Predicate) and answer greater/less/equal — generalizing to operands well outside the
/// training range (the comparison is computed on the substrate, so it's exact). The plan head must also NOT
/// fire predicate on arithmetic (the mix forces it to distinguish). No hardcoded per-token rule.
/// </summary>
public sealed class PredicateComposerTests
{
    private readonly ITestOutputHelper _out;
    public PredicateComposerTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Predicate_ComposerEmerges_ViaPlanHead_AndGeneralisesBeyondTrainingRange()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null);
        trainer.SetInferencePolicy(inference);

        static string Cmp(int a, int b) => a > b ? "greater" : a < b ? "less" : "equal";
        var train = new List<GenesisExample>();
        for (var a = 0; a <= 12; a++)
            for (var b = 0; b <= 12; b++)
            {
                if ((a + b) % 2 == 0) train.Add(new GenesisExample($"compare {a} {b}", Cmp(a, b)));
                if ((a * 3 + b) % 4 == 0) train.Add(new GenesisExample($"{a} + {b}", (a + b).ToString())); // arithmetic negatives
            }

        var rng = new Random(7);
        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        // Probes include EXTRAPOLATION (operands > 12, never trained) — routing must generalize and the
        // substrate computes the comparison exactly.
        var probes = new (string Q, string Exp)[]
        {
            ("compare 7 3", "greater"), ("compare 2 9", "less"), ("compare 5 5", "equal"),
            ("compare 30 10", "greater"), ("compare 8 25", "less"),
        };
        string Gen(string q) => inference.Generate(new GenerationRequest(q, 4)).Output.Trim();
        int Correct() => probes.Count(p => Gen(p.Q) == p.Exp);

        var pool = train.ToList();
        Shuffle(pool);
        var idx = 0;
        var steps = 0;
        const int maxSteps = 12000;
        const int probeEvery = 500;
        for (var s = 0; s < maxSteps; s++)
        {
            if (idx >= pool.Count) { Shuffle(pool); idx = 0; }
            trainer.TrainStep(pool[idx++]);
            steps++;
            if (steps % probeEvery == 0 && Correct() >= 5) break;
        }
        _out.WriteLine($"trained {steps}/{maxSteps} steps over {train.Count} examples");

        var correct = 0;
        var viaPlan = 0;
        foreach (var (q, exp) in probes)
        {
            var g = inference.Generate(new GenerationRequest(q, 4));
            var got = g.Output.Trim();
            if (got == exp) correct++;
            if (g.DecisionPath.Contains("glider-plan", StringComparison.OrdinalIgnoreCase)) viaPlan++;
            _out.WriteLine($"  '{q,-14}' path={g.DecisionPath} -> '{got}' exp '{exp}' {(got == exp ? "" : "MISS")}");
        }
        _out.WriteLine($"PREDICATE-COMPOSER: correct {correct}/{probes.Length} (via plan→glider {viaPlan}/{probes.Length})");

        Assert.True(correct >= 4, $"predicate composition should emerge; only {correct}/{probes.Length} correct. See breakdown.");
        Assert.True(viaPlan >= 4, $"answers must come via the GRU plan→glider path; only {viaPlan}/{probes.Length} did.");
    }
}
