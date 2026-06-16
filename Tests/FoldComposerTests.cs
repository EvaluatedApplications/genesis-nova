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
/// LEARNED-COMPOSER, FOLD shape (2026-06-15): the GRU plan head selects fold-sum / fold-product and the
/// substrate REDUCES an arbitrary-length operand list via ONE N-way R2 compose (poly-sum = +, log-sum = ×) —
/// the variadic ability the 2-operand routes can't express, with the reduce computed entirely in the
/// platonic space (the GRU only picks the shape). Trained on a sum/product mix, the model must route
/// "sum a b c" / "product a b c" to the fold glider and answer correctly, generalizing to unseen operands.
/// </summary>
public sealed class FoldComposerTests
{
    private readonly ITestOutputHelper _out;
    public FoldComposerTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void Fold_ComposerEmerges_ReducesVariadically_ViaPlanHead()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null);
        trainer.SetInferencePolicy(inference);

        var rng = new Random(7);
        var train = new List<GenesisExample>();
        // fold-sum: 3 operands in 1..6. fold-product: 3 operands in 1..3 (positive, small → log face decodes).
        for (var k = 0; k < 200; k++)
        {
            var a = rng.Next(1, 7); var b = rng.Next(1, 7); var c = rng.Next(1, 7);
            train.Add(new GenesisExample($"sum {a} {b} {c}", (a + b + c).ToString()));
            var p = rng.Next(1, 4); var q = rng.Next(1, 4); var r = rng.Next(1, 4);
            train.Add(new GenesisExample($"product {p} {q} {r}", (p * q * r).ToString()));
        }

        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        var probes = new (string Q, string Exp)[]
        {
            ("sum 2 3 4", "9"), ("sum 5 1 2", "8"), ("sum 6 6 2", "14"),
            ("product 2 3 2", "12"), ("product 3 3 3", "27"),
        };
        string Gen(string q) => inference.Generate(new GenerationRequest(q, 4)).Output.Trim();
        int Correct() => probes.Count(p => Gen(p.Q) == p.Exp);

        var pool = train.ToList();
        Shuffle(pool);
        var idx = 0;
        var steps = 0;
        const int maxSteps = 14000;
        const int probeEvery = 700;
        for (var s = 0; s < maxSteps; s++)
        {
            if (idx >= pool.Count) { Shuffle(pool); idx = 0; }
            trainer.TrainStep(pool[idx++]);
            steps++;
            if (steps % probeEvery == 0 && Correct() >= probes.Length) break;
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
        _out.WriteLine($"FOLD-COMPOSER: correct {correct}/{probes.Length} (via plan→glider {viaPlan}/{probes.Length})");

        Assert.True(correct >= 4, $"variadic fold should emerge; only {correct}/{probes.Length}. See breakdown.");
        Assert.True(viaPlan >= 4, $"answers must come via the GRU plan→glider fold path; only {viaPlan}/{probes.Length}.");
    }
}
