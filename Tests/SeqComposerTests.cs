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
/// LEARNED-COMPOSER, SEQ shape (2026-06-15): the GRU plan head selects the SEQ shape (plan-kind 7) and the
/// substrate assembles a CONCATENATE-COMPOSITION — a cached scaffold chunk ("the answer is") bound to a
/// SUBSTRATE-COMPUTED value (Compute(Add) → one R2 compose / face homomorphism) — joining them via the
/// interpreter's Seq block into "the answer is N". The GRU only PICKS the shape; the chunk is a known phrase
/// (not a per-query answer table) and the value is computed on the substrate. Trained on a mix of seq +
/// plain arithmetic, the model must route "answer a b" to the Seq glider and generalize to UNSEEN operands.
/// </summary>
public sealed class SeqComposerTests
{
    private readonly ITestOutputHelper _out;
    public SeqComposerTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Seq_ComposerEmerges_ScaffoldPlusComputed_ViaPlanHead_AndGeneralises()
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
        // SEQ: "answer a b" -> "the answer is {a+b}", operands in 0..9 (held-out combos probed below).
        // Mixed with plain digit-arithmetic so the plan head must DISTINGUISH the seq shape from arithmetic.
        for (var a = 0; a <= 9; a++)
            for (var b = 0; b <= 9; b++)
            {
                if ((a + b) % 2 == 0) train.Add(new GenesisExample($"answer {a} {b}", $"the answer is {a + b}"));
                if ((a * 3 + b) % 4 == 0) train.Add(new GenesisExample($"{a} + {b}", (a + b).ToString()));
            }

        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        // Probes include operand pairs whose (a+b) parity excluded them from training (held-out), and large
        // operands the substrate computes exactly.
        var probes = new (string Q, string Exp)[]
        {
            ("answer 2 3", "the answer is 5"),   // a+b odd -> excluded from training
            ("answer 4 5", "the answer is 9"),   // held-out
            ("answer 7 6", "the answer is 13"),  // held-out
            ("answer 8 9", "the answer is 17"),  // held-out, large
            ("answer 6 6", "the answer is 12"),
        };
        string Gen(string q) => inference.Generate(new GenerationRequest(q, 6)).Output.Trim();
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
            var g = inference.Generate(new GenerationRequest(q, 6));
            var got = g.Output.Trim();
            if (got == exp) correct++;
            if (g.DecisionPath.Contains("glider-plan", StringComparison.OrdinalIgnoreCase)) viaPlan++;
            _out.WriteLine($"  '{q,-14}' path={g.DecisionPath} -> '{got}' exp '{exp}' {(got == exp ? "" : "MISS")}");
        }
        _out.WriteLine($"SEQ-COMPOSER: correct {correct}/{probes.Length} (via plan->glider {viaPlan}/{probes.Length})");

        Assert.True(correct >= 4, $"seq composition should emerge; only {correct}/{probes.Length}. See breakdown.");
        Assert.True(viaPlan >= 4, $"answers must come via the GRU plan->glider seq path; only {viaPlan}/{probes.Length}.");
    }
}
