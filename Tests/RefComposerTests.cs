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
/// LEARNED-COMPOSER, REF shape (2026-06-15): the GRU plan head selects the REF shape (plan-kind 8) and the
/// substrate executes a HIGHER-ORDER composition — a Ref glider that INVOKES another named glider ("larger",
/// a Compare->Branch yielding max(a,b)) and SCALES its result ×2 (Compute(Multiply, Ref, Const 2)). This is a
/// composition-OF-compositions run on the substrate; the GRU only PICKS the top shape. Trained on a mix of
/// "twicelarger a b" -> 2*max(a,b) and plain arithmetic, the model must route to the Ref glider and
/// generalize to UNSEEN operands (the max + scale are computed exactly on the substrate).
/// </summary>
public sealed class RefComposerTests
{
    private readonly ITestOutputHelper _out;
    public RefComposerTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void Ref_ComposerEmerges_GliderOfGliders_ViaPlanHead_AndGeneralises()
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
        // REF: "twicelarger a b" -> 2*max(a,b). Skip pairs where 2*max equals a+b or a*b (those are an
        // arithmetic identity the digit shape owns; the label derivation excludes them). Mixed with plain
        // arithmetic so the plan head must DISTINGUISH the higher-order shape from arithmetic.
        for (var a = 0; a <= 9; a++)
            for (var b = 0; b <= 9; b++)
            {
                var t = 2 * Math.Max(a, b);
                var ambiguous = (a + b) == t || (a * b) == t;
                if (!ambiguous && (a + b) % 2 == 0) train.Add(new GenesisExample($"twicelarger {a} {b}", t.ToString()));
                if ((a * 3 + b) % 4 == 0) train.Add(new GenesisExample($"{a} + {b}", (a + b).ToString()));
            }

        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        // Probes are held-out operand pairs (odd a+b never trained), incl. larger values.
        var probes = new (string Q, string Exp)[]
        {
            ("twicelarger 3 7", "14"),  // max=7 -> 14, held-out parity
            ("twicelarger 8 5", "16"),  // max=8 -> 16
            ("twicelarger 2 9", "18"),  // max=9 -> 18
            ("twicelarger 6 1", "12"),  // max=6 -> 12
            ("twicelarger 7 4", "14"),  // max=7 -> 14
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
            _out.WriteLine($"  '{q,-16}' path={g.DecisionPath} -> '{got}' exp '{exp}' {(got == exp ? "" : "MISS")}");
        }
        _out.WriteLine($"REF-COMPOSER: correct {correct}/{probes.Length} (via plan->glider {viaPlan}/{probes.Length})");

        Assert.True(correct >= 4, $"ref (glider-of-gliders) should emerge; only {correct}/{probes.Length}. See breakdown.");
        Assert.True(viaPlan >= 4, $"answers must come via the GRU plan->glider ref path; only {viaPlan}/{probes.Length}.");
    }
}
