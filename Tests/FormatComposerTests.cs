using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// LEARNED-COMPOSER, increment 2 (2026-06-14): the GRU plan head selects the arithmetic→WORD shape and the
/// substrate executes Hop(Compute(op, operands), Word) — computing the sum AND formatting it as its
/// number-word via the learned digit↔word edge. This is a genuinely NEW capability the digit-only routes
/// cannot produce, assembled from blocks by a learned decision (no hardcoded formatter). Trained on
/// number-word equivalence (so the digit→word edges exist) + arithmetic with word outputs, the model must
/// emit "nine" for held-out "4 + 5" via the plan→glider path.
/// </summary>
public sealed class FormatComposerTests
{
    private readonly ITestOutputHelper _out;
    public FormatComposerTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void ArithmeticToWord_ComposerEmerges_FormatsResultAsWord()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null);
        trainer.SetInferencePolicy(inference);

        var v2w = NumberWordVocabulary.Entries.ToDictionary(e => e.Value, e => e.Word);
        string Word(int n) => v2w[n];

        var train = new List<GenesisExample>();
        // (a) number-word equivalence — establishes the digit↔word relation edges Hop-to-word needs.
        foreach (var (i, o) in new NumberWordCreator().Generate(200, 0, true))
            train.Add(new GenesisExample(i, o, SourceCreatorName: "corenova:number-word-equiv"));
        // (b) arithmetic with WORD output (the new shape) + (c) digit output (so the plan head distinguishes).
        for (var a = 0; a <= 9; a++)
            for (var b = 0; b <= 9; b++)
            {
                if (!v2w.ContainsKey(a + b)) continue;
                if ((a + b) % 2 == 0) train.Add(new GenesisExample($"{a} + {b}", Word(a + b)));  // → word
                else train.Add(new GenesisExample($"{a} + {b}", (a + b).ToString()));            // → digit
            }

        var rng = new Random(7);
        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        // Held-out word-output sums (pairs not necessarily trained); answer must be the WORD via the plan path.
        var probes = new (string Q, string Exp)[]
        {
            ("4 + 5", "nine"), ("6 + 6", "twelve"), ("7 + 2", "nine"), ("3 + 5", "eight"),
        };
        string Gen(string q) => inference.Generate(new GenerationRequest(q, 4)).Output.Trim();
        int Correct() => probes.Count(p => Gen(p.Q) == p.Exp);

        var pool = train.ToList();
        Shuffle(pool);
        var idx = 0;
        var steps = 0;
        const int maxSteps = 16000;
        const int probeEvery = 800;
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
            _out.WriteLine($"  '{q,-10}' path={g.DecisionPath} -> '{got}' exp '{exp}' {(got == exp ? "" : "MISS")}");
        }
        _out.WriteLine($"FORMAT-COMPOSER: correct {correct}/{probes.Length} (via plan→glider {viaPlan}/{probes.Length})");

        Assert.True(correct >= 3, $"arithmetic→word should emerge; only {correct}/{probes.Length}. See breakdown.");
        Assert.True(viaPlan >= 3, $"answers must come via the GRU plan→glider path; only {viaPlan}/{probes.Length}.");
    }
}
