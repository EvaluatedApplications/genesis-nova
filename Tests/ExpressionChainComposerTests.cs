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
/// LEARNED-COMPOSER, EXPRESSION-CHAIN shape (plan-kind 8) — the GENERAL "complex chaining of elements"
/// capability, with NO locked cue token (see nova-no-token-locking). A MULTI-operator expression
/// "2 x 7 + 3" is solved by CHAINING compute-elements: the plan head selects the expression-chain shape;
/// the route classifies EACH operator from CONTEXT via the learned op head (so "x" means multiply only
/// because of the operands around it — no symbol→op map); it evaluates with precedence (× before +) where
/// every binary step is one substrate R2 compose + homomorphic decode. It must generalise to UNSEEN
/// operands (the probes are held out) — the chain COMPUTES, it does not memorise. Trained mixed with plain
/// binary arithmetic (so the op head is reliable per-operator) and the plan head must DISTINGUISH a
/// multi-operator expression from a single binary op.
/// </summary>
public sealed class ExpressionChainComposerTests
{
    private readonly ITestOutputHelper _out;
    public ExpressionChainComposerTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void ExpressionChain_ComposerEmerges_MultiOperatorPrecedence_ViaPlanHead_AndGeneralises()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null);
        trainer.SetInferencePolicy(inference);

        // Held-out probe expressions — never trained; the chain must COMPUTE them with precedence.
        var probes = new (string Q, string Exp)[]
        {
            ("2 x 7 + 3", "17"),  // (2*7)+3
            ("3 + 4 x 5", "23"),  // 3+(4*5)
            ("8 x 2 - 5", "11"),  // (8*2)-5
            ("6 x 3 + 4", "22"),  // (6*3)+4
            ("9 + 2 x 4", "17"),  // 9+(2*4)
        };
        var heldOut = new HashSet<string>(probes.Select(p => p.Q));

        var rng = new Random(7);
        var train = new List<GenesisExample>();
        // Plain BINARY arithmetic (heavy) so the op head classifies each operator reliably from context.
        for (var a = 0; a <= 12; a++)
            for (var b = 0; b <= 12; b++)
            {
                train.Add(new GenesisExample($"{a} + {b}", (a + b).ToString()));
                if (b <= a) train.Add(new GenesisExample($"{a} - {b}", (a - b).ToString()));
                if (a >= 1 && b >= 1) train.Add(new GenesisExample($"{a} x {b}", (a * b).ToString()));
            }
        // MULTI-operator expressions (mixed precedence) — the chain. Generated over small operands; the
        // five probe expressions are held OUT so success proves the chain computes, not memorises.
        for (var a = 1; a <= 6; a++)
            for (var b = 1; b <= 6; b++)
                for (var c = 0; c <= 6; c++)
                {
                    void Add(string q, int v) { if (!heldOut.Contains(q)) train.Add(new GenesisExample(q, v.ToString())); }
                    Add($"{a} x {b} + {c}", a * b + c);
                    Add($"{c} + {a} x {b}", c + a * b);
                    if (c <= a * b) Add($"{a} x {b} - {c}", a * b - c);
                }

        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        string Gen(string q) => inference.Generate(new GenerationRequest(q, 4)).Output.Trim();
        int Correct() => probes.Count(p => Gen(p.Q) == p.Exp);

        var pool = train.ToList();
        Shuffle(pool);
        var idx = 0;
        var steps = 0;
        const int maxSteps = 12000;
        const int probeEvery = 400;
        for (var s = 0; s < maxSteps; s++)
        {
            if (idx >= pool.Count) { Shuffle(pool); idx = 0; }
            trainer.TrainStep(pool[idx++]);
            steps++;
            if (steps % probeEvery == 0)
            {
                var c = Correct();
                Console.Error.WriteLine($"[expr] step {steps}/{maxSteps} correct {c}/{probes.Length}");
                if (c >= probes.Length) break;
            }
        }
        _out.WriteLine($"trained {steps}/{maxSteps} steps over {train.Count} examples");

        var correct = 0;
        var viaChain = 0;
        foreach (var (q, exp) in probes)
        {
            var g = inference.Generate(new GenerationRequest(q, 4));
            var got = g.Output.Trim();
            if (got == exp) correct++;
            if (g.DecisionPath.Contains("expression-chain", StringComparison.OrdinalIgnoreCase)) viaChain++;
            _out.WriteLine($"  '{q,-12}' path={g.DecisionPath} hops={g.PlatonicHopCount} -> '{got}' exp '{exp}' {(got == exp ? "" : "MISS")}");
        }
        _out.WriteLine($"EXPRESSION-CHAIN: correct {correct}/{probes.Length} (via plan->expression-chain {viaChain}/{probes.Length})");

        Assert.True(correct >= 4, $"multi-operator expression chaining should emerge; only {correct}/{probes.Length}.");
        Assert.True(viaChain >= 4, $"answers must come via the GRU plan->expression-chain route; only {viaChain}/{probes.Length}.");
    }
}
