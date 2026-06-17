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
/// GENERALIZED transform learning (was arithmetic-gated): a NON-arithmetic unary function "fn x → k·x" taught
/// from examples is learned as a transform vector T(fn) by the TransformAccumulator (UpdateTransformDiscovery),
/// the router learns to send it to the learned-function route (ResolveRouteLabel reproduces-via-transform), and
/// inference APPLIES it to a HELD-OUT operand — generalization, not memorisation. Production dims.
/// </summary>
public sealed class LearnedFunctionTrainingTests
{
    private readonly ITestOutputHelper _out;
    public LearnedFunctionTrainingTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void UnaryFunction_LearnedFromTraining_RoutesAndGeneralises()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, model, memory, config);
        var inference = new GenesisInferenceEngine(tok, model, memory, null,
            transformAccumulator: trainer.TransformAccumulator, foldPathDiscovery: trainer.FoldPathDiscovery);
        trainer.SetInferencePolicy(inference);

        // "fn" is a nonce function = ×3 (multiplicative → constant translation on the LOG face → generalizes).
        var trainOperands = new[] { 2, 3, 4, 5, 6, 7, 8 };           // held out: 9, 11
        var probes = new (string Q, string Exp)[] { ("fn 9", "27"), ("fn 11", "33") };

        string Gen(string q) => inference.Generate(new GenerationRequest(q, 4)).Output.Trim();
        int Correct() => probes.Count(p => Gen(p.Q) == p.Exp);

        var steps = 0;
        const int maxPasses = 60;
        for (var pass = 1; pass <= maxPasses; pass++)
        {
            foreach (var x in trainOperands)
            {
                trainer.TrainStep(new GenesisExample($"fn {x}", (3 * x).ToString()));
                steps++;
            }
            if (pass % 6 == 0)
            {
                var c = Correct();
                Console.Error.WriteLine($"[fn] pass {pass} steps {steps} correct {c}/{probes.Length}  T(fn)={trainer.TransformAccumulator.TryGetTransform("fn", out _)}");
                if (c >= probes.Length) break;
            }
        }

        // 1. the transform was LEARNED via the generalized (non-arithmetic) path.
        Assert.True(trainer.TransformAccumulator.TryGetTransform("fn", out _), "T(fn) should be learned from the function examples");

        // 2. inference applies it to HELD-OUT operands via the learned-function route.
        var correct = 0; var viaFn = 0;
        foreach (var (q, exp) in probes)
        {
            var g = inference.Generate(new GenerationRequest(q, 4));
            if (g.Output.Trim() == exp) correct++;
            if (g.DecisionPath.Contains("learned-function", StringComparison.OrdinalIgnoreCase)) viaFn++;
            _out.WriteLine($"  '{q}' path={g.DecisionPath} -> '{g.Output.Trim()}' exp '{exp}'");
        }
        Assert.True(correct >= probes.Length, $"learned unary function should generalize to held-out operands; {correct}/{probes.Length}");
        Assert.True(viaFn >= probes.Length, $"answers must come via the learned-function route; {viaFn}/{probes.Length}");
    }
}
