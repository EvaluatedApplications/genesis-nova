using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// LEARNED-FUNCTION ROUTE (2026-06-14). A function learned as a transform vector T(f)=avg(embed(out)-
/// embed(in)) is a first-class element of the space, APPLIED BY COMPOSITION (embed(x)+T(f)), and SELECTED
/// FROM THE SPACE by following a learned RELATION from a cue concept — not parsed by name, not a fixed op
/// vocabulary. The GRU's route head gates the path. This verifies the WIRED route end-to-end through
/// Generate: two functions learned from a FEW examples generalise to HELD-OUT operands, and a synonym cue
/// reaches its function through the learned relation edge. Novel cue words (not glider capabilities) are
/// used so the learned route — not the hand-built glider — is what answers.
/// </summary>
public sealed class LearnedFunctionRouteTests
{
    private readonly ITestOutputHelper _out;
    public LearnedFunctionRouteTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void LearnedFunction_GeneralisesToHeldOutOperands_ViaCompositionAndRelationalSelection()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var dim = memory.FaceDimension;
        var transforms = new TransformAccumulator(dim);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null, transforms);

        // Learn two functions from a FEW examples each — the transform IS the learning (no gradient descent).
        // "flonk" = ×2 (constant in the LOG face); "grix" = +5 (constant in the POLY face). Novel names so the
        // hand-built glider (which knows double/triple) abstains and the LEARNED route does the work.
        void Learn(string fn, Func<long, long> f, long[] train)
        {
            foreach (var n in train)
                transforms.Learn(fn, PlatonicFaceComposer.Compose(n.ToString(), dim),
                                     PlatonicFaceComposer.Compose(f(n).ToString(), dim));
        }
        Learn("flonk", n => n * 2, new long[] { 2, 3, 5, 8 });
        Learn("grix", n => n + 5, new long[] { 1, 3, 4, 8 });

        // The functions are elements of the space; relate a SYNONYM cue so selection is by learned relation,
        // not literal name: "doubler" → "flonk".
        for (var i = 0; i < 8; i++) memory.ObserveContradiction("doubler", "flonk", 0.0);

        // Train ONLY the GRU route head to route these function queries platonic-direct (route 1). All labels
        // are route 1, so the head learns to send "<cue> <number>" to the platonic path; it generalises the
        // ROUTING to held-out operands. The ANSWER comes from the transform, not the token decoder.
        var routeTrain = new List<(string Input, string Output)>();
        foreach (var n in new long[] { 2, 3, 4, 5, 6, 8 })
        {
            routeTrain.Add(($"flonk {n}", (n * 2).ToString()));
            routeTrain.Add(($"grix {n}", (n + 5).ToString()));
            routeTrain.Add(($"doubler {n}", (n * 2).ToString()));
        }

        // HELD-OUT operands (none used to learn the transforms): the transform must GENERALISE.
        var probes = new (string Query, string Expected)[]
        {
            ("flonk 7", "14"), ("flonk 11", "22"), ("flonk 20", "40"),  // ×2 via log face
            ("grix 10", "15"), ("grix 12", "17"),                       // +5 via poly face
            ("doubler 9", "18"),                                        // ×2 reached via learned relation
        };

        // Warm the tokenizer vocab on every token used, then size the model's embedding table to match — the
        // trainer does this before each TrainExample; without it new token ids overflow the table (CUDA OOB).
        foreach (var (inp, outp) in routeTrain) { tokenizer.Encode(inp); tokenizer.Encode(outp, addEos: true); }
        foreach (var (q, e) in probes) { tokenizer.Encode(q); tokenizer.Encode(e); }
        model.EnsureVocabularySize(tokenizer.VocabularySize);

        var rng = new Random(11);
        bool RoutesPlatonic(string q) => model.PredictRoute(tokenizer.Encode(q)).RouteId == 1;
        var routed = 0;
        for (var step = 0; step < 1500; step++)
        {
            var ex = routeTrain[rng.Next(routeTrain.Count)];
            model.TrainExample(tokenizer.Encode(ex.Input), tokenizer.Encode(ex.Output, addEos: true),
                tokenizer.BosTokenId, routeLabel: 1);
            model.CloneParametersToBreakGraph();
            if (step % 50 == 0 && step > 0 && RoutesPlatonic("flonk 7") && RoutesPlatonic("grix 7") && RoutesPlatonic("doubler 7"))
            {
                routed = step;
                break;
            }
        }
        _out.WriteLine($"route head routes function queries platonic after {routed} steps");

        var correct = 0;
        var viaRoute = 0;
        foreach (var (q, exp) in probes)
        {
            var g = inference.Generate(new GenerationRequest(q, 4));
            var got = g.Output.Trim();
            var ok = got == exp;
            if (ok) correct++;
            if (g.DecisionPath.Contains("platonic-learned-function", StringComparison.OrdinalIgnoreCase)) viaRoute++;
            _out.WriteLine($"  '{q,-12}' path={g.DecisionPath} -> '{got}' exp '{exp}' {(ok ? "" : "MISS")}");
        }
        _out.WriteLine($"LEARNED-FUNCTION: correct {correct}/{probes.Length} (via learned-function route {viaRoute}/{probes.Length})");

        // The function GENERALISES to unseen operands AND is answered by the learned-function route (the
        // composition+relational-selection path), not memorised by the decoder or intercepted by a glider.
        Assert.True(correct >= probes.Length - 1, $"learned functions should generalise; only {correct}/{probes.Length}. See breakdown.");
        Assert.True(viaRoute >= probes.Length - 1, $"answers must come via the learned-function route; only {viaRoute}/{probes.Length}.");
    }

    [Fact]
    public void LearnedBinaryOp_DiscoveredStructure_GeneralisesToHeldOutOperands_ViaTheRoute()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var folds = new FoldPathDiscovery();
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null,
            transformAccumulator: null, foldPathDiscovery: folds);

        // Learn a NOVEL binary op "widget" = a*b from examples — FoldPathDiscovery DISCOVERS that it is a
        // FOLD of a known base op (multiply = repeated add); nothing hardcodes that "widget" means multiply.
        foreach (var (a, b) in new[] { (2, 3), (3, 4), (2, 5), (4, 6), (3, 7), (5, 2) })
            folds.ObserveTrainingPair("widget", a, b, a * b);
        foreach (var (a, b) in new[] { (1, 1), (2, 2), (3, 1) }) // a known base op for the fold search
            folds.ObserveTrainingPair("add", a, b, a + b);
        folds.TryRunDiscovery("widget", (op, x, y) => op == "add" ? x + y : (double?)null);
        Assert.True(folds.HasOperation("widget"), "FoldPathDiscovery should have discovered the 'widget' op");
        Assert.True(folds.TryPredict("widget", 6, 7, out var pv, out _) && Math.Abs(pv - 42) < 0.5,
            "FoldPathDiscovery should predict widget(6,7)=42 from the discovered structure");

        // Route head → platonic for "widget a b" (only the routing is learned; the op head stays abstain,
        // so the GRU-query route doesn't intercept and the binary learned-op route answers).
        var routeTrain = new List<(string, string)>();
        foreach (var (a, b) in new[] { (2, 3), (3, 4), (4, 6), (5, 2) })
            routeTrain.Add(($"widget {a} {b}", (a * b).ToString()));
        foreach (var (inp, outp) in routeTrain) { tokenizer.Encode(inp); tokenizer.Encode(outp, addEos: true); }
        var heldQueries = new[] { "widget 6 7", "widget 8 3", "widget 9 4" };
        foreach (var q in heldQueries) tokenizer.Encode(q);
        model.EnsureVocabularySize(tokenizer.VocabularySize);

        var rng = new Random(13);
        for (var step = 0; step < 1500; step++)
        {
            var (inp, outp) = routeTrain[rng.Next(routeTrain.Count)];
            model.TrainExample(tokenizer.Encode(inp), tokenizer.Encode(outp, addEos: true), tokenizer.BosTokenId, routeLabel: 1);
            model.CloneParametersToBreakGraph();
            if (step % 50 == 0 && step > 0 && heldQueries.All(q => model.PredictRoute(tokenizer.Encode(q)).RouteId == 1))
                break;
        }

        var probes = new (string Query, string Expected)[] { ("widget 6 7", "42"), ("widget 8 3", "24"), ("widget 9 4", "36") };
        var correct = 0;
        var viaRoute = 0;
        foreach (var (q, exp) in probes)
        {
            var g = inference.Generate(new GenerationRequest(q, 4));
            if (g.Output.Trim() == exp) correct++;
            if (g.DecisionPath.Contains("platonic-learned-op", StringComparison.OrdinalIgnoreCase)) viaRoute++;
            _out.WriteLine($"  '{q,-12}' path={g.DecisionPath} -> '{g.Output.Trim()}' exp '{exp}'");
        }
        _out.WriteLine($"LEARNED-BINARY-OP: correct {correct}/{probes.Length} (via route {viaRoute}/{probes.Length})");

        Assert.True(correct >= probes.Length - 1, $"discovered binary op should generalise; only {correct}/{probes.Length}.");
        Assert.True(viaRoute >= probes.Length - 1, $"answers must come via the learned-op route; only {viaRoute}/{probes.Length}.");
    }
}
