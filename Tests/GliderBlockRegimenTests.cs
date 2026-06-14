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
/// PER-COMPONENT GLIDER-BLOCK REGIMENS (PROJECT_GLIDER.md §6). Two proofs that the Compare / Branch /
/// Const blocks have a real training path — not just a hand-built demo:
///  (1) substrate correctness: each capability's hand-built block composition resolves EXACTLY on the
///      platonic substrate (PlatonicGliderInterpreter.TryResolveCapability) — fast, no NN.
///  (2) emergence: trained on the focused creators, the model learns to ROUTE each capability to the
///      platonic-direct glider-block path and answer correctly (majority bar, demonstrate-don't-overfit).
/// </summary>
public sealed class GliderBlockRegimenTests
{
    private readonly ITestOutputHelper _out;
    public GliderBlockRegimenTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void BlockCapabilities_ResolveExactly_OnTheSubstrate()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var glider = new PlatonicGliderInterpreter(memory);

        var cases = new (string Input, string Expected)[]
        {
            ("compare 7 3", "greater"), ("compare 2 9", "less"), ("compare 4 4", "equal"),
            ("compare -5 -8", "greater"),
            ("larger 7 3", "7"), ("larger 2 9", "9"), ("larger -1 -4", "-1"),
            ("double 4", "8"), ("double 7", "14"), ("triple 5", "15"), ("triple 6", "18"),
            ("twicelarger 7 3", "14"), ("twicelarger 2 9", "18"), ("twicelarger 5 5", "10"),
        };

        foreach (var (input, expected) in cases)
        {
            var ok = glider.TryResolveCapability(input, out var answer, out var capability);
            _out.WriteLine($"  '{input,-14}' -> '{answer}' ({capability}) exp '{expected}' {(ok && answer == expected ? "" : "MISS")}");
            Assert.True(ok, $"'{input}' should resolve as a block capability");
            Assert.Equal(expected, answer);
        }

        // Non-capabilities must abstain so NL flows through the ML path.
        Assert.False(glider.TryResolveCapability("what is the capital of france", out _, out _));
        Assert.False(glider.TryResolveCapability("compare 7", out _, out _)); // wrong arity
    }

    [Fact]
    public void BlockCapabilities_CanEmerge_ViaTraining()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(
            tokenizer, model, memory, null,
            trainer.FoldPathDiscovery, trainer.TransformAccumulator,
            enableDiagnosticFaceArithmeticShortcut: true);
        trainer.SetInferencePolicy(inference);

        var data = new List<GenesisExample>();
        var creators = new IExampleCreator[]
            { new ComparisonCreator(), new BranchSelectCreator(), new ConstScaleCreator(), new RefComposeCreator() };
        foreach (var creator in creators)
            foreach (var diff in new[] { 0, 1 })
                foreach (var (inp, outp) in creator.Generate(120, diff, forTraining: true))
                    data.Add(new GenesisExample(inp, outp, SourceCreatorName: creator.Name));

        var rng = new Random(123);
        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (d[i], d[j]) = (d[j], d[i]);
            }
        }

        var probes = new[]
        {
            ("compare 7 3", "greater"), ("compare 2 9", "less"), ("compare 4 4", "equal"),
            ("larger 7 3", "7"), ("larger 2 9", "9"),
            ("double 4", "8"), ("triple 5", "15"),
            ("twicelarger 7 3", "14"), ("twicelarger 2 9", "18"),
        };
        bool GoalReached() =>
            probes.Count(p => inference.Generate(new GenerationRequest(p.Item1, 4)).Output.Trim() == p.Item2) >= 8;

        var pool = data.ToList();
        Shuffle(pool);
        var idx = 0;
        const int maxSteps = 2500;
        const int probeEvery = 250;
        var stepsRun = 0;
        for (var s = 0; s < maxSteps; s++)
        {
            if (idx >= pool.Count) { Shuffle(pool); idx = 0; }
            trainer.TrainStep(pool[idx++]);
            stepsRun++;
            if (stepsRun % probeEvery == 0 && GoalReached())
                break;
        }
        _out.WriteLine($"training stopped at {stepsRun}/{maxSteps} steps");

        var correct = 0;
        var viaBlock = 0;
        _out.WriteLine("── glider-block capabilities via the platonic path ──");
        foreach (var (q, exp) in probes)
        {
            var g = inference.Generate(new GenerationRequest(q, 4));
            var got = g.Output.Trim();
            var ok = got == exp;
            if (ok) correct++;
            if (g.DecisionPath.Contains("glider-block", StringComparison.OrdinalIgnoreCase)) viaBlock++;
            _out.WriteLine($"  '{q,-14}' path={g.DecisionPath} -> '{got}' exp '{exp}' {(ok ? "" : "MISS")}");
        }

        _out.WriteLine("");
        _out.WriteLine($"EMERGENCE: correct {correct}/{probes.Length} (via glider-block {viaBlock}/{probes.Length})");

        // CAN-OCCUR claim: the components, trained focused, are routed to their platonic block path and
        // answer correctly. Majority bar (demonstrate-don't-overfit), and the majority must go VIA the
        // block path (else it would be neural memorisation, not the component doing the work).
        Assert.True(correct >= 7, $"block capabilities should emerge; only {correct}/{probes.Length} correct. See breakdown.");
        Assert.True(viaBlock >= 7, $"the majority must answer via the platonic glider-block path; only {viaBlock}/{probes.Length} did.");
    }
}
