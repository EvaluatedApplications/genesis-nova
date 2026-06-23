using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
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
/// M4 — does the GRU actually LEARN through the new DialecticalSpace on a REAL training loop? Builds the full stack
/// (model + DialecticalSpace + trainer + inference) and trains a flat associative index broad, then measures
/// retrieval% / routed% / diversity (collapse detector) — the same instrument as AssociativeIndexLearningTests, now
/// against the rebuilt core. Also confirms arithmetic stays exact through the trained space (homomorphism intact).
/// Opt-in [SlowFact] (real NN training at production sizing).
/// </summary>
public sealed class DialecticalSpaceM4Tests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceM4Tests(ITestOutputHelper o) => _out = o;

    private static string Tok3(int i, char p)
        => $"{p}{(char)('a' + (i / 676) % 26)}{(char)('a' + (i / 26) % 26)}{(char)('a' + i % 26)}";
    private static string Canon(string s) => new((s ?? "").ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());

    [SlowFact]
    public void GruLearnsRetrieval_ThroughDialecticalCore()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        IPlatonicSpace mem = new DialecticalSpace(config.FaceDimension, seed: 1234);
        mem.RegisterOperationToken("find"); // op-token = route trigger, never a relation participant
        var trainer = new GenesisTrainer(tok, model, mem, config);
        var infer = new GenesisInferenceEngine(tok, model, mem, null);
        trainer.SetInferencePolicy(infer);

        const int N = 12;
        var data = Enumerable.Range(0, N).Select(i => ($"find {Tok3(i, 'q')}", Tok3(i, 'z'))).ToList();
        var rng = new Random(7);

        for (var ep = 0; ep < 30; ep++)
        {
            var order = data.OrderBy(_ => rng.Next()).ToList();
            foreach (var (cue, key) in order) trainer.TrainStep(new GenesisExample(cue, key));
        }

        int hits = 0, routed = 0; var outputs = new HashSet<string>();
        foreach (var (cue, key) in data)
        {
            var r = infer.Generate(new GenerationRequest(cue, 12));
            if (Canon(r.Output).Contains(Canon(key))) hits++;
            if (r.UsedPlatonicQuery && !r.UsedNeuralFallback) routed++;
            outputs.Add(Canon(r.Output));
        }
        double retrieval = hits / (double)N, routedPct = routed / (double)N, diversity = outputs.Count / (double)N;
        _out.WriteLine($"[M4 new core] N={N}  retrieval={retrieval:P0}  routed={routedPct:P0}  diversity={diversity:P0}  nodes={mem.NodeCount} rels={mem.RelationCount}");

        // Arithmetic stays exact through the trained space (the homomorphism boundary is untouched by learning).
        var interp = new PlatonicGliderInterpreter(mem);
        var glider = new PlatonicGlider("a", new Compute(GliderOp.Add, new GliderBlock[] { new Operand(0), new Operand(1) }));
        var sum = double.Parse(interp.Execute(glider, new[] { "84", "57" }), System.Globalization.CultureInfo.InvariantCulture);
        _out.WriteLine($"[M4 new core] arithmetic 84+57 through trained space = {sum}");

        Assert.Equal(141.0, sum, 6);                                   // homomorphism intact after training
        Assert.True(diversity > 0.5, $"outputs must not collapse to one answer; diversity={diversity:P0}");
        Assert.True(retrieval >= 0.6, $"GRU must learn to retrieve through the new core; retrieval={retrieval:P0} routed={routedPct:P0}");
    }
}
