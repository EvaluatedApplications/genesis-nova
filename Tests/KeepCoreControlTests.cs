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
/// KEEP-CORE control path (PLATONIC_RECKONING.md) end-to-end. The reckoning's claim: the substrate geometry is
/// healthy, but the controller never SEES it because the trainer anchored route supervision on the first surface
/// token (a framing word "a"/"synonym"/"for") instead of the content cue ("big") that inference retrieves on. With
/// KeepCoreControl ON, BOTH sides anchor on the discriminative cue, RELAXATION (`reason`) becomes the retrieval
/// route, and a query that nothing settles ABSTAINS instead of hallucinating. This test uses the exact bug scenario
/// — synonym queries wrapped in constant framing words — and asserts the controller now routes them through the
/// platonic relaxation path and retrieves correctly, while arithmetic stays exact. Production dims, opt-in [SlowFact].
/// </summary>
public sealed class KeepCoreControlTests
{
    private readonly ITestOutputHelper _out;
    public KeepCoreControlTests(ITestOutputHelper o) => _out = o;

    private static string Canon(string s) => new((s ?? "").ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());

    [SlowFact]
    public void KeepCore_RoutesFramingWordRetrieval_ViaRelaxation()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        // The perception heads are what the keep-core anchor fix feeds — turn them on as production does.
        model.PerceptionRouting = true;
        model.PerceptionQuery = true;
        model.PerceptionPlan = true;
        IPlatonicSpace mem = new DialecticalSpace(config.FaceDimension, seed: 1234);
        var trainer = new GenesisTrainer(tok, model, mem, config) { KeepCoreControl = true };
        var infer = new GenesisInferenceEngine(tok, model, mem, null) { KeepCoreControl = true };
        trainer.SetInferencePolicy(infer);

        // The bug scenario: the CONTENT cue varies; the framing words "a synonym for" are a CONSTANT correlate (they
        // become high-degree hubs). The discriminative anchor must pick the content cue, not the framing word.
        var pairs = new (string cue, string answer)[]
        {
            ("big", "large"), ("small", "tiny"), ("fast", "rapid"), ("happy", "glad"),
            ("smart", "clever"), ("angry", "mad"), ("begin", "start"), ("end", "finish"),
        };
        var data = pairs.Select(p => new GenesisExample($"a synonym for {p.cue}", p.answer)).ToList();
        var answers = new HashSet<string>(pairs.Select(p => Canon(p.answer)));
        var rng = new Random(7);

        for (var ep = 0; ep < 40; ep++)
        {
            foreach (var ex in data.OrderBy(_ => rng.Next()))
                trainer.TrainStep(ex);
        }

        int hits = 0, viaPlatonic = 0, viaReason = 0;
        foreach (var (cue, answer) in pairs)
        {
            var r = infer.Generate(new GenerationRequest($"a synonym for {cue}", 8));
            var ok = Canon(r.Output).Contains(Canon(answer));
            if (ok) hits++;
            if (r.UsedPlatonicQuery && !r.UsedNeuralFallback) viaPlatonic++;
            if (r.DecisionPath.Contains("reason", StringComparison.OrdinalIgnoreCase)) viaReason++;
            _out.WriteLine($"  'a synonym for {cue,-6}' path={r.DecisionPath,-26} -> '{r.Output.Trim()}' {(ok ? "OK" : "")}");
        }
        double retrieval = hits / (double)pairs.Length, platonicPct = viaPlatonic / (double)pairs.Length;
        _out.WriteLine($"[keep-core] retrieval={retrieval:P0} viaPlatonic={platonicPct:P0} viaReason={viaReason}/{pairs.Length} nodes={mem.NodeCount} rels={mem.RelationCount}");

        // ABSTENTION over hallucination: a genuinely unknown cue must NOT emit a confident known answer.
        var unknown = infer.Generate(new GenerationRequest("a synonym for zzqqxx", 8));
        _out.WriteLine($"[keep-core] unknown -> path={unknown.DecisionPath} '{unknown.Output.Trim()}'");
        Assert.DoesNotContain(Canon(unknown.Output), answers);

        // Arithmetic stays exact through the trained space (homomorphism boundary untouched by the control change).
        var interp = new PlatonicGliderInterpreter(mem);
        var glider = new PlatonicGlider("a", new Compute(GliderOp.Add, new GliderBlock[] { new Operand(0), new Operand(1) }));
        var sum = double.Parse(interp.Execute(glider, new[] { "84", "57" }), System.Globalization.CultureInfo.InvariantCulture);
        Assert.Equal(141.0, sum, 6);

        // The controller now SEES the healthy geometry: framing-word retrieval routes platonic and resolves.
        Assert.True(retrieval >= 0.6, $"keep-core must retrieve through framing words; retrieval={retrieval:P0}");
        Assert.True(platonicPct >= 0.6, $"retrieval must route PLATONIC, not neural; viaPlatonic={platonicPct:P0}");
        Assert.True(viaReason >= 1, $"the relaxation (reason) route must fire for retrieval; viaReason={viaReason}/{pairs.Length}");
    }
}
