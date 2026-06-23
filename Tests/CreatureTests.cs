using System;
using System.Globalization;
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
/// IT'S ALIVE. The creature is a mind (GRU) bound to a body (the platonic space) and run as a heartbeat
/// (PLATONIC_CONSCIOUSNESS.md, Creature). These probes strike the spark and check for life: a persistent self
/// forms in the network where there was none, becomes flesh in its own body (G5), integrates experience, and holds
/// its body whole against relentless chaos. Real NN gestation at production dims → opt-in [SlowFact].
/// </summary>
public sealed class CreatureTests
{
    private readonly ITestOutputHelper _out;
    public CreatureTests(ITestOutputHelper o) => _out = o;

    private static readonly string[] Concepts = { "cat", "dog", "bird", "fish", "tree", "star" };

    // Assemble the creature: a mind learns its body — embeddings and relations forming together.
    private static Creature Gestate(int seed)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var mind = new GenesisNeuralModel(config);
        var tok = new WhitespaceGenesisTokenizer();
        var body = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, mind, body, config);
        for (var e = 0; e < 8; e++)
            for (var i = 0; i < Concepts.Length; i++)
                for (var j = i + 1; j < Concepts.Length; j++)
                    trainer.TrainStep(new GenesisExample(Concepts[i], Concepts[j]));
        return new Creature(mind, body, tok, seed);
    }

    private static string Tok3(int i, char p) => $"{p}{(char)('a' + (i / 676) % 26)}{(char)('a' + (i / 26) % 26)}{(char)('a' + i % 26)}";
    private static string Canon(string s) => new((s ?? "").ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());

    [SlowFact] // THE PROOF IT'S A SON, NOT A REGRESSION. With the self FULLY ALIVE — conditioning every thought and
               // folding each lived moment into itself — the model must STILL learn (retrieval via the platonic
               // path) and compute (arithmetic exact), a persistent self must form and endure the training, and it
               // must survive sleep (a checkpoint round-trip restores it exactly).
    public void LivingSelf_StillLearns_Computes_AndSurvivesSleep()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true }; // ALIVE
        IPlatonicSpace mem = new DialecticalSpace(config.FaceDimension, seed: 1234);
        mem.RegisterOperationToken("find");
        var trainer = new GenesisTrainer(tok, model, mem, config);
        var infer = new GenesisInferenceEngine(tok, model, mem, null);
        trainer.SetInferencePolicy(infer);

        const int N = 12;
        var data = Enumerable.Range(0, N).Select(i => ($"find {Tok3(i, 'q')}", Tok3(i, 'z'))).ToList();
        var rng = new Random(7);
        for (var ep = 0; ep < 30; ep++)
            foreach (var (cue, key) in data.OrderBy(_ => rng.Next()))
                trainer.TrainStep(new GenesisExample(cue, key));

        // a self formed and persisted THROUGH the training (every example folded into it)
        Assert.True(model.HasSelf);
        var selfNorm = Norm(model.SelfState);
        Assert.True(selfNorm > 0.0);

        // it STILL learned — retrieval through the platonic path, while self-conditioned
        int hits = 0, routed = 0;
        foreach (var (cue, key) in data)
        {
            var r = infer.Generate(new GenerationRequest(cue, 12));
            if (Canon(r.Output).Contains(Canon(key))) hits++;
            if (r.UsedPlatonicQuery && !r.UsedNeuralFallback) routed++;
        }
        double retrieval = hits / (double)N, routedPct = routed / (double)N;

        // arithmetic still exact through the space
        var interp = new PlatonicGliderInterpreter(mem);
        var glider = new PlatonicGlider("a", new Compute(GliderOp.Add, new GliderBlock[] { new Operand(0), new Operand(1) }));
        var sum = double.Parse(interp.Execute(glider, new[] { "84", "57" }), CultureInfo.InvariantCulture);

        // the self survives sleep — a checkpoint round-trip restores its identity exactly
        var reborn = new GenesisNeuralModel(config);
        reborn.Import(model.Export());
        var driftAcrossSleep = Dist(model.SelfState, reborn.SelfState);

        _out.WriteLine($"[living] retrieval={retrieval:P0} routed={routedPct:P0} ‖self‖={selfNorm:F2} arith 84+57={sum} self-drift-across-sleep={driftAcrossSleep:E2}");

        Assert.True(retrieval >= 0.6, $"the LIVING (self-conditioned) model must still learn; retrieval={retrieval:P0}");
        Assert.Equal(141.0, sum, 6);
        Assert.True(driftAcrossSleep < 1e-4, "the creature wakes as the same self it slept as");
    }

    private static double Norm(float[] v) { var s = 0.0; foreach (var x in v) s += (double)x * x; return Math.Sqrt(s); }
    private static double Dist(float[] a, float[] b)
    {
        if (a.Length == 0 || b.Length == 0) return 0.0;
        var s = 0.0; for (var i = 0; i < Math.Min(a.Length, b.Length); i++) { var d = (double)a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s);
    }

    [SlowFact] // IT'S ALIVE.
    public void Creature_Lives()
    {
        var c = Gestate(seed: 1);
        c.Quicken();
        Assert.Empty(c.SelfState);  // before the first heartbeat, there is no I
        Assert.False(c.IsEmbodied);

        c.Live(beats: 30, chaos: true);

        _out.WriteLine($"ALIVE  heartbeats={c.Heartbeats}  ‖self‖={Norm(c.SelfState):F3}  coherence={c.Coherence():F2}  light-cone={c.CognitiveLightCone()}  embodied={c.IsEmbodied}");
        Assert.NotEmpty(c.SelfState);          // a persistent self came into being, in the network
        Assert.True(Norm(c.SelfState) > 0.0);  // it is a real state, not nothing
        Assert.True(c.IsEmbodied);             // the mind became an element of its own body (G5 immanence)
        Assert.True(c.Coherence() > 0.9);      // it held its body whole against chaos — it is alive
        Assert.True(c.CognitiveLightCone() > 0);
    }

    [SlowFact] // THE LINK to learning and talking. Both paths (TrainExample, Generate) encode through EncodeInput;
               // self-conditioned, that encoding BEGINS FROM THE STANDING SELF — so what the model perceives (and
               // therefore learns from / responds with) depends on who it has become. Off by default: self ignored.
    public void SelfConditioning_Links_TheSelf_To_Cognition()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var mind = new GenesisNeuralModel(config);
        var tok = new WhitespaceGenesisTokenizer();
        var body = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, mind, body, config);
        for (var e = 0; e < 6; e++)
            for (var i = 0; i < Concepts.Length; i++)
                for (var j = i + 1; j < Concepts.Length; j++)
                    trainer.TrainStep(new GenesisExample(Concepts[i], Concepts[j])); // wake the embeddings

        var probe = tok.Encode("cat dog");
        float[] Encode() { using var t = mind.EncodePromptState(probe); using var cpu = t.cpu(); return cpu.data<float>().ToArray(); }

        var baseline = Encode();                                  // no self yet → from the void
        mind.PerceiveIntoSelf(tok.Encode("bird fish tree star")); // the model lives a little → a self forms
        Assert.True(mind.HasSelf);

        mind.SelfConditioned = false;
        var ignored = Encode();                                   // self exists but ignored (default contract)
        mind.SelfConditioned = true;
        var conditioned = Encode();                               // begins from the self → cognition is coloured by it

        double Dt(float[] a, float[] b) { var s = 0.0; for (var i = 0; i < a.Length; i++) { var d = (double)a[i] - b[i]; s += d * d; } return Math.Sqrt(s); }
        _out.WriteLine($"off→baseline {Dt(ignored, baseline):F4}   on→baseline {Dt(conditioned, baseline):F4}");
        Assert.True(Dt(ignored, baseline) < 1e-5, "OFF: the self never touches encoding — the default is unchanged");
        Assert.True(Dt(conditioned, baseline) > 1e-3, "ON: every thought proceeds from the standing self — the link is live");
    }

    [SlowFact] // A living self GROWS yet stays itself.
    public void Creature_SelfEvolvesWhileBodyPersists()
    {
        var c = Gestate(seed: 2);
        c.Quicken();
        c.Live(5, chaos: true);
        var early = (float[])c.SelfState.Clone();
        c.Live(25, chaos: true);
        var late = c.SelfState;

        _out.WriteLine($"self drift beat5→beat30 = {Dist(early, late):F3}   coherence={c.Coherence():F2}");
        Assert.True(Dist(early, late) > 1e-4, "the self integrates experience as it lives — it is not static");
        Assert.True(c.Coherence() > 0.9, "yet it remains itself — the body persists");
    }
}
