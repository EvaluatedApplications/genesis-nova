using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT — before a LONG training run: does the field-cognition mind hold its knowledge AT SCALE (lattice
// engaged, >384 concepts), or does retrieval rot as the body fills? Long training only makes sense if it does.
public sealed class LivingScaleExperiment
{
    private readonly ITestOutputHelper _out;
    public LivingScaleExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Does_Recall_Hold_AtScale()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true };
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        void Tell(string s) => mind.Generate(new GenerationRequest(s, 8));
        string Ask(string who) => mind.Generate(new GenerationRequest($"what is {who}", 8)).Output?.Trim() ?? "";
        static string Nm(int i) => $"{(char)('a' + i / 676 % 26)}{(char)('a' + i / 26 % 26)}{(char)('a' + i % 26)}";

        // A pool of ~30 shared values (so each becomes a hub of many subjects — the realistic, hard case for retrieval).
        var values = new[] { "fruit","metal","bird","tool","river","plant","cloth","stone","spice","drink",
                             "fish","wood","grain","flower","gem","beast","insect","fungus","resin","oil",
                             "salt","clay","glass","paper","wax","silk","leather","bone","shell","feather" };
        const int N = 600;
        var truth = new System.Collections.Generic.Dictionary<string, string>();
        for (var i = 0; i < N; i++)
        {
            var who = "q" + Nm(i);
            var val = values[i % values.Length];
            Tell($"{who} is {val}");
            truth[who] = val;
        }
        _out.WriteLine($"[scale] taught {N} facts; active concepts = {space.NodeCount} (lattice engages >384)");

        // Probe a spread of 60 facts taught throughout the run (early, middle, late).
        var rng = new Random(3);
        var sample = Enumerable.Range(0, N).Where(i => i % 10 == 0).Select(i => "q" + Nm(i)).ToArray();
        // DIAGNOSE the lattice path: what does GetNearestConcepts return for a probe, vs the exact scan?
        var w0 = sample[0];
        _out.WriteLine($"[diag] {w0} (truth {truth[w0]}): nearest(lattice) = " +
            string.Join(", ", space.GetNearestConcepts(w0, null, 5).Select(n => $"{n.Symbol}:{n.Distance:F2}")));
        _out.WriteLine($"[diag] {w0}: nearest(scan) = " +
            string.Join(", ", space.GetNearestConcepts(w0, space.ActiveConcepts, 5, 2000).Select(n => $"{n.Symbol}:{n.Distance:F2}")));
        _out.WriteLine($"[diag] {w0} -> Ask = '{Ask(w0)}'");

        var hits = sample.Count(w => Ask(w).Equals(truth[w], StringComparison.OrdinalIgnoreCase));
        _out.WriteLine($"[scale] recall on {sample.Length} probes spread across the run = {hits}/{sample.Length} ({hits / (double)sample.Length:P0})");

        Assert.True(hits >= sample.Length * 0.9, $"recall must hold at scale before a long run; {hits}/{sample.Length}");
    }

    [Fact] // The two living integrations TOGETHER, AT SCALE: over a long run the world keeps CHANGING (belief revision)
           // while the body keeps GROWING (hub dilution). Belief-revision was only proven at ~70 concepts; does the
           // mind still answer the CURRENT truth when the updates are buried under a large, lattice-engaged space?
    public void Does_BeliefRevision_Hold_AtScale()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true };
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        void Tell(string s) => mind.Generate(new GenerationRequest(s, 8));
        string Ask(string who) => mind.Generate(new GenerationRequest($"what is {who}", 8)).Output?.Trim() ?? "";
        static string Nm(char p, int i) => $"{p}{(char)('a' + i / 676 % 26)}{(char)('a' + i / 26 % 26)}{(char)('a' + i % 26)}";

        var v1 = new[] { "red","blue","green","gold","black","white","brown","pink","cyan","lime","navy","teal","plum","ruby","jade","sand","rose","mint","grey","tan" };
        var v2 = new[] { "iron","zinc","lead","copper","steel","brass","nickel","silver","cobalt","chrome","bronze","pewter","mercury","platinum","aluminium","titanium","tungsten","carbon","cadmium","gallium" };
        var subj = Enumerable.Range(0, 20).Select(i => Nm('s', i)).ToArray();
        var current = new System.Collections.Generic.Dictionary<string, string>();

        // 20 beliefs are asserted EARLY (v1)...
        for (var i = 0; i < subj.Length; i++) { Tell($"{subj[i]} is {v1[i]}"); current[subj[i]] = v1[i]; }

        // ...then the body GROWS to lattice scale with hundreds of unrelated facts (each value a populous hub)...
        var fillers = new[] { "fruit","metal","bird","tool","river","plant","cloth","stone","spice","drink",
                              "fish","wood","grain","flower","gem","beast","insect","fungus","resin","oil" };
        for (var i = 0; i < 600; i++) Tell($"{Nm('f', i)} is {fillers[i % fillers.Length]}");

        // ...and only NOW, deep into a large space, does the world CHANGE for 12 of the 20 (current truth becomes v2).
        for (var i = 0; i < 12; i++) { Tell($"{subj[i]} is {v2[i]}"); current[subj[i]] = v2[i]; }

        // ...then a final flood buries the updates so recency alone cannot carry them.
        for (var i = 0; i < 200; i++) Tell($"{Nm('g', i)} is {fillers[i % fillers.Length]}");
        _out.WriteLine($"[revise@scale] active concepts = {space.NodeCount}");

        int updHit = 0, updStale = 0, stableHit = 0;
        for (var i = 0; i < subj.Length; i++)
        {
            var ans = Ask(subj[i]);
            if (i < 12) { if (ans.Equals(v2[i], StringComparison.OrdinalIgnoreCase)) updHit++; else if (ans.Equals(v1[i], StringComparison.OrdinalIgnoreCase)) updStale++; }
            else if (ans.Equals(v1[i], StringComparison.OrdinalIgnoreCase)) stableHit++;
        }
        _out.WriteLine($"[revise@scale] contradicted current={updHit}/12 stale={updStale}  |  untouched recalled={stableHit}/8");

        // The mind must still update to the CURRENT truth at scale (relation-first carries the dominant fresh belief
        // without dilution) AND still hold the untouched ones.
        Assert.True(updHit >= 10, $"belief revision must hold at scale; current={updHit}/12 stale={updStale}");
        Assert.True(stableHit >= 7, $"untouched beliefs must survive the flood at scale; {stableHit}/8");
    }
}
