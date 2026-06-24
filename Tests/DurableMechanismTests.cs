using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Runtime;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// DURABILITY (not a clean toy world): the mechanisms must hold in a REAL POLLUTED model — one trained on the whole
// gym curriculum, full of hubs, framing-word concepts, number pollution and competing relations. A solution that
// only works in isolation hasn't earned its keep (the scale-recall lesson). This trains such a model once, then runs
// each mechanism IN it and asserts it survives the noise — targeting the known weaknesses, generally, not test-fit.
public sealed class DurableMechanismTests
{
    private readonly ITestOutputHelper _out;
    public DurableMechanismTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Mechanisms_SurviveInANoisyTrainedModel()
    {
        const int HIDDEN = 256, SEED = 7;
        var rng = new Random(SEED);
        var tok = new WhitespaceGenesisTokenizer();
        var nova = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: HIDDEN, LearningRate: 0.05, Seed: SEED).WithProductionMechanisms());
        string Gen(string i) => nova.Inference.Generate(new GenerationRequest(i, 8)).Output?.Trim() ?? "";

        // ── Build the pollution: train on the whole gym curriculum (correctness-gated, like the race) ──
        var creators = ExampleCreatorRegistry.All.Append(new GenesisNova.Data.Creators.AssociationRecallCreator()).ToList();
        var train = new List<(string Input, string Output)>();
        foreach (var c in creators)
        {
            var ex = new List<(string, string)>();
            foreach (var diff in new[] { 0, 1 }) ex.AddRange(c.Generate(400, diff, true));
            var uniq = ex.GroupBy(e => e.Item1).Select(g => g.First()).OrderBy(_ => rng.Next()).ToList();
            train.AddRange(uniq.Take(Math.Min((int)(uniq.Count * 0.65), 160)));
        }
        for (var epoch = 1; epoch <= 6; epoch++)
            foreach (var (i, o) in train.OrderBy(_ => rng.Next()))
                if (!AnswerEquivalence.Equivalent(Gen(i), o)) nova.Trainer.TrainStep(new GenesisExample(i, o));

        var ds = (DialecticalSpace)nova.Memory;
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) ds.FineEditFromExample(new[] { a }, new[] { b }, false); }

        // ── Heavier POLLUTION at scale — the known weakness triggers: hundreds of concepts, MEGA-HUBS, framing-word
        //    hubs (the apple→"kind" class), and dense random noise. The mechanisms must survive hub dilution. ──
        var noise = Enumerable.Range(0, 400).Select(i => $"n{i}").ToArray();
        foreach (var hub in new[] { "thing", "stuff", "item", "object", "category" })
            for (var k = 0; k < 45; k++) Rel(hub, noise[rng.Next(noise.Length)]);
        foreach (var f in new[] { "the", "of", "what", "is", "kind", "a" }) // framing-word hubs
            for (var k = 0; k < 30; k++) Rel(f, noise[rng.Next(noise.Length)]);
        for (var k = 0; k < 700; k++) Rel(noise[rng.Next(noise.Length)], noise[rng.Next(noise.Length)]);
        _out.WriteLine($"polluted model: {ds.NodeCount} concepts, {ds.GetAllRelations().Count} relations");

        // ── 1. LEARNED OP-CUE survives the noise: teach a NOVEL operator into the polluted model, resolve it ──
        nova.Inference.LearnArithmeticCue("4 zorp 3", "12");
        nova.Inference.LearnArithmeticCue("5 zorp 2", "10");
        nova.Inference.LearnArithmeticCue("6 zorp 3", "18");
        var zorp = Gen("7 zorp 2");
        _out.WriteLine($"[op-cue]  7 zorp 2 -> {zorp} (want 14)");
        Assert.Equal("14", zorp); // learned ∘mul despite a space full of other relations/hubs

        // ── 2. ARITHMETIC still computes exactly in the noisy model ──
        Assert.Equal("22", Gen("13 + 9"));
        Assert.Equal("48", Gen("8 x 6"));

        // ── 3. BELIEF REVISION holds under noise: the world changes, the mind stays CURRENT ──
        nova.Inference.Generate(new GenerationRequest("qzx is red", 8));
        nova.Inference.Generate(new GenerationRequest("qzx is iron", 8)); // the world changed
        var current = Gen("what is qzx");
        _out.WriteLine($"[revise]  what is qzx -> {current} (want iron, not red)");
        Assert.Equal("iron", current);

        // ── 3b. SINGLE-FACT RECALL survives HUB DILUTION: a fact buried in the 500-concept noise stays retrievable ──
        nova.Inference.Generate(new GenerationRequest("vquit is copper", 8));
        var recall = Gen("what is vquit");
        _out.WriteLine($"[recall]  what is vquit -> {recall} (want copper)");
        Assert.Equal("copper", recall);

        // ── 4. COMPOSE + ANALOGY survive HUB DILUTION: add a small meaning world ON TOP of the pollution ──
        Rel("apple", "red"); Rel("apple", "fruit"); Rel("cherry", "red"); Rel("cherry", "fruit");
        Rel("lemon", "yellow"); Rel("lemon", "fruit"); Rel("brick", "red"); Rel("brick", "stone");
        Rel("paris", "france"); Rel("rome", "italy"); Rel("tokyo", "japan");
        // permissive engine (no director gate) on the SAME polluted space, to exercise the meaning routes directly
        var mind = new GenesisInferenceEngine(tok, nova.Model, ds, null) { ConsciousField = true, MeaningOpsEnabled = true };
        var compose = mind.Generate(new GenerationRequest("red fruit", 6));
        var analogy = mind.Generate(new GenerationRequest("paris is to france as tokyo is to", 6));
        _out.WriteLine($"[compose] red fruit -> {compose.Output?.Trim()} [{compose.DecisionPath}]");
        _out.WriteLine($"[analogy] paris:france::tokyo -> {analogy.Output?.Trim()} [{analogy.DecisionPath}]");
        Assert.Contains(compose.Output?.Trim(), new[] { "apple", "cherry" }); // not drowned by the hubs
        Assert.Equal("japan", analogy.Output?.Trim());
    }
}
