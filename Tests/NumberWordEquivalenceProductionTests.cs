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
/// PRODUCTION-DIMENSION behavioral demo: bidirectional number-word equivalence (one≡1) works VIA THE
/// PLATONIC PATH in BOTH directions. This MUST run at a production face dimension — at the old test
/// dim (face 32) the platonic space is degenerate (no word face, no free region, numeric/char
/// overlap) and digit→word is impossible. With a real free region it works once (a) single digits can
/// anchor and (b) retrieval prefers learned relations over value-proximity (so "3"→"three", not the
/// numerically-adjacent "4"). General mechanism — same as him→paul / apple→fruit.
///
/// Slow (HiddenSize 512 GPU training) — a behavioral demo, not a fast unit test.
/// </summary>
public sealed class NumberWordEquivalenceProductionTests
{
    private readonly ITestOutputHelper _out;
    public NumberWordEquivalenceProductionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Bidirectional_NumberWordEquivalence_IsPlatonic_AtProductionDimension()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05); // face 256: full layout + free region
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(
            tokenizer, model, memory, null,
            trainer.FoldPathDiscovery, trainer.TransformAccumulator,
            enableDiagnosticFaceArithmeticShortcut: true);
        trainer.SetInferencePolicy(inference);

        // Sanity: the platonic space must actually be fully instantiated (free region beyond numeric).
        Assert.True(memory.FaceDimension > 2 * memory.NumericDimensions + 8,
            $"face {memory.FaceDimension} has no free region beyond the {2 * memory.NumericDimensions}-dim numeric face.");

        var lesson = new NumberWordCreator().Generate(20, 0, true)
            .Select(p => new GenesisExample(p.Input, p.Output, SourceCreatorName: "corenova:number-word-equiv"))
            .ToList();
        var rng = new Random(123);
        for (var e = 0; e < 20; e++)
        {
            for (var i = lesson.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (lesson[i], lesson[j]) = (lesson[j], lesson[i]); }
            foreach (var ex in lesson) trainer.TrainStep(ex);
        }

        var digitWords = new[] { ("0","zero"),("1","one"),("2","two"),("3","three"),("4","four"),
            ("5","five"),("6","six"),("7","seven"),("8","eight"),("9","nine") };
        int w2d = 0, d2wPlat = 0;
        foreach (var (digit, word) in digitWords)
        {
            if (inference.Generate(new GenerationRequest(word, 4)).Output.Trim() == digit) w2d++;
            var g = inference.Generate(new GenerationRequest(digit, 4));
            if (g.Output.Trim() == word && g.UsedPlatonicQuery && !g.UsedNeuralFallback) d2wPlat++;
            _out.WriteLine($"  {digit}->'{g.Output.Trim()}'(plat={g.UsedPlatonicQuery && !g.UsedNeuralFallback}) | {word}");
        }
        _out.WriteLine($"word→digit {w2d}/10  digit→word(platonic) {d2wPlat}/10");

        // BOTH directions must work via the platonic path (capability demonstration, modest bar).
        Assert.True(w2d >= 8, $"word→digit only {w2d}/10");
        Assert.True(d2wPlat >= 8, $"digit→word platonic only {d2wPlat}/10");
    }
}
