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

/// <summary>
/// VALUE-CARRY in chunk-traversal speech: the model speaks a COMPUTED/RETRIEVED value inside a composed utterance, with the
/// value as a NODE the walk carries — NOT a wrapper/template. Seed the walk with the value, walk outward over LEARNED chunk
/// transitions; the surrounding words are learned predictions. Falsifier for "traversal not template": the value is present
/// AND the surrounding words VARY with the learned transitions (a template would say a fixed frame regardless).
/// HONEST BOUNDARY: a BARE numeric token ("19") never forms substrate edges (numbers-never-edge rule), so its walk can't
/// leave the value — the answer must be carried in its number-WORD / concept form. This test uses word-form values.
/// </summary>
public sealed class ChunkTraversalValueCarryTests
{
    private readonly ITestOutputHelper _out;
    public ChunkTraversalValueCarryTests(ITestOutputHelper o) => _out = o;

    private static GenesisInferenceEngine Engine(int seed)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        return new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config),
            new DialecticalSpace(config.FaceDimension, seed: seed), null) { ChunkTraversalSpeech = true };
    }

    [Fact]
    public void SpeaksAValue_CarriedAsAWalkedNode_NotATemplate()
    {
        // Teaching A: transitions AROUND the value word 'nineteen' (nineteen→is→the→total).
        var a = Engine(5);
        a.ObserveUtteranceSequence("nineteen is the total");
        var outA = a.ComposeWithValue("nineteen");
        _out.WriteLine($"A taught 'nineteen is the total' → ComposeWithValue('nineteen') = '{outA}'");

        // Teaching B: a DIFFERENT learned surround for the SAME value.
        var b = Engine(5);
        b.ObserveUtteranceSequence("nineteen sounds about right");
        var outB = b.ComposeWithValue("nineteen");
        _out.WriteLine($"B taught 'nineteen sounds about right' → ComposeWithValue('nineteen') = '{outB}'");

        // Concept value carried the same way (retrieved-answer case).
        var c = Engine(5);
        c.ObserveUtteranceSequence("fruit is quite tasty");
        var outC = c.ComposeWithValue("fruit");
        _out.WriteLine($"C taught 'fruit is quite tasty' → ComposeWithValue('fruit') = '{outC}'");

        // (a) the VALUE appears in the composed utterance.
        Assert.Contains("nineteen", outA, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("nineteen", outB, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("fruit", outC, StringComparison.OrdinalIgnoreCase);

        // (b) it's a COMPOSED multi-chunk utterance (value + >=1 learned chunk), by traversal.
        Assert.True(outA.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length >= 2, $"value + learned continuation, got '{outA}'");

        // (c) TRAVERSAL not template: same value, DIFFERENT learned edges → DIFFERENT surrounding words (value still present).
        Assert.NotEqual(outA, outB);
        _out.WriteLine(">>> value carried as a WALKED NODE; the surround varies with the learned transitions — not a template/wrapper.");
    }
}
