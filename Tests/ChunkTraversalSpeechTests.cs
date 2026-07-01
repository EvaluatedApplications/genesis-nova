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
/// GENERATIVE SPEECH by MULTI-CHUNK TRAVERSAL (the talk-by-chunk "sequencing" next level). NOT a template: the utterance
/// is COMPOSED by a REPEATED-QUERY cascade — predict the next chunk from the accumulated context (ds.Reason, which
/// excludes the context so it returns a CONTINUATION), append it, re-query with the grown utterance, until nothing
/// settles. Order lives in LEARNED substrate edges. The falsifier for "traversal not template": the composed output must
/// VARY with the learned transitions — a template would emit a fixed frame regardless of what was taught.
/// </summary>
public sealed class ChunkTraversalSpeechTests
{
    private readonly ITestOutputHelper _out;
    public ChunkTraversalSpeechTests(ITestOutputHelper o) => _out = o;

    private static GenesisInferenceEngine Engine(int seed)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        return new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config),
            new DialecticalSpace(config.FaceDimension, seed: seed), null) { ChunkTraversalSpeech = true };
    }

    [Fact]
    public void ComposesUtterance_ByRepeatedQuery_AndVariesWithLearnedTransitions()
    {
        // Teaching A: the chunk order "hello → my → good → friend" is learned as substrate edges.
        var a = Engine(5);
        a.ObserveUtteranceSequence("hello my good friend");
        var outA = a.ComposeByTraversal("hello");
        _out.WriteLine($"A taught 'hello my good friend' → compose('hello') = 'hello {outA}'");

        // Teaching B: a DIFFERENT learned order from the SAME cue.
        var b = Engine(5);
        b.ObserveUtteranceSequence("hello there everyone today");
        var outB = b.ComposeByTraversal("hello");
        _out.WriteLine($"B taught 'hello there everyone today' → compose('hello') = 'hello {outB}'");

        var chunksA = outA.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var chunksB = outB.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // (1) COMPOSED, not one whole chunk: >= 2 distinct chunks walked out.
        Assert.True(chunksA.Length >= 2, $"A should compose >=2 chunks by traversal, got '{outA}'");
        Assert.True(chunksA.Distinct(StringComparer.OrdinalIgnoreCase).Count() == chunksA.Length, "chunks should be distinct (a walk, not a loop)");

        // (2) TRAVERSAL not template: same compose code, DIFFERENT learned edges → DIFFERENT utterance.
        Assert.NotEqual(outA, outB);
        _out.WriteLine(">>> composed by repeated-query traversal; output tracks the LEARNED transitions (not a fixed template).");
    }
}
