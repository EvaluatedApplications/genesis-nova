using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// EMIT-TIME SPEAK wired into the LIVE answer path (SpeakAnswer): a route computes a RAW value ("19"); when the space has
/// learned chunk transitions around that value's spoken form, the answer comes out as a composed PHRASE that CARRIES the
/// value (via ComposeWithValue — the value is a walked node, no template). Not gated behind an off-by-default flag; the
/// honest fallback (no learned transitions → raw answer, byte-identical) self-scopes it, so a space never taught to speak
/// is unchanged. Falsifiable: taught-to-speak → phrase containing the value's word form; untaught → raw digit, byte-identical.
/// </summary>
public sealed class SpeakAnswerInlineTests
{
    private readonly ITestOutputHelper _out;
    public SpeakAnswerInlineTests(ITestOutputHelper o) => _out = o;

    private static GenesisInferenceEngine Engine()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 5);
        var model = new GenesisNeuralModel(config);
        // TalkEnabled = conversational mode — the only place answers SPEAK (the gym/reasoning paths report bare values).
        return new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), model, space, null) { ConsciousField = true, TalkEnabled = true };
    }

    [Fact]
    public void ComputedAnswer_SpeaksAsPhrase_WhenTaughtToSpeak_ElseRawByteIdentical()
    {
        // (A) TAUGHT to speak around the answer's number-word → the computed answer comes out as a value-carrying phrase.
        var speak = Engine();
        for (var i = 0; i < 8; i++) speak.ObserveUtteranceSequence("nineteen is the total");
        var spoken = speak.Generate(new GenerationRequest("12 + 7", 16));
        _out.WriteLine($"TAUGHT  : '12 + 7' -> '{spoken.Output}' [{spoken.DecisionPath}]");

        // (B) IDENTICAL engine, NOT taught to speak → the same query returns the raw digit, byte-identical (self-scoped).
        var raw = Engine();
        var rawRes = raw.Generate(new GenerationRequest("12 + 7", 16));
        _out.WriteLine($"UNTAUGHT: '12 + 7' -> '{rawRes.Output}' [{rawRes.DecisionPath}]");

        Assert.Equal("19", rawRes.Output.Trim());                                          // no transitions → raw, unchanged
        Assert.Contains("nineteen", spoken.Output, StringComparison.OrdinalIgnoreCase);     // carried the value (word form)
        Assert.Contains(" ", spoken.Output.Trim());                                         // a PHRASE, not a bare token
        Assert.Equal("field-speak", spoken.DecisionPath);                                  // spoke via the emit-speak path
    }
}
