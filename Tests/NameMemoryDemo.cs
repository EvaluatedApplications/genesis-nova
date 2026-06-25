using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// THE DEMO: a rude bot that REMEMBERS your name. Greet it → rudeness; tell it your name → it remembers (rudely);
// ask "what's my name" → it recalls it; and it addresses you by name — still rude. Session-scoped, so restarting the
// app forgets and the demo runs fresh. Gated by TalkEnabled (the persona/chat mode).
public sealed class NameMemoryDemo
{
    private readonly ITestOutputHelper _out;
    public NameMemoryDemo(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Persona_RemembersAndUsesYourName_StaysRude()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var persona = new PersonalityCurriculum();
        foreach (var (cue, reply) in persona.Repertoire) // seed the rude repertoire as chunks
            space.FineEditFromExample(new[] { cue }, new[] { reply }, false);

        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null)
        { ConsciousField = true, TalkEnabled = true };
        string Ask(string q) { var r = mind.Generate(new GenerationRequest(q, 12)); _out.WriteLine($"  you: {q}\n  bot: {r.Output?.Trim()}  [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        var hi = Ask("hi");                          // rude greeting (may even ask "and you are")
        Assert.NotEmpty(hi);

        var unknown = Ask("what's my name");         // doesn't know yet → rude deflection, no name
        Assert.DoesNotContain("stephen", unknown, StringComparison.OrdinalIgnoreCase);

        var captured = Ask("my name is stephen");    // remembers it (rudely)
        Assert.Contains("stephen", captured, StringComparison.OrdinalIgnoreCase);

        var recalled = Ask("what's my name");        // recalls it
        Assert.Contains("stephen", recalled, StringComparison.OrdinalIgnoreCase);

        // Addresses you by name in normal rude replies (alternates, so check a few turns).
        var addressed = false;
        foreach (var q in new[] { "hello", "thanks", "bye", "help", "hey" })
            if (Ask(q).Contains("stephen", StringComparison.OrdinalIgnoreCase)) addressed = true;
        Assert.True(addressed, "the bot addresses you by name");
    }
}
