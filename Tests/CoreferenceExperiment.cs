using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT: within-conversation holding. The conscious field threads working memory as MEANING (_focus/_selfField)
// across turns; coreference makes the binding EXPLICIT so a follow-up "what kind of thing is IT" resolves the anaphor
// to the entity the PREVIOUS turn was about. Proves "it/that" resolves across turns, and that it FAILS without the
// mechanism (so the binding is doing real work). Gated CoreferenceEnabled (default off) → suite byte-identical.
public sealed class CoreferenceExperiment
{
    private readonly ITestOutputHelper _out;
    public CoreferenceExperiment(ITestOutputHelper o) => _out = o;

    private static DialecticalSpace SeededSpace(int dim)
    {
        var s = new DialecticalSpace(dim, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 4; i++) s.FineEditFromExample(new[] { a }, new[] { b }, false); }
        Rel("apple", "fruit"); Rel("dog", "animal"); Rel("rose", "flower");
        return s;
    }

    [Fact]
    public void Field_ResolvesCoreference_AcrossTurns()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var dim = config.FaceDimension;

        // ── WITH coreference ──
        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), SeededSpace(dim), null)
        { ConsciousField = true, CoreferenceEnabled = true };
        string Ask(GenesisInferenceEngine m, string q) { var r = m.Generate(new GenerationRequest(q, 8)); _out.WriteLine($"  '{q}' → '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        var t1 = Ask(mind, "what kind of thing is apple");   // fruit — establishes apple as the referent
        var t2 = Ask(mind, "what kind of thing is it");      // "it" → apple → fruit
        var t3 = Ask(mind, "what kind of thing is dog");     // animal — referent is now dog
        var t4 = Ask(mind, "what kind of thing is it");      // "it" → dog (most recent) → animal, NOT fruit

        Assert.Contains("fruit", t1, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("fruit", t2, StringComparison.OrdinalIgnoreCase); // coreference resolved "it" to apple
        Assert.Contains("animal", t4, StringComparison.OrdinalIgnoreCase); // bound to the MOST-RECENT entity (dog)
        Assert.DoesNotContain("fruit", t4, StringComparison.OrdinalIgnoreCase);

        // ── CONTROL: coreference OFF — "it" can't bind, so the follow-up does NOT resolve to the prior entity ──
        _out.WriteLine("control (coreference off):");
        var plain = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), SeededSpace(dim), null)
        { ConsciousField = true, CoreferenceEnabled = false };
        Ask(plain, "what kind of thing is apple");
        var off = Ask(plain, "what kind of thing is it");
        Assert.DoesNotContain("fruit", off, StringComparison.OrdinalIgnoreCase); // unresolved → abstain / wrong, not "fruit"
    }
}
