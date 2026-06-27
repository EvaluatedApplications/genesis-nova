using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// M2 of the Merge foundation ([[nova-merge-substrate-plan]]): the surface parser builds a recursive MERGE TREE for the
// key (left-branching over the whole span, possessives included), wired through the engine. Two things the flat parses
// could NOT do: (1) DISTINGUISH possessives — "my name" ≠ "your name" (the coreference the role head / flat copula-pivot
// collapsed); (2) ARBITRARY-LENGTH recursive subjects. No role head, no training — structural + novel-token general.
public sealed class MergeParserTests
{
    private readonly ITestOutputHelper _out;
    public MergeParserTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void MergeParser_DistinguishesPossessives_AndRecursesArbitrarily()
    {
        var config = new GenesisNovaConfig(HiddenSize: 128, FaceDimensionOverride: 128);
        var infer = new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config),
            new DialecticalSpace(config.FaceDimension, seed: 7), null) { ConsciousField = true };
        string P(string s) { var r = infer.Generate(new GenerationRequest(s, 8)); var o = r.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r.DecisionPath}]"); return o; }

        // POSSESSIVE DISTINCTION — the key is the WHOLE span ⟨my·np·name⟩ vs ⟨your·np·name⟩, so they don't collide.
        P("my name is sam");
        P("your name is rex");
        Assert.Equal("sam", P("what is my name"));
        Assert.Equal("rex", P("what is your name"));

        // ARBITRARY-LENGTH recursive subject (4 content/function tokens) on NOVEL tokens — a left-branching Merge tree.
        P("my favorite warm color is amber");
        Assert.Equal("amber", P("what is my favorite warm color"));

        // Belief revision still holds with the structured key.
        P("my name is max");
        Assert.Equal("max", P("what is my name"));
        Assert.Equal("rex", P("what is your name")); // unaffected — a DIFFERENT key
    }
}
