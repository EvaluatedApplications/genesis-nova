using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// FIRST-PRINCIPLES fact memory (user: "our implementation is overfitting the remember-my-name thing"). Storing/recalling
// a fact is ASSOCIATIVE MEMORY — the substrate's own relation (key→value) — parsed by the COPULA PIVOT (the function word
// splits key|value), NOT by a 4-role NN classifier that overfits "X is Y" and won't generalise. This proves the parse
// generalises to NOVEL never-seen tokens with the role head NEVER TRAINED: the copula/fillers come from the content/
// function split (here the cold-start Framing default; in production the LEARNED IsFunctionLike centrality signal), and
// the split is positional, so any unseen key/value works.
public sealed class FactMemoryFirstPrinciples
{
    private readonly ITestOutputHelper _out;
    public FactMemoryFirstPrinciples(ITestOutputHelper o) => _out = o;

    [Fact]
    public void StoresAndRecalls_NovelFacts_ByCopulaPivot_NoRoleHead()
    {
        var config = new GenesisNovaConfig(HiddenSize: 128, FaceDimensionOverride: 128);
        var infer = new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config),
            new DialecticalSpace(config.FaceDimension, seed: 7), null) { ConsciousField = true };
        string P(string s) { var r = infer.Generate(new GenerationRequest(s, 8)); var o = r.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r.DecisionPath}]"); return o; }

        // NOVEL tokens never seen anywhere — generalisation is positional (copula pivot), not memorised roles. The role
        // head is never trained, so a pass means the parse does NOT depend on it.
        P("my quux is florp");                            // ASSERT: store quux → florp
        Assert.Equal("florp", P("what is my quux"));      // RECALL

        P("my zibble is wozzle");
        Assert.Equal("wozzle", P("what is my zibble"));

        // MULTI-TOKEN subject ("my favorite color") — the copula bounds the whole noun phrase before it, so the recall key
        // matches the stored key (the case the per-token role head kept collapsing).
        P("my favorite color is blarg");
        Assert.Equal("blarg", P("what is my favorite color"));
    }
}
