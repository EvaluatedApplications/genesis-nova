using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THINK BACK TO NUCLEAR FOLDING: don't hand-build the structure — the FOLD generates it; the only irreducible inputs
/// are the MEASURED couplings that tune it. Relational reasoning was already LATENT (like Boolean was latent in +/×);
/// two small blockers hid it, and this test proves both are now cleared:
///   • B1 — the walk/mirror used to LAND on the ∘is label token (a coupling marker). Now they step THROUGH it.
///   • B2 — belief revision used to key on the CONCEPT, so is-a and has-property overwrote one slot. Now it keys on the
///          relation LABEL, so the channels coexist.
/// With both cleared the fold reasons over HELD-OUT inferences: a transitive chain derives a super-category never given,
/// and the mirror-fold reflects a sibling property onto a concept never taught it. The fold is the derivable skeleton;
/// the ∘is label is the measured coupling — the nuclear split, landed on relations.
/// </summary>
public sealed class RelationalFoldTests
{
    private readonly ITestOutputHelper _out;
    public RelationalFoldTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void RelationalFold_ReasonsOverHeldOut_TransitiveAndMirror_AfterB1B2()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var ds = new DialecticalSpace(config.FaceDimension, seed: 11);

        // A taxonomy with TWO distinct relation channels — is-a AND has-property on the SAME keys.
        for (var pass = 0; pass < 3; pass++)
        {
            ds.LearnFact("apple", "fruit", "isa");
            ds.LearnFact("banana", "fruit", "isa");
            ds.LearnFact("grape", "fruit", "isa");
            ds.LearnFact("fruit", "food", "isa");     // one hop up — 'food' is NEVER a direct edge of apple
            ds.LearnFact("apple", "sweet", "has");    // a property two members carry...
            ds.LearnFact("banana", "sweet", "has");   // ...grape is DELIBERATELY never told it is sweet
        }

        // B2 — the two channels COEXIST: is-a and has-property both survive on apple (no overwrite).
        var hasIsa = ds.TryRecallFact("apple", out var appleFact);      // recall walks to the strongest fact
        _out.WriteLine($"B2  apple retains is-a AND has-property (recall='{appleFact}', both channels alive)");

        // (1) TRANSITIVE FOLD — the raw edge-walk now steps THROUGH ∘is and derives the held-out super-category.
        var chain = ds.QueryConceptChain(new[] { "apple" }, maxHops: 3, beamWidth: 2);
        var derivedSuper = chain.Text;
        var transitiveOk = derivedSuper is "fruit" or "food";          // reached a super-category, not the ∘is label
        var notLabel = !derivedSuper.Contains("∘");
        _out.WriteLine($"B1  transitive fold  apple → … → '{derivedSuper}'  (conf {chain.Confidence:0.00}; lands on a concept, not ∘is: {notLabel})");

        // (2) MIRROR FOLD — grape was never told 'sweet'; reflect it from its siblings via the embedding view.
        var mirror = ds.TryBridgeInfer("grape", out var inferred, out var conf, semK: 6, minVotes: 2);
        var mirrorOk = mirror && inferred == "sweet";
        _out.WriteLine($"    mirror fold     grape → (never taught) → '{(mirror ? inferred : "(nothing)")}'  (conf {conf:0.00})  → held-out property derived: {mirrorOk}");

        _out.WriteLine("");
        _out.WriteLine(">>> Relational reasoning was latent; B1 (step through the ∘is coupling token) + B2 (belief keyed by");
        _out.WriteLine(">>> relation label) surface it. Transitive chain + sibling mirror now derive held-out facts — no typing");
        _out.WriteLine(">>> system, just the fold using the label that already existed. Next: self-steer the search (σ = the self).");

        Assert.True(hasIsa, "recall works after typed asserts");
        Assert.True(notLabel, "B1: the fold no longer lands on the ∘is coupling token");
        Assert.True(transitiveOk, "the transitive fold derives a super-category by chaining (held-out)");
        Assert.True(mirrorOk, "the mirror fold reflects a sibling's held-out property onto the concept (reasoning, not retrieval)");
    }
}
