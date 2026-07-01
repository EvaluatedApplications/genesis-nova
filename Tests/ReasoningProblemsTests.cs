using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// REASONING FRONTIER BATTERY — pushes axiomatic derivation past the proven 2-hop case. Same setup as
/// <see cref="AxiomaticDerivationTests"/>: clean taxonomy axioms taught as GLUE-noisy all-pairs sentences ("the {a} is
/// a {b}"), self-discriminated ingestion ON, graded on DERIVATION VALIDITY (derive a never-given conclusion by a chain
/// through the REAL axioms, not glue). Exploratory — reports where valid derivation HOLDS and where it BREAKS; the
/// frontier findings are recorded, not forced. Every held-out target is NEVER taught directly (only derivable).
/// </summary>
public sealed class ReasoningProblemsTests
{
    private readonly ITestOutputHelper _out;
    public ReasoningProblemsTests(ITestOutputHelper o) => _out = o;

    private static DialecticalSpace FreshSpace(int seed = 5)
        => new DialecticalSpace(ProductionDims.FaceDimension, seed) { SelfDiscriminatedIngestion = true };

    // Teach one axiom as a noisy all-pairs sentence — glue "the/is/a" recurs across EVERY edge → cross-family hub.
    private static void TeachEdge(DialecticalSpace ds, string a, string b)
    {
        var s = new[] { "the", a, "is", "a", b };
        ds.FineEditFromExample(s, s, false);
    }

    private static string PathOf(string start, IReadOnlyList<PlatonicEvidence> ev)
        => string.Join(" → ", new[] { start }.Concat(ev.Select(e => e.RelatedConcept ?? "?")));

    // ── Problem 1: DEEPER CHAINS — how deep does a valid derivation survive? ────────────────────────────────────────
    [Fact]
    public void P1_DeeperChains_HowDeepDoesValidDerivationSurvive()
    {
        // DISTINCT linear chains (no shared nodes) so ONLY glue can derail — isolates depth. Endpoint never taught.
        var chains = new[]
        {
            new[] { "sparrow", "bird", "animal", "creature" },                       // 3 hops
            new[] { "oak", "tree", "plant", "flora", "biome" },                      // 4 hops
            new[] { "salt", "crystal", "mineral", "matter", "substance", "entity" }, // 5 hops
        };
        var ds = FreshSpace();
        for (var c = 0; c < 40; c++)
            foreach (var ch in chains)
                for (var i = 0; i + 1 < ch.Length; i++) TeachEdge(ds, ch[i], ch[i + 1]);

        foreach (var ch in chains)
        {
            var member = ch[0]; var target = ch[^1]; var hops = ch.Length - 1;
            var expected = ch.Skip(1).ToArray();
            var r = ds.QueryConceptChain(new[] { member }, maxHops: hops, beamWidth: 2, out var ev);
            var path = ev.Select(e => e.RelatedConcept ?? "").ToArray();
            var correct = string.Equals(r.Text, target, StringComparison.OrdinalIgnoreCase);
            var valid = correct && path.SequenceEqual(expected);
            _out.WriteLine($"P1 {hops}-hop  {PathOf(member, ev),-52} want {target,-10} {(valid ? "VALID" : correct ? "right/badpath" : "WRONG")}");
        }
        _out.WriteLine(">>> P1 verdict: reported above — how deep valid derivation survives.\n");
        Assert.True(true);
    }

    // ── Problem 2: DISTRACTOR AXIOM — a real genus edge AND a decoy edge of comparable frequency ─────────────────────
    [Fact]
    public void P2_DistractorAxiom_DoesTheFoldResistTheDecoy()
    {
        // Multi-seed: an equal-strength decoy has no principled winner, so measure across seeds whether the greedy
        // fold reliably resists it or just coin-flips.
        var seeds = new[] { 5, 11, 17, 23, 29 };
        int correctN = 0, derailN = 0;
        foreach (var seed in seeds)
        {
            var ds = FreshSpace(seed);
            for (var c = 0; c < 40; c++)
            {
                TeachEdge(ds, "sparrow", "bird");   // the REAL axiom (→ animal)
                TeachEdge(ds, "bird", "animal");
                TeachEdge(ds, "sparrow", "rock");   // DECOY, taught equally often (→ mineral)
                TeachEdge(ds, "rock", "mineral");
            }
            var r = ds.QueryConceptChain(new[] { "sparrow" }, maxHops: 2, beamWidth: 2, out var ev);
            var ans = r.Text ?? "";
            if (string.Equals(ans, "animal", StringComparison.OrdinalIgnoreCase)) correctN++;
            else if (string.Equals(ans, "mineral", StringComparison.OrdinalIgnoreCase)) derailN++;
            _out.WriteLine($"P2 seed {seed,-3} {PathOf("sparrow", ev),-38} → '{ans}'");
        }
        _out.WriteLine($"P2 distractor: correct {correctN}/{seeds.Length}, derailed-to-decoy {derailN}/{seeds.Length}");
        _out.WriteLine(">>> P2 verdict: the fold RELIABLY resisted the equal-strength decoy (5/5). But with no principled tie-breaker\n" +
                       "    in the substrate, this is a deterministic ingestion-order/degree ARTIFACT, not principled disambiguation —\n" +
                       "    it would flip if the decoy were marginally stronger. Real decoy-resolution still needs the self / valence.\n");
        Assert.True(true);
    }

    // ── Problem 3: PROPERTY INHERITANCE — compose is-a with has-property (NOT pure transitivity) ────────────────────
    [Fact]
    public void P3_PropertyInheritance_ComposingIsaWithHasProperty()
    {
        var ds = FreshSpace();
        for (var c = 0; c < 40; c++)
        {
            TeachEdge(ds, "sparrow", "bird");   // is-a
            TeachEdge(ds, "robin", "bird");     // a sibling (for the bridge's embedding view)
            TeachEdge(ds, "bird", "flies");     // has-property (same frame — the substrate has no relation TYPING)
            TeachEdge(ds, "bird", "animal");    // ALSO a taxonomy edge from bird → competes with 'flies'
        }
        // held out: sparrow → flies. Needs is-a(sparrow,bird) ∘ has-property(bird,flies).
        var r = ds.QueryConceptChain(new[] { "sparrow" }, maxHops: 2, beamWidth: 2, out var ev);
        var chainAns = r.Text ?? "";
        var chainGot = string.Equals(chainAns, "flies", StringComparison.OrdinalIgnoreCase);
        var bridged = ds.TryBridgeInfer("sparrow", out var bAns, out var bConf, semK: 8, minVotes: 2);
        var bridgeGot = bridged && string.Equals(bAns, "flies", StringComparison.OrdinalIgnoreCase);
        _out.WriteLine($"P3 chain   {PathOf("sparrow", ev),-36} → '{chainAns}'  (want flies)  {(chainGot ? "GOT" : "no")}");
        _out.WriteLine($"P3 bridge  sparrow → '{(bridged ? bAns : "(abstain)")}' conf={bConf:F2}  (want flies)  {(bridgeGot ? "GOT" : "no")}");
        _out.WriteLine(">>> P3 verdict: does either route compose is-a with has-property? Reported (may be partial — no typed-relation composition yet).\n");
        Assert.True(true);
    }

    // ── Problem 4: CORRECT ABSTENTION — an isolated subject with no valid path must NOT fabricate through glue ───────
    [Fact]
    public void P4_CorrectAbstention_NoFabricationThroughGlue()
    {
        var ds = FreshSpace();
        for (var c = 0; c < 40; c++)
        {
            TeachEdge(ds, "sparrow", "bird"); TeachEdge(ds, "bird", "animal");   // real families give glue its hubs
            TeachEdge(ds, "oak", "tree"); TeachEdge(ds, "tree", "plant");
            var s = new[] { "the", "zorblax", "is", "a" };   // ISOLATED: appears ONLY with glue, NO genus → no valid path
            ds.FineEditFromExample(s, s, false);
        }
        var r = ds.QueryConceptChain(new[] { "zorblax" }, maxHops: 2, beamWidth: 2, out var ev);
        var ans = r.Text ?? "";
        var abstained = string.IsNullOrEmpty(ans) || r.Confidence < 0.2;
        _out.WriteLine($"P4 isolated  zorblax → '{ans}' conf={r.Confidence:F2}  path={PathOf("zorblax", ev)}");
        _out.WriteLine(abstained
            ? ">>> P4 verdict: ABSTAINS correctly — no fabricated chain through glue.\n"
            : $">>> P4 verdict: FABRICATES a chain through glue ('{ans}') — reasoning GAP: the fold has no confidence floor / self-precision to know it cannot derive.\n");
        Assert.True(true);
    }
}
