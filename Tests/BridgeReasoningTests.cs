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
/// BRIDGE-DIMENSIONS reasoning (ported concept from the original genesis engine's PortalBridge: reconcile the
/// SYMBOLIC GRAPH view with the EMBEDDING view). Empirical question first, before any hot-path wiring: does bridging
/// to the embedding space let the substrate REASON something the relation graph ALONE cannot?
///
/// Test = inductive category inference on a HELD-OUT member. Known fruits carry (category + shared attributes); a
/// held-out "grape" carries ONLY the shared attributes and NO category edge. The graph therefore cannot answer
/// "grape is a fruit". The bridge: grape's EMBEDDING neighbours are the known fruits (shared-attribute clouds
/// overlap), and their category — the property grape doesn't yet have — is fruit. If nova's (graph-derived)
/// embedding still supports this, the bridge adds real reasoning; if not, it's an honest no.
/// </summary>
public sealed class BridgeReasoningTests
{
    private readonly ITestOutputHelper _out;
    public BridgeReasoningTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Bridge_InfersHeldOutMemberCategory_WhereGraphAloneCannot()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var s = new DialecticalSpace(config.FaceDimension, seed: 9);
        void O(string a, string b) { for (var i = 0; i < 20; i++) { s.ObserveContradiction(a, b, 0.05); s.ObserveContradiction(b, a, 0.05); } }

        // known fruits: CATEGORY + shared attributes
        O("apple", "fruit"); O("apple", "sweet"); O("apple", "food"); O("apple", "round");
        O("banana", "fruit"); O("banana", "sweet"); O("banana", "food");
        O("cherry", "fruit"); O("cherry", "sweet"); O("cherry", "round");
        // HELD-OUT member: shared attributes ONLY, NO category edge
        O("grape", "sweet"); O("grape", "food"); O("grape", "round");
        s.FlushCloudBatch();

        // ── GRAPH-ONLY: grape has no "fruit" relation → the graph cannot answer.
        var graph = s.GetNeighbors("grape", PlatonicNeighborhoodType.Relational, 16, 0.0)
            .Select(n => n.Concept).ToHashSet(StringComparer.Ordinal);
        _out.WriteLine($"grape graph neighbours: {string.Join(",", graph)}");
        Assert.DoesNotContain("fruit", graph);

        // ── BRIDGE: grape's EMBEDDING neighbours, then THE category they carry that grape does NOT yet have.
        var sem = s.GetNeighbors("grape", PlatonicNeighborhoodType.Semantic, 6, 0.0).Select(n => n.Concept).ToList();
        _out.WriteLine($"grape embedding neighbours: {string.Join(",", sem)}");
        var votes = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var nb in sem)
        {
            if (nb == "grape" || graph.Contains(nb)) continue;
            foreach (var cat in s.GetNeighbors(nb, PlatonicNeighborhoodType.Relational, 8, 0.0).Select(n => n.Concept))
                if (cat != "grape" && !graph.Contains(cat))        // the NEW property, not a shared attribute
                    votes[cat] = votes.GetValueOrDefault(cat) + 1;
        }
        var inferred = votes.Count > 0 ? votes.OrderByDescending(kv => kv.Value).First().Key : "(none)";
        _out.WriteLine($"bridge inferred category: '{inferred}'  votes: {string.Join(", ", votes.OrderByDescending(k => k.Value).Take(5).Select(k => $"{k.Key}:{k.Value}"))}");

        Assert.Equal("fruit", inferred);   // the bridge REASONS grape→fruit via semantically-similar known members
    }

    // Delegates to the REAL substrate method, so the A/B measures production code (not test-local logic).
    private static string BridgedCategory(DialecticalSpace s, string x)
        => s.TryBridgeInfer(x, out var ans, out _) ? ans : "(none)";

    /// <summary>RESOLVE-RATE A/B across many held-out members: graph-only can answer NONE (no category edges); the
    /// bridge should answer MOST. This is the "does it increase reasoning skills" measurement.</summary>
    [Fact]
    public void Bridge_LiftsHeldOutCategoryResolveRate_FarAboveGraphOnly()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var s = new DialecticalSpace(config.FaceDimension, seed: 4);
        void O(string a, string b) { for (var i = 0; i < 20; i++) { s.ObserveContradiction(a, b, 0.05); s.ObserveContradiction(b, a, 0.05); } }

        // three categories, each: KNOWN members carry (category + attributes); HELD-OUT members carry attributes ONLY.
        var cats = new (string cat, string[] attrs, string[] known, (string m, string[] a)[] heldOut)[]
        {
            ("fruit",  new[]{"sweet","juicy","food"},   new[]{"apple","banana","cherry","pear"}, new[]{ ("grape", new[]{"sweet","juicy"}), ("plum", new[]{"sweet","food"}) }),
            ("animal", new[]{"breathes","moves","eats"}, new[]{"dog","cat","horse","cow"},        new[]{ ("wolf", new[]{"breathes","moves"}), ("deer", new[]{"moves","eats"}) }),
            ("tool",   new[]{"metal","held","works"},    new[]{"hammer","wrench","saw","drill"},  new[]{ ("pliers", new[]{"metal","held"}), ("chisel", new[]{"held","works"}) }),
        };
        foreach (var (cat, attrs, known, _) in cats)
            foreach (var m in known) { O(m, cat); foreach (var a in attrs) O(m, a); }
        foreach (var (_, _, _, heldOut) in cats)
            foreach (var (m, a) in heldOut) foreach (var at in a) O(m, at);
        s.FlushCloudBatch();

        int graphHits = 0, bridgeHits = 0, total = 0;
        foreach (var (cat, _, _, heldOut) in cats)
            foreach (var (m, _) in heldOut)
            {
                total++;
                var g = s.GetNeighbors(m, PlatonicNeighborhoodType.Relational, 16, 0.0).Any(n => n.Concept == cat);
                var b = BridgedCategory(s, m) == cat;
                if (g) graphHits++;
                if (b) bridgeHits++;
                _out.WriteLine($"{m}: graph={(g ? cat : "—")}  bridge={BridgedCategory(s, m)}  (true={cat})");
            }
        _out.WriteLine($"RESOLVE RATE  graph-only {graphHits}/{total}   bridged {bridgeHits}/{total}");

        Assert.Equal(0, graphHits);                       // held-outs have no category edge → graph answers none
        Assert.True(bridgeHits >= total - 1, $"bridge should resolve nearly all held-out categories, got {bridgeHits}/{total}");
    }

    /// <summary>WHICH open-ended reasoning trainers does the bridge lift? Exercises the query SHAPES the gym trains —
    /// item→category (gym), member→genus (nav taxonomy), word→group (synonym) — on held-out instances, and reports
    /// per-shape resolve. Property-shaped reasoning (category, taxonomy, group-membership) is the bridge's domain.</summary>
    [Fact]
    public void Bridge_AcrossReasoningTrainerShapes_ReportsWhichItLifts()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var s = new DialecticalSpace(config.FaceDimension, seed: 3);
        void O(string a, string b) { for (var i = 0; i < 20; i++) { s.ObserveContradiction(a, b, 0.05); s.ObserveContradiction(b, a, 0.05); } }

        // CATEGORY (gym item→category): shared attrs + category; held-out item = attrs only
        foreach (var m in new[] { "apple", "banana", "cherry" }) { O(m, "fruit"); O(m, "sweet"); O(m, "food"); }
        O("grape", "sweet"); O("grape", "food");
        // TAXONOMY (nav member→genus): shared attrs + genus; held-out = attrs only
        foreach (var m in new[] { "dog", "cat", "horse" }) { O(m, "mammal"); O(m, "furry"); O(m, "warm"); }
        O("wolf", "furry"); O("wolf", "warm");
        // SYNONYM (gym word→group): shared meaning attr + group tag; held-out = attr only
        foreach (var w in new[] { "big", "large", "huge" }) { O(w, "sizegroup"); O(w, "sizey"); }
        O("giant", "sizey");
        s.FlushCloudBatch();

        (string shape, string subj, string want)[] cases =
        {
            ("category", "grape", "fruit"),
            ("taxonomy", "wolf",  "mammal"),
            ("synonym",  "giant", "sizegroup"),
        };
        var hits = 0;
        foreach (var (shape, subj, want) in cases)
        {
            var ok = s.TryBridgeInfer(subj, out var ans, out var conf);
            var hit = ok && ans == want; if (hit) hits++;
            _out.WriteLine($"{shape,-9} {subj,-6} -> {(ok ? ans : "(abstain)"),-10} conf={conf:F2}  want={want}  {(hit ? "HIT" : "MISS")}");
        }
        _out.WriteLine($"bridge lifts {hits}/{cases.Length} open-ended reasoning shapes");
        // category + taxonomy are the bridge's core domain — assert those; synonym is reported.
        Assert.True(s.TryBridgeInfer("grape", out var a1, out _) && a1 == "fruit", "category shape");
        Assert.True(s.TryBridgeInfer("wolf", out var a2, out _) && a2 == "mammal", "taxonomy shape");
    }
}
