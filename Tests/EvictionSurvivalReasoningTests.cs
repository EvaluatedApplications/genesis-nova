using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// IS REASONING LATENT-SPACE OR ELEMENT-WALKING? The transitive fold derives member→kingdom by WALKING stored relation
/// edges through the intermediate genus (sparrow→bird→animal). The user's thesis: elements are a CACHE over the conserved
/// latent space; a fully-mapped substrate should reason from the GEOMETRY even after an element disappears. So we teach
/// the clean axioms, then EVICT the intermediate node (bird/tree) — which drops the node AND its edges — and ask whether
/// the derivation survives: (a) via the chain at all, (b) via decode-from-void re-materialisation, (c) from the raw
/// latent geometry (nearest-concept / relaxation). This MEASURES how latent the reasoning is vs how much it leans on the
/// discrete element cache. Reported, not forced.
/// </summary>
public sealed class EvictionSurvivalReasoningTests
{
    private readonly ITestOutputHelper _out;
    public EvictionSurvivalReasoningTests(ITestOutputHelper o) => _out = o;

    private static readonly (string member, string genus, string kingdom)[] Families =
    {
        ("sparrow", "bird", "animal"),
        ("oak",     "tree", "plant"),
    };
    private static readonly HashSet<string> Glue = new(StringComparer.OrdinalIgnoreCase) { "the", "is", "a" };

    private static DialecticalSpace Build(bool recover)
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5)
        { SelfDiscriminatedIngestion = true, RecoverFromVoid = recover };
        for (var c = 0; c < 40; c++)
            foreach (var (m, g, k) in Families)
            {
                var s1 = new[] { "the", m, "is", "a", g };   // axiom member→genus
                var s2 = new[] { "the", g, "is", "a", k };   // axiom genus→kingdom
                ds.FineEditFromExample(s1, s1, false);
                ds.FineEditFromExample(s2, s2, false);
            }
        return ds;
    }

    // maxHops:2 = the derivation depth (member→genus→kingdom); a 3rd hop only overshoots into glue. `intermediate` is the
    // node the chain routed THROUGH — a VALID derivation goes through the real genus, not glue.
    private static (string path, string answer, string intermediate) Chain(DialecticalSpace ds, string member)
    {
        var r = ds.QueryConceptChain(new[] { member }, maxHops: 2, beamWidth: 2, out var ev);
        var path = string.Join(" → ", new[] { member }.Concat(ev.Select(e => e.RelatedConcept ?? "?")));
        var intermediate = ev.Count > 0 ? (ev[0].RelatedConcept ?? "") : "";
        return (path, r.Text ?? "", intermediate);
    }

    [Fact]
    public void Reasoning_SurvivesIntermediateEviction_OrBreaks()
    {
        // BASELINE — pre-eviction, the derivation is VALID (sparrow→bird→animal), re-confirming the AxiomaticDerivation result.
        var baseline = Build(false);
        var (basePath, baseAns, baseMid) = Chain(baseline, "sparrow");
        var baseValid = baseAns == "animal" && baseMid == "bird";
        _out.WriteLine("EVICTION-SURVIVAL — is transitive reasoning latent-space or element-walking?\n");
        _out.WriteLine($"  BASELINE (pre-evict):        sparrow ⇒ '{baseAns}'   [{basePath}]   {(baseValid ? "VALID (through bird)" : "invalid")}");

        // POST-EVICT A/B — remove the INTERMEDIATE node (drops node + its edges). A VALID chain must still go through the
        // real genus 'bird' — but bird is gone, so any answer=='animal' can only be a GLUE-laundered path (invalid).
        (string path, string answer, string mid, bool contains) PostEvict(bool recover)
        {
            var ds = Build(recover);
            ds.EvictConcept("bird");
            ds.EvictConcept("tree");
            var stillThere = ds.ContainsConcept("bird");
            var (path, ans, mid) = Chain(ds, "sparrow");
            return (path, ans, mid, stillThere);
        }
        var off = PostEvict(false);
        var on = PostEvict(true);
        bool Valid(string ans, string mid) => ans == "animal" && mid == "bird";
        _out.WriteLine($"  bird element after evict:     ContainsConcept(bird) = {off.contains}");
        _out.WriteLine($"  POST-EVICT  recovery OFF:    sparrow ⇒ '{off.answer}'   [{off.path}]   {(Valid(off.answer, off.mid) ? "SURVIVED (valid)" : "BROKE (no valid chain)")}");
        _out.WriteLine($"  POST-EVICT  recovery ON:     sparrow ⇒ '{on.answer}'   [{on.path}]   {(Valid(on.answer, on.mid) ? "SURVIVED (valid)" : "BROKE (no valid chain)")}");

        // GEOMETRY-ONLY PROBE — recovery off, node+edges gone: does the raw LATENT GEOMETRY still place 'animal' near
        // 'sparrow' (the relation as a conserved geometric trace in the clouds), or is it lost with the node?
        var geo = Build(false);
        geo.EvictConcept("bird");
        geo.EvictConcept("tree");
        var near = geo.GetNearestConcepts("sparrow", maxNeighbors: 12)
            .Where(n => !Glue.Contains(n.Symbol) && !string.Equals(n.Symbol, "sparrow", StringComparison.OrdinalIgnoreCase))
            .ToList();
        var animalRank = near.FindIndex(n => string.Equals(n.Symbol, "animal", StringComparison.OrdinalIgnoreCase));
        var reason = geo.Reason(new[] { "sparrow" });
        _out.WriteLine("");
        _out.WriteLine($"  GEOMETRY-ONLY (post-evict, no recovery):");
        _out.WriteLine($"    sparrow's nearest (non-glue): {string.Join(", ", near.Take(5).Select(n => $"{n.Symbol}:{n.Distance:F2}"))}");
        _out.WriteLine($"    'animal' in sparrow's latent neighbourhood: {(animalRank >= 0 ? $"YES (rank {animalRank + 1})" : "no")}");
        _out.WriteLine($"    relaxation Reason(sparrow) settles to: '{reason.Symbol}' (conf {reason.Confidence:F2})");

        // VERDICT
        var chainSurvived = Valid(off.answer, off.mid) || Valid(on.answer, on.mid);
        var geomRetains = animalRank >= 0 || string.Equals(reason.Symbol, "animal", StringComparison.OrdinalIgnoreCase);
        _out.WriteLine("");
        _out.WriteLine(chainSurvived
            ? ">>> VERDICT: the derivation SURVIVES element eviction — reasoning is (at least partly) latent/cache-conserved."
            : geomRetains
                ? ">>> VERDICT: the CHAIN breaks (edge-walking is element-dependent), but the raw GEOMETRY still retains the relation — the latent trace outlives the node/edges. Reasoning-by-walk is element-bound; reasoning-by-geometry is not."
                : ">>> VERDICT: reasoning is ELEMENT-WALKING — evicting the intermediate drops node+edges and the relation is lost; not yet latent.");

        // Assert ONLY the baseline (a real valid derivation) + that eviction actually happened; the eviction/geometry
        // outcomes are REPORTED measurements, not forced.
        Assert.True(baseValid, "baseline derivation is valid (sparrow→bird→animal) before eviction");
        Assert.False(off.contains, "eviction actually removed the intermediate element");
    }
}
