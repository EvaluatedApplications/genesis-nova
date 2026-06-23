using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// CONTRACT-level invariants of the platonic space, encoded as PROPERTY assertions (orderings / monotone
/// directions / distinguishability) rather than magic numbers, so they survive any future substrate rework.
/// Variables are typed as the <see cref="IPlatonicSpace"/> boundary contract wherever possible and constructed
/// via <c>new PlatonicSpaceMemory(...)</c> — the tests pin the CONTRACT, not the implementation. These encode
/// the laws of PLATONIC_THEORY.md that the CURRENT substrate already satisfies and that MUST keep holding:
///   Law M (identity-by-difference, §2), Law D / Confidence = 1−κ (§3), G4 (Conservation, §5),
///   G6 (Irreversibility, §5 — see note on test 3), and the disruption direction (task-outcome backprop).
/// Production face dimension (512); fixed seeds; small and fast.
/// </summary>
public sealed class PlatonicContractTests
{
    private static IPlatonicSpace NewSpace() =>
        new PlatonicSpaceMemory(faceDimension: 512, seed: 7, maxNodes: 1000, maxRelations: 5000);

    // Confidence of an observed pair = 1 − synthesis contradiction. Higher = more related. (Law D, §3.)
    private static double Conf(IPlatonicSpace s, string a, string b) => 1.0 - s.GetContradiction(a, b);

    // Rank of `target` among `anchor`'s fresh-geometry nearest neighbours (0 = nearest). NaN-guarded via int.MaxValue.
    private static int FreshRank(IPlatonicSpace s, string anchor, string target)
    {
        var nearest = s.GetNearestConceptsFresh(anchor, seeds: null, maxNeighbors: 64);
        for (var i = 0; i < nearest.Count; i++)
            if (string.Equals(nearest[i].Symbol, target, StringComparison.OrdinalIgnoreCase))
                return i;
        return int.MaxValue; // not retrieved at all → "infinitely far"
    }

    // ── 1. Relatedness ordering ──────────────────────────────────────────────────────────────────────────
    // Law D / Confidence (§3): an element's place is the synthesis of its agreements (pulled near) and
    // contradictions (pushed far). A strongly+repeatedly low-contradiction partner must rank as MORE related
    // than a weakly/never observed one. We assert the ORDERING (relatedness and nearest-rank), never a value.
    [Fact]
    public void StronglyObservedPartner_RanksMoreRelated_ThanWeaklyObservedOne()
    {
        var s = NewSpace();

        // (a,b): agree strongly, many times.   (a,c): a single weak (high-contradiction) brush.
        for (var i = 0; i < 40; i++) s.ObserveContradiction("a", "b", 0.05);
        s.ObserveContradiction("a", "c", 0.9);

        // Relational confidence: b must be more related to a than c is.
        Assert.True(Conf(s, "a", "b") > Conf(s, "a", "c"),
            $"strong partner must out-rank weak one: conf(a,b)={Conf(s, "a", "b"):F3} conf(a,c)={Conf(s, "a", "c"):F3}");

        // The relational neighbourhood must reflect the same ordering.
        var neighbours = s.GetNeighbors("a", PlatonicNeighborhoodType.Any, maxNeighbors: 32);
        var cb = neighbours.FirstOrDefault(n => string.Equals(n.Concept, "b", StringComparison.OrdinalIgnoreCase));
        var cc = neighbours.FirstOrDefault(n => string.Equals(n.Concept, "c", StringComparison.OrdinalIgnoreCase));
        Assert.False(string.IsNullOrEmpty(cb.Concept), "the strong partner must appear in the neighbourhood");
        Assert.True(cb.Confidence > cc.Confidence,
            $"neighbour confidence must order strong>weak: b={cb.Confidence:F3} c={cc.Confidence:F3}");

        // And in the fresh geometry, the strong partner ranks at least as near (smaller rank index) as the weak one.
        Assert.True(FreshRank(s, "a", "b") <= FreshRank(s, "a", "c"),
            "the strong partner must be at least as near as the weak one in the fresh geometry");
    }

    // ── 2. Conservation direction (G4) ───────────────────────────────────────────────────────────────────
    // Property, not a number: observing a pair MORE strongly (lower contradiction, more times) yields a
    // retrieved relatedness/confidence that is NON-DECREASING relative to a single weak observation. Two
    // independent spaces with the same seed isolate "strong vs weak" as the only difference.
    [Fact]
    public void StrongerObservation_NeverDecreasesRetrievedRelatedness()
    {
        var weak = NewSpace();
        var strong = NewSpace();

        weak.ObserveContradiction("x", "y", 0.5);                                   // one weak brush
        for (var i = 0; i < 30; i++) strong.ObserveContradiction("x", "y", 0.05);   // strong, repeated

        Assert.True(Conf(strong, "x", "y") >= Conf(weak, "x", "y"),
            $"stronger observation must not lower confidence: strong={Conf(strong, "x", "y"):F3} weak={Conf(weak, "x", "y"):F3}");

        // Monotone within a single space too: re-observing strongly never reduces the standing confidence.
        var s = NewSpace();
        s.ObserveContradiction("p", "q", 0.5);
        var afterWeak = Conf(s, "p", "q");
        for (var i = 0; i < 30; i++) s.ObserveContradiction("p", "q", 0.05);
        Assert.True(Conf(s, "p", "q") >= afterWeak,
            $"reinforcing low-contradiction must not lower confidence: before={afterWeak:F3} after={Conf(s, "p", "q"):F3}");
    }

    // ── 3. Archival monotonicity (G6) ────────────────────────────────────────────────────────────────────
    // COVERED by PlatonicArchivalTests (Eviction_Archives_NeverDestroys_NoDistinctionLost +
    // ReObservingArchivedConcept_ReactivatesItIntact): NodeCount stays bounded, NodeCount+ArchivedNodeCount
    // equals the distinct concepts ever observed, and re-observing an evicted concept reactivates it. Not
    // duplicated here. Below is a DISTINCT contract-framed guarantee not asserted there: the conservation
    // identity holds at the IPlatonicSpace boundary EVEN WITHOUT any capacity pressure — a freshly observed
    // set is fully retained and active, archive empty (the monotone-only-expands invariant in its trivial case).
    [Fact]
    public void NoEviction_AllDistinctionsRetainedAndActive()
    {
        IPlatonicSpace s = new PlatonicSpaceMemory(faceDimension: 512, seed: 7, maxNodes: 1000, maxRelations: 5000);
        const int pairs = 50; // 100 distinct concepts, far under the cap → nothing should archive

        for (var i = 0; i < pairs; i++) s.ObserveContradiction($"u{i}", $"v{i}", 0.1);

        Assert.Equal(0, s.ArchivedNodeCount); // no pressure → no archival
        Assert.Equal(pairs * 2, s.NodeCount); // every distinction present and active
        Assert.Equal(pairs * 2, s.NodeCount + s.ArchivedNodeCount); // conservation identity (G6) holds trivially
    }

    // ── 4. Disruption direction ──────────────────────────────────────────────────────────────────────────
    // FunctionDisruptionTests covers the push-away MAGNITUDE (after > before by a measurable amount). The
    // distinct CONTRACT here is purely directional and stated on relatedness (not geometric distance):
    // a DisruptAssociation(anchor, wrong) call must NOT INCREASE the wrong answer's relatedness to the anchor.
    [Fact]
    public void DisruptAssociation_DoesNotIncreaseWrongAnswerRelatedness()
    {
        var s = NewSpace();
        for (var i = 0; i < 40; i++) s.ObserveContradiction("anchor", "wrong", 0.05); // made wrongly retrievable
        var before = Conf(s, "anchor", "wrong");

        for (var i = 0; i < 10; i++) s.DisruptAssociation("anchor", "wrong"); // task-wrong outcome → repel/leave

        var after = Conf(s, "anchor", "wrong");
        Assert.True(after <= before + 1e-9,
            $"disruption must not raise the wrong answer's relatedness: before={before:F3} after={after:F3}");
    }

    // ── 5. Differential identity (Law M smoke, §2) ───────────────────────────────────────────────────────
    // "Meaning is nothing but the structure of differences." Two distinct, differently-observed concepts must
    // be DISTINGUISHABLE — their positions (faces) differ, so they are not the same point and not equidistant
    // to everything. We assert distinguishability (faces differ; one is strictly nearer a shared probe), not a
    // particular separation value.
    [Fact]
    public void DistinctlyObservedConcepts_HaveDistinguishablePositions()
    {
        var s = NewSpace();

        // Observe two concepts into DIFFERENT neighbourhoods of a shared probe.
        for (var i = 0; i < 30; i++) s.ObserveContradiction("alpha", "probe", 0.05); // alpha agrees with probe
        for (var i = 0; i < 30; i++) s.ObserveContradiction("beta", "probe", 0.95);  // beta opposes probe

        Assert.True(s.TryGetConceptFace("alpha", out var fa), "alpha must have a position");
        Assert.True(s.TryGetConceptFace("beta", out var fb), "beta must have a position");
        Assert.Equal(fa.Length, fb.Length);

        // Faces are not byte-identical: the two concepts occupy different points (indiscernibles ⇒ identical;
        // discernible observations ⇒ distinct positions).
        var identical = true;
        for (var i = 0; i < fa.Length && identical; i++)
            if (Math.Abs(fa[i] - fb[i]) > 1e-12) identical = false;
        Assert.False(identical, "distinctly observed concepts must not share an identical position (Law M)");

        // And they are not equidistant to everything: relative to the shared probe, the agreeing concept is
        // strictly more related than the opposing one.
        Assert.True(Conf(s, "alpha", "probe") > Conf(s, "beta", "probe"),
            $"differently observed concepts must differ in relation to a shared foil: " +
            $"alpha={Conf(s, "alpha", "probe"):F3} beta={Conf(s, "beta", "probe"):F3}");
    }
}
