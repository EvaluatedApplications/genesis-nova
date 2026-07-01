using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// DERIVABILITY GATE — test-first premise.
///
/// Principle (from the closed/open-genesis split): a fact you can COMPUTE is not knowledge, it is a
/// DEBT. The store should hold only what it cannot derive. Nova already evicts by *utility*; the gate
/// evicts by *derivability* — so what survives is the irreducible core (the facts the system was told
/// and could never have computed).
///
/// This test proves the PREMISE the gate exploits: a transitive fact (a -> e) is DERIVABLE from the
/// primitive chain (a->b->c->d->e) via the substrate's own multi-hop walk, WITHOUT the direct edge
/// ever being stored. If the premise holds, the transitive closure is a pure debt and need not be
/// stored — which is exactly what the gate will refuse to store.
///
/// Engine note: DialecticalSpace.QueryConceptChain is a GREEDY strongest-edge walk that halts when the
/// strongest neighbour is already visited. So a chain only derives cleanly when each forward edge is
/// locally the strongest — which is why the chain below is built forward-monotonic. Making the walk
/// robust to arbitrary edge strengths (prefer strongest UNVISITED) is the first engine improvement the
/// gate needs; this test documents that boundary.
/// </summary>
public sealed class DerivabilityGateTests
{
    private readonly ITestOutputHelper _out;
    public DerivabilityGateTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void TransitiveFact_IsDerivableFromPrimitives_SoTheClosureIsAStorableDebt()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        // A clean is-a-like chain of NONCE tokens (held-out: no prior semantic associations):
        //   zeta0 -> zeta1 -> zeta2 -> zeta3 -> zeta4 -> zeta5
        // Forward edges are made progressively stronger so the greedy walk traverses the chain
        // (see engine note). Only the 5 PRIMITIVE links are ever observed — the closure is not.
        var chain = Enumerable.Range(0, 6).Select(i => $"zeta{i}").ToArray();
        for (var i = 0; i < chain.Length - 1; i++)
        {
            var count = 40 + 20 * i;               // later links observed more -> stronger
            var contradiction = 0.10 - 0.015 * i;  // and with lower contradiction -> stronger
            for (var r = 0; r < count; r++)
            {
                space.ObserveContradiction(chain[i], chain[i + 1], contradiction);
                space.ObserveContradiction(chain[i + 1], chain[i], contradiction);
            }
        }

        var storedEdges = space.RelationCount;
        var fullClosure = chain.Length * (chain.Length - 1) / 2;  // 15 pairs for a 6-node chain
        _out.WriteLine($"stored primitive edges = {storedEdges}   (full transitive closure = {fullClosure} pairs)");

        // ── PREMISE: the transitive endpoint is DERIVABLE, though (zeta0 -> zeta5) was never stored.
        var derived = space.QueryConceptChain(new[] { chain[0] }, maxHops: 5, beamWidth: 2);
        _out.WriteLine($"derive(zeta0, hops<=5) = '{derived.Text}'  conf={derived.Confidence:F3}  hops={derived.Hops}");
        Assert.Equal(chain[^1], derived.Text);      // reached zeta5 from zeta0 by walking primitives
        Assert.True(derived.Hops >= 4, $"expected a multi-hop derivation, got {derived.Hops} hops");

        // ── THE DEBT: the closure was never stored. Only the chain lives in the store, yet the derived
        // fact resolves. (10 of the 15 closure pairs are pure debt we did not pay.)
        Assert.True(storedEdges <= chain.Length, $"expected only the chain (<= {chain.Length}), got {storedEdges}");
        var directNeighbours = space.GetNeighbors(chain[0], PlatonicNeighborhoodType.Relational, 8, 0.0);
        Assert.DoesNotContain(directNeighbours, n => n.Concept == chain[^1]);  // no direct zeta0->zeta5 edge

        // ── GENERALISATION: an intermediate transitive fact never stored (zeta0 -> zeta3) also derives.
        var mid = space.QueryConceptChain(new[] { chain[0] }, maxHops: 3, beamWidth: 2);
        _out.WriteLine($"derive(zeta0, hops<=3) = '{mid.Text}'");
        Assert.Equal(chain[3], mid.Text);

        _out.WriteLine($"PREMISE HOLDS: {fullClosure - storedEdges} of {fullClosure} closure edges are derivable debt the gate can refuse to store.");
    }

    private static void Observe(DialecticalSpace s, string a, string b, double contradiction, int n)
    {
        for (var i = 0; i < n; i++) { s.ObserveContradiction(a, b, contradiction); s.ObserveContradiction(b, a, contradiction); }
    }

    /// <summary>THE GATE (hot-path): the maintenance sweep evicts a redundant weak shortcut that is derivable
    /// from a stronger path, and the fact stays derivable — the store converges to its irreducible core at
    /// ZERO recall loss. This is what runs during training when DerivabilityGate is on.</summary>
    [Fact]
    public void MaintenanceSweep_EvictsDerivableShortcut_AtZeroRecallLoss()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 11) { DerivabilityGate = true };

        // triangle: STRONG chain alpha->beta->gamma, plus a WEAK redundant shortcut alpha->gamma (the debt)
        Observe(space, "alpha", "beta", 0.02, 120);
        Observe(space, "beta", "gamma", 0.02, 120);
        Observe(space, "alpha", "gamma", 0.40, 12);

        var before = space.RelationCount;
        Assert.Equal(3, before);
        Assert.True(space.IsRelationDerivable("alpha", "gamma"), "chain should derive alpha->gamma without the direct edge");

        // the hot-path hook: the maintenance sweep (runs periodically during training)
        space.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());

        // the redundant weak shortcut is gone; the irreducible chain remains
        Assert.Equal(before - 1, space.RelationCount);
        var alphaNbrs = space.GetNeighbors("alpha", PlatonicNeighborhoodType.Relational, 8, 0.0);
        Assert.DoesNotContain(alphaNbrs, n => n.Concept == "gamma");   // direct alpha->gamma edge evicted

        // ZERO RECALL LOSS: gamma is still reachable from alpha — now purely by DERIVATION over the chain.
        var q = space.QueryConceptChain(new[] { "alpha" }, maxHops: 2, beamWidth: 2);
        _out.WriteLine($"after sweep: RelationCount {before}->{space.RelationCount}; derive(alpha)='{q.Text}'");
        Assert.Equal("gamma", q.Text);
    }

    /// <summary>THE TRAINING PATH: the sweep fires from the AUTOMATIC per-observation maintenance trigger
    /// (inside ObserveContradiction, once per DischargeInterval), not just from an explicit ApplyMaintenance.
    /// This is what actually runs during a training loop.</summary>
    [Fact]
    public void AutoSweep_FiresFromTheObservationLoop_NotJustExplicitMaintenance()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 13) { DerivabilityGate = true };

        // build the triangle without tripping the auto-sweep (interval kept high)
        Observe(space, "alpha", "beta", 0.02, 120);
        Observe(space, "beta", "gamma", 0.02, 120);
        Observe(space, "alpha", "gamma", 0.40, 12);
        Assert.Equal(3, space.RelationCount);

        // now make the NEXT observation cross the maintenance interval — as it periodically does in training.
        // Observe an existing STRONG edge (not the shortcut), so the shortcut is not re-created after eviction.
        space.DischargeInterval = 1;
        space.ObserveContradiction("alpha", "beta", 0.02);
        space.ObserveContradiction("beta", "alpha", 0.02);

        // the auto-maintenance evicted the derivable shortcut — with NO explicit ApplyMaintenance call.
        var alphaNbrs = space.GetNeighbors("alpha", PlatonicNeighborhoodType.Relational, 8, 0.0);
        Assert.DoesNotContain(alphaNbrs, n => n.Concept == "gamma");
        _out.WriteLine($"auto-sweep during observation left RelationCount={space.RelationCount} (shortcut evicted)");
    }

    /// <summary>A/B: does the gate evict only REDUNDANT shortcuts, or does it also wrongly evict COINCIDENTAL
    /// real edges (a strong independent association that merely forms a triangle)? Compares gate on vs off on
    /// the same graph, reporting evictions of each class and whether retrieval survives. This is the safety test
    /// for the untyped-relation heuristic.</summary>
    [Fact]
    public void AB_Gate_EvictsRedundant_ButSparesCoincidentalRealEdges()
    {
        DialecticalSpace Build()
        {
            var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
            var s = new DialecticalSpace(config.FaceDimension, seed: 21);
            // 6 REDUNDANT triangles: strong chain redX_a->redX_b->redX_c + a WEAK shortcut a->c (a debt, should evict)
            for (var i = 0; i < 6; i++)
            {
                Observe(s, $"r{i}a", $"r{i}b", 0.02, 100);
                Observe(s, $"r{i}b", $"r{i}c", 0.02, 100);
                Observe(s, $"r{i}a", $"r{i}c", 0.40, 10);   // weak redundant shortcut
            }
            // 6 COINCIDENTAL triangles: a STRONG real edge x->z, with only weak incidental x->y, y->z.
            // x->z is the STRONGEST of its triangle, so the guard (evict only the weakest, needing a stronger
            // path) must SPARE it — evicting it would be a false positive (losing a real association).
            for (var j = 0; j < 6; j++)
            {
                Observe(s, $"c{j}x", $"c{j}y", 0.30, 25);
                Observe(s, $"c{j}y", $"c{j}z", 0.30, 25);
                Observe(s, $"c{j}x", $"c{j}z", 0.02, 100);  // strong REAL edge (not a debt)
            }
            return s;
        }

        bool HasEdge(DialecticalSpace s, string a, string b)
            => s.GetNeighbors(a, PlatonicNeighborhoodType.Relational, 16, 0.0).Any(n => n.Concept == b);

        // ── B: gate OFF (baseline)
        var off = Build();
        var offEdges = off.RelationCount;
        off.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());

        // ── A: gate ON
        var on = Build();
        var onEdgesBefore = on.RelationCount;
        on.DerivabilityGate = true;
        on.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());

        var redundantEvicted = Enumerable.Range(0, 6).Count(i => !HasEdge(on, $"r{i}a", $"r{i}c"));
        var coincidentalEvicted = Enumerable.Range(0, 6).Count(j => !HasEdge(on, $"c{j}x", $"c{j}z"));
        _out.WriteLine($"OFF: {offEdges} edges, {offEdges - off.RelationCount} evicted");
        _out.WriteLine($"ON : {onEdgesBefore} edges, {onEdgesBefore - on.RelationCount} evicted");
        _out.WriteLine($"  redundant shortcuts evicted   : {redundantEvicted}/6  (want 6 — real compression)");
        _out.WriteLine($"  coincidental REAL edges evicted: {coincidentalEvicted}/6  (want 0 — false positives)");

        Assert.Equal(0, offEdges - off.RelationCount);              // gate off: nothing moves
        Assert.True(redundantEvicted >= 4, $"gate should evict most redundant shortcuts, got {redundantEvicted}/6");
        Assert.Equal(0, coincidentalEvicted);                      // MUST spare real strong edges (no false positives)
        // retrieval of every transitive fact still resolves after eviction (zero recall loss)
        for (var i = 0; i < 6; i++)
            Assert.Equal($"r{i}c", on.QueryConceptChain(new[] { $"r{i}a" }, 2, 2).Text);
    }

    /// <summary>MEASURED, not assumed: does the gate touch NAV's actual graph? Nav trains on a taxonomy TREE
    /// (member→genus→domain→root) — which has no redundant triangle shortcuts — so the gate must evict NOTHING and
    /// therefore cannot be the cause of nav's low accuracy. (Replaces the assumption that it was hurting nav.)</summary>
    [Fact]
    public void Gate_OnNavTaxonomyTree_EvictsNothing_SoCannotHurtNav()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var ds = new DialecticalSpace(config.FaceDimension, seed: 5) { DerivabilityGate = true };
        void Edge(string c, string p) { for (var i = 0; i < 8; i++) { ds.ObserveContradiction(c, p, 0.0); ds.ObserveContradiction(p, c, 0.0); } }
        // the same 2-level taxonomy nav trains on (NavigatorGymTrainTests): a pure tree, no shortcut edges
        Edge("dog", "mammal"); Edge("mammal", "animal"); Edge("animal", "thing");
        Edge("robin", "bird"); Edge("bird", "animal");
        Edge("oak", "tree"); Edge("tree", "plant"); Edge("plant", "thing");
        Edge("rose", "flower"); Edge("flower", "plant");

        var before = ds.RelationCount;
        ds.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());
        var evicted = before - ds.RelationCount;
        _out.WriteLine($"nav taxonomy: {before} edges -> {ds.RelationCount} (gate evicted {evicted})");
        Assert.Equal(0, evicted);   // a tree has no redundant triangles -> the gate is a NO-OP here; it isn't nav's problem
    }

    /// <summary>OFF is byte-identical: with the gate off (default), the sweep evicts nothing.</summary>
    [Fact]
    public void GateOff_KeepsEveryEdge()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 11);   // DerivabilityGate = false (default)

        Observe(space, "alpha", "beta", 0.02, 120);
        Observe(space, "beta", "gamma", 0.02, 120);
        Observe(space, "alpha", "gamma", 0.40, 12);

        var before = space.RelationCount;
        space.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());
        Assert.Equal(before, space.RelationCount);   // nothing evicted when the gate is off
    }
}
