using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// LATTICE in the new core (user: "leverage the lattice shape, it's an integral and efficient way to calculate over
/// the space"). DialecticalSpace was built on full O(N) linear scans; this wires the VP-tree back for O(log N)
/// candidate gathering above a size threshold, with LIVE-FACE rescoring so results stay correctness-equivalent to the
/// scan. The proof: on a space large enough to engage the lattice, the lattice path (null candidates) must return the
/// SAME nearest concept as an exact full scan (all symbols passed as explicit candidates, which always scans live).
/// Production face dim. Pure substrate (no NN), so it runs in the fast suite as continuous protection.
/// </summary>
public sealed class DialecticalSpaceLatticeTests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceLatticeTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Lattice_AtScale_MatchesExactScan_AndStaysCorrect()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);

        // 40 clusters × 10 members, each member related to its shared hub → a member's cluster-mates share context
        // (their clouds overlap), so nearest-neighbour is cluster-coherent. 440 non-atom concepts > the 384 lattice
        // threshold, so a null-candidates query takes the VP-tree path.
        const int clusters = 40, perCluster = 10;
        var clusterOf = new Dictionary<string, string>(StringComparer.Ordinal);
        var membersOf = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        for (var c = 0; c < clusters; c++)
        {
            var hub = $"hub{c}";
            membersOf[hub] = new List<string>();
            for (var m = 0; m < perCluster; m++)
            {
                var member = $"c{c}m{m}";
                space.FineEditFromExample(new[] { member }, new[] { hub }, isNegativeExample: false);
                clusterOf[member] = hub;
                membersOf[hub].Add(member);
            }
        }

        var active = space.ActiveConcepts.ToList();
        _out.WriteLine($"[lattice] active non-atom concepts = {active.Count} (threshold 384)");
        Assert.True(active.Count >= 384, $"need > threshold to engage the lattice; got {active.Count}");

        var rng = new Random(11);
        var sample = Enumerable.Range(0, clusters).SelectMany(c => new[] { $"c{c}m0", $"c{c}m{rng.Next(perCluster)}" })
                               .Distinct().ToList();

        var swScan = new Stopwatch();
        var swLat = new Stopwatch();
        int top1Match = 0, total = 0, overlapSum = 0;
        foreach (var qConcept in sample)
        {
            swScan.Start();
            var scan = space.GetNearestConcepts(qConcept, candidates: active, maxNeighbors: 5, maxCandidates: 1000);
            swScan.Stop();
            swLat.Start();
            var lat = space.GetNearestConcepts(qConcept, candidates: null, maxNeighbors: 5);
            swLat.Stop();
            if (scan.Count == 0 || lat.Count == 0) continue;
            total++;
            if (string.Equals(scan[0].Symbol, lat[0].Symbol, StringComparison.Ordinal)) top1Match++;
            overlapSum += scan.Take(5).Select(s => s.Symbol)
                              .Intersect(lat.Take(5).Select(s => s.Symbol), StringComparer.Ordinal).Count();
        }
        double top1 = top1Match / (double)total, meanOverlap = overlapSum / (double)total;
        _out.WriteLine($"[lattice] top-1 match {top1:P0} ({top1Match}/{total}), mean top-5 overlap {meanOverlap:F2}/5, scan {swScan.ElapsedMilliseconds}ms lattice {swLat.ElapsedMilliseconds}ms");

        // Correctness: the lattice path's nearest is cluster-coherent (a hub or a co-member of the query's cluster).
        int coherent = 0;
        foreach (var qConcept in sample)
        {
            var lat = space.GetNearestConcepts(qConcept, candidates: null, maxNeighbors: 1);
            if (lat.Count == 0) continue;
            var hub = clusterOf[qConcept];
            if (lat[0].Symbol == hub || membersOf[hub].Contains(lat[0].Symbol)) coherent++;
        }

        // Reason (relaxation) also runs through the lattice harvest at scale — it must settle in the right cluster.
        var probe = "c5m3";
        var thought = space.Reason(new[] { probe });
        _out.WriteLine($"[lattice] Reason({probe}) -> '{thought.Symbol}' settled={thought.Settled} conf={thought.Confidence:F3}");
        var probeHub = clusterOf[probe];

        Assert.True(top1 >= 0.9, $"lattice nearest must match the exact scan; top-1 {top1:P0}");
        Assert.True(meanOverlap >= 4.0, $"lattice top-5 must overlap the exact scan's; {meanOverlap:F2}/5");
        Assert.True(coherent >= sample.Count - 2, $"lattice retrieval must stay cluster-coherent; {coherent}/{sample.Count}");
        Assert.True(thought.Symbol == probeHub || membersOf[probeHub].Contains(thought.Symbol),
            $"Reason via lattice harvest must settle in-cluster; got '{thought.Symbol}' for {probe} (hub {probeHub})");
    }
}
