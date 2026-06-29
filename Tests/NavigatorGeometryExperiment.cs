using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// EXPERIMENT — does SETTLING the substrate geometry (Law D: related concepts cluster) make the platonic space more
/// NAVIGABLE for the learned navigator? (PLATONIC_NAVIGATOR.md §7; nova-distributional-large-face.)
///
/// HYPOTHESIS (first principles): the navigator reads MEANING differentials (candFace − goalFace). Meaning lives in
/// the orbital cloud [OrbitalStart=416, dim); the frozen address [0,416) is random spelling/identity (noise for
/// navigation). By Law D a concept's orbital cloud = superposition of its relational context. If that geometry
/// reflects graph structure, a goal-ward gradient EXISTS and the navigator should generalise better.
///
/// The VALUE of this test is the REPORTED NUMBERS, not a bar. It is gated [SlowFact] (RUN_SLOW=1). It asserts only
/// weakly (it ran + sane ranges). Read the test output for the data summary.
/// </summary>
public sealed class NavigatorGeometryExperiment
{
    private readonly ITestOutputHelper _out;
    public NavigatorGeometryExperiment(ITestOutputHelper o) => _out = o;

    private const int Dim = 1024; // production face: orbital [416,1024) = 608 learned meaning dims
    private const int Hidden = 2048;
    private const int K = 16;
    private const int BcEpochs = 30;
    private const int DaggerRounds = 4;
    private const int DaggerEpochs = 35;
    private const double Lr = 1e-3;
    private const int MaxSteps = 16;
    private const int OrbStart = FaceLayout.OrbitalStart; // 416

    private static void Relate(DialecticalSpace s, string a, string b) => s.ObserveContradiction(a, b, 0.0);

    private static IEnumerable<string> Pseudowords(int seed)
    {
        const string cons = "bdfgklmnprstvz";
        const string vow = "aeiou";
        var rng = new Random(seed);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        while (true)
        {
            var syl = rng.Next(2, 4);
            var sb = new StringBuilder();
            for (var s = 0; s < syl; s++) { sb.Append(cons[rng.Next(cons.Length)]); sb.Append(vow[rng.Next(vow.Length)]); }
            var w = sb.ToString();
            if (seen.Add(w)) yield return w;
        }
    }

    private sealed record Curriculum(
        DialecticalSpace Space,
        List<string[]> TrainChains,
        List<(string Category, string[] Entities)> FactStars,
        List<string[]> HoldoutChains,
        List<string> TrainAnswers,
        List<(string Start, string Answer)> TrainStarts,
        List<(string A, string B)> Edges,
        List<string> Concepts);

    // EXACT replica of NavigatorDaggerTests.BuildWorld (same seeds, same construction order → identical graph), plus a
    // captured edge list + concept list for the geometry diagnostic.
    private static Curriculum BuildWorld()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var words = Pseudowords(123).GetEnumerator();
        string NextWord() { words.MoveNext(); return words.Current!; }
        var rng = new Random(7);
        var edges = new List<(string, string)>();
        var concepts = new HashSet<string>(StringComparer.Ordinal);

        List<string[]> MakeChains(int count)
        {
            var chains = new List<string[]>(count);
            for (var i = 0; i < count; i++)
            {
                var len = rng.Next(3, 6);
                var chain = new string[len];
                for (var j = 0; j < len; j++) { chain[j] = NextWord(); concepts.Add(chain[j]); }
                for (var j = 0; j < len - 1; j++) { Relate(space, chain[j], chain[j + 1]); edges.Add((chain[j], chain[j + 1])); }
                chains.Add(chain);
            }
            return chains;
        }

        var trainChains = MakeChains(50);
        var holdoutChains = MakeChains(12);

        var factStars = new List<(string, string[])>();
        for (var c = 0; c < 5; c++)
        {
            var category = NextWord(); concepts.Add(category);
            var entities = new string[6];
            for (var e = 0; e < 6; e++)
            {
                entities[e] = NextWord(); concepts.Add(entities[e]);
                Relate(space, entities[e], category); edges.Add((entities[e], category));
            }
            factStars.Add((category, entities));
        }

        var trainAnswers = trainChains.Select(ch => ch[^1]).Concat(factStars.Select(f => f.Item1)).ToList();
        var trainStarts = new List<(string, string)>();
        foreach (var ch in trainChains)
            for (var j = 0; j < ch.Length - 1; j++) trainStarts.Add((ch[j], ch[^1]));
        foreach (var (cat, ents) in factStars)
            foreach (var e in ents) trainStarts.Add((e, cat));

        return new Curriculum(space, trainChains, factStars, holdoutChains, trainAnswers, trainStarts,
            edges, concepts.ToList());
    }

    private static IEnumerable<string[]> StarChains(IEnumerable<(string Category, string[] Entities)> stars)
        => stars.SelectMany(s => s.Entities.Select(e => new[] { e, s.Category }));

    // The orbital region [416,dim) of a concept's assembled face (unit-normalised by NormalizeSemantic).
    private static double[]? Orbital(DialecticalSpace space, string c)
    {
        if (!space.TryGetConceptFace(c, out var face) || face.Length < Dim) return null;
        var o = new double[Dim - OrbStart];
        for (var i = 0; i < o.Length; i++) o[i] = face[OrbStart + i];
        return o;
    }

    private static double Euclid(double[] a, double[] b)
    {
        var s = 0.0;
        for (var i = 0; i < a.Length; i++) { var d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s);
    }

    private static double Cosine(double[] a, double[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    private double FullSetReachedPct(NavigatorPolicyNet net, Curriculum w, Device device)
    {
        var (r1, t1) = NavigatorDaggerTrainer.CountReached(net, w.Space, w.TrainChains, device, MaxSteps, K);
        var (r2, t2) = NavigatorDaggerTrainer.CountReached(net, w.Space, StarChains(w.FactStars), device, MaxSteps, K);
        return 100.0 * (r1 + r2) / Math.Max(1, t1 + t2);
    }

    private double HeldoutReachedPct(NavigatorPolicyNet net, Curriculum w, Device device, out int r, out int t)
    {
        (r, t) = NavigatorDaggerTrainer.CountReached(net, w.Space, w.HoldoutChains, device, MaxSteps, K);
        return 100.0 * r / Math.Max(1, t);
    }

    // Train BC + DAgger on the (already-settled) space with the current NavFeatures.MeaningFloorStart; report held-out%.
    private (double heldFinal, double trainedFinal, double[] heldByStage) TrainNavigator(Curriculum world, Device device, string tag)
    {
        var space = world.Space;
        using var net = new NavigatorPolicyNet(Dim, Hidden, device);

        var bc = NavigatorDaggerTrainer.BuildOracleTrajectories(space, world.TrainAnswers, K);
        var bcLoss = NavigatorDaggerTrainer.TrainOnTrajectories(net, bc, BcEpochs, Lr, device, K);
        var heldBc = HeldoutReachedPct(net, world, device, out var hr0, out var ht0);
        var trainedBc = FullSetReachedPct(net, world, device);
        _out.WriteLine($"  [{tag}] BC-only: CE={bcLoss.CrossEntropy:F3} | trained={trainedBc:F1}% | HELD-OUT={heldBc:F1}% ({hr0}/{ht0})");

        var aggregate = new List<NavTrajectory>(bc);
        var heldByStage = new List<double> { heldBc };
        double trainedFinal = trainedBc, heldFinal = heldBc;
        for (var round = 1; round <= DaggerRounds; round++)
        {
            var rollouts = NavigatorDaggerTrainer.RolloutDaggerTrajectories(net, space, world.TrainStarts, device, MaxSteps, K);
            aggregate.AddRange(rollouts);
            var loss = NavigatorDaggerTrainer.TrainOnTrajectories(net, aggregate, DaggerEpochs, Lr, device, K);
            trainedFinal = FullSetReachedPct(net, world, device);
            heldFinal = HeldoutReachedPct(net, world, device, out var hr, out var ht);
            heldByStage.Add(heldFinal);
            _out.WriteLine($"  [{tag}] DAgger {round}: CE={loss.CrossEntropy:F3} | trained={trainedFinal:F1}% | HELD-OUT={heldFinal:F1}% ({hr}/{ht})");
        }
        return (heldFinal, trainedFinal, heldByStage.ToArray());
    }

    [SlowFact]
    public void SettlingGeometry_NavigabilityExperiment()
    {
        var device = cuda.is_available() ? CUDA : CPU;
        manual_seed(7);
        var world = BuildWorld();
        var space = world.Space;
        _out.WriteLine($"=== device={device.type} | {world.Concepts.Count} concepts | {world.Edges.Count} edges | " +
                       $"{world.TrainChains.Count} train + {world.HoldoutChains.Count} held-out chains + {world.FactStars.Count} stars ===");

        // ──────────────────────────────────────────────────────────────────────── STEP 1: SETTLE THE CLOUDS.
        // The substrate recomputes each concept's orbital cloud INLINE on every ObserveContradiction (DialecticalSpace
        // .RecomputeCloud): cloud = token(self) + Σ_neighbour (1−2κ)·token(neighbour). It is a FIXED 1-HOP TOKEN SUM —
        // deterministic from the current adjacency set. So the geometry is ALREADY settled after planting; there is no
        // separate multi-pass relaxation. We demonstrate idempotency: snapshot orbitals, re-observe every edge (a second
        // "settle pass"), and measure the change. ≈0 ⇒ multi-pass settling is a NO-OP (the cloud cannot diffuse).
        var before = world.Concepts.ToDictionary(c => c, c => Orbital(space, c), StringComparer.Ordinal);
        foreach (var (a, b) in world.Edges) Relate(space, a, b); // a second full observation pass over the graph
        double maxDelta = 0, sumDelta = 0; var n = 0;
        foreach (var c in world.Concepts)
        {
            var ob = before[c]; var oa = Orbital(space, c);
            if (ob is null || oa is null) continue;
            var d = Euclid(ob, oa); maxDelta = Math.Max(maxDelta, d); sumDelta += d; n++;
        }
        _out.WriteLine($"=== STEP 1 SETTLE: re-observed all {world.Edges.Count} edges (2nd pass) | orbital Δ mean={sumDelta / Math.Max(1, n):E3} " +
                       $"max={maxDelta:E3}  ⇒ {(maxDelta < 1e-9 ? "NO-OP (cloud is a fixed 1-hop token sum; cannot diffuse multi-hop)" : "changed")} ===");

        // ──────────────────────────────────────────────────────────────────────── STEP 2: DOES GEOMETRY REFLECT STRUCTURE?
        // Build undirected adjacency + BFS hop-distance over the planted graph (a forest of disjoint chains + stars).
        var adj = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        void Link(string a, string b)
        {
            if (!adj.TryGetValue(a, out var sa)) { sa = new HashSet<string>(StringComparer.Ordinal); adj[a] = sa; }
            if (!adj.TryGetValue(b, out var sb)) { sb = new HashSet<string>(StringComparer.Ordinal); adj[b] = sb; }
            sa.Add(b); sb.Add(a);
        }
        foreach (var (a, b) in world.Edges) Link(a, b);

        Dictionary<string, int> Bfs(string src)
        {
            var dist = new Dictionary<string, int>(StringComparer.Ordinal) { [src] = 0 };
            var q = new Queue<string>(); q.Enqueue(src);
            while (q.Count > 0)
            {
                var u = q.Dequeue();
                if (!adj.TryGetValue(u, out var ns)) continue;
                foreach (var v in ns) if (!dist.ContainsKey(v)) { dist[v] = dist[u] + 1; q.Enqueue(v); }
            }
            return dist;
        }

        var orb = world.Concepts.ToDictionary(c => c, c => Orbital(space, c), StringComparer.Ordinal);
        var byHopEuclid = new Dictionary<int, List<double>>();
        var byHopCos = new Dictionary<int, List<double>>();
        var disconnEuclid = new List<double>();
        var disconnCos = new List<double>();
        // accumulate (hop, euclid) pairs for a Pearson correlation over CONNECTED pairs
        var corrHop = new List<double>(); var corrDist = new List<double>();

        var nodes = world.Concepts.Where(c => orb[c] != null).ToList();
        for (var i = 0; i < nodes.Count; i++)
        {
            var di = Bfs(nodes[i]);
            for (var j = i + 1; j < nodes.Count; j++)
            {
                var a = orb[nodes[i]]!; var b = orb[nodes[j]]!;
                var e = Euclid(a, b); var cs = Cosine(a, b);
                if (di.TryGetValue(nodes[j], out var hop))
                {
                    if (!byHopEuclid.TryGetValue(hop, out var le)) { le = new List<double>(); byHopEuclid[hop] = le; byHopCos[hop] = new List<double>(); }
                    le.Add(e); byHopCos[hop].Add(cs);
                    corrHop.Add(hop); corrDist.Add(e);
                }
                else { disconnEuclid.Add(e); disconnCos.Add(cs); }
            }
        }

        _out.WriteLine($"=== STEP 2 GEOMETRY vs GRAPH STRUCTURE (orbital region [{OrbStart},{Dim}), unit-norm) ===");
        foreach (var hop in byHopEuclid.Keys.OrderBy(h => h))
            _out.WriteLine($"   hop {hop}: n={byHopEuclid[hop].Count,6} | mean orbital-dist={byHopEuclid[hop].Average():F3} | mean cos={byHopCos[hop].Average():F3}");
        _out.WriteLine($"   disconnected (different component): n={disconnEuclid.Count,6} | mean orbital-dist={(disconnEuclid.Count > 0 ? disconnEuclid.Average() : double.NaN):F3} | mean cos={(disconnCos.Count > 0 ? disconnCos.Average() : double.NaN):F3}");

        var adjEuclid = byHopEuclid.TryGetValue(1, out var h1) ? h1 : new List<double>();
        var nonAdjEuclid = byHopEuclid.Where(kv => kv.Key >= 2).SelectMany(kv => kv.Value).Concat(disconnEuclid).ToList();
        var adjMean = adjEuclid.Count > 0 ? adjEuclid.Average() : double.NaN;
        var nonAdjMean = nonAdjEuclid.Count > 0 ? nonAdjEuclid.Average() : double.NaN;
        _out.WriteLine($"   ADJACENT(hop1) mean orbital-dist={adjMean:F3}  vs  NON-ADJACENT(hop≥2 + disconnected) mean={nonAdjMean:F3}  " +
                       $"(separation={nonAdjMean - adjMean:F3}, ratio={nonAdjMean / adjMean:F2}×)");
        _out.WriteLine($"   PEARSON corr(hop-distance, orbital-distance) over connected pairs = {Pearson(corrHop, corrDist):F3}  " +
                       $"(positive ⇒ farther-in-graph is farther-in-geometry)");

        // ──────────────────────────────────────────────────────────────────────── STEP 3 & 4: NAVIGATOR ON SETTLED SPACE.
        // Step 3 = full-face differential features (reproduces the prior ~64% baseline; the space is already settled).
        // Step 4 = orbital-only scoring features (frozen spelling excluded from the per-candidate differential).
        _out.WriteLine("=== STEP 3: NAVIGATOR on settled space, FULL-FACE differential features ===");
        NavFeatures.MeaningFloorStart = 0;
        manual_seed(7);
        var full = TrainNavigator(world, device, "full");

        _out.WriteLine($"=== STEP 4: NAVIGATOR on settled space, ORBITAL-ONLY scoring features [{OrbStart},{Dim}) ===");
        double heldOrbital, trainedOrbital; double[] orbStages;
        try
        {
            NavFeatures.MeaningFloorStart = OrbStart;
            manual_seed(7);
            var orbResult = TrainNavigator(world, device, "orbital");
            heldOrbital = orbResult.heldFinal; trainedOrbital = orbResult.trainedFinal; orbStages = orbResult.heldByStage;
        }
        finally { NavFeatures.MeaningFloorStart = 0; }

        // ──────────────────────────────────────────────────────────────────────── DATA SUMMARY.
        _out.WriteLine("=== DATA SUMMARY ===");
        _out.WriteLine($"  SETTLE mechanism: inline 1-hop token superposition (RecomputeCloud on ObserveContradiction); " +
                       $"2nd settle pass orbital Δmax={maxDelta:E2} ⇒ {(maxDelta < 1e-9 ? "multi-pass is a NO-OP" : "non-idempotent")}");
        _out.WriteLine($"  GEOMETRY: adjacent={adjMean:F3} vs non-adjacent={nonAdjMean:F3} ({nonAdjMean / adjMean:F2}×); " +
                       $"hop↔dist Pearson={Pearson(corrHop, corrDist):F3}");
        _out.WriteLine($"  HELD-OUT reached%:  prior-baseline≈64%  |  settled FULL-face={full.heldFinal:F1}%  |  settled ORBITAL-only={heldOrbital:F1}%");
        _out.WriteLine($"  full held-out by stage:    {string.Join(" ", full.heldByStage.Select(x => $"{x:F0}"))}");
        _out.WriteLine($"  orbital held-out by stage: {string.Join(" ", orbStages.Select(x => $"{x:F0}"))}");
        _out.WriteLine($"  trained-graph final: full={full.trainedFinal:F1}% orbital={trainedOrbital:F1}%");

        // Weak asserts: it ran end-to-end and produced sane numbers.
        Assert.True(adjEuclid.Count > 0 && nonAdjEuclid.Count > 0, "geometry diagnostic produced no pairs");
        Assert.True(full.heldFinal >= 0 && heldOrbital >= 0, "navigator runs produced no held-out number");
    }

    private static double Pearson(List<double> x, List<double> y)
    {
        var n = x.Count;
        if (n == 0) return double.NaN;
        double mx = x.Average(), my = y.Average(), sxy = 0, sxx = 0, syy = 0;
        for (var i = 0; i < n; i++) { var dx = x[i] - mx; var dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
        return sxy / (Math.Sqrt(sxx * syy) + 1e-12);
    }
}
