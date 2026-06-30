using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// STRESS TEST — does #54's per-level goal-REGION multi-hop conditioning still lift held-out multi-hop LANDING on a
/// NOISY / messy taxonomy, or does the depth-pooled region get fuzzy and SOFTEN the lift? This is a MEASUREMENT test:
/// the honest goal is to FIND WHERE IT BREAKS, not to manufacture a 100%.
///
/// #54 broke the multi-hop ceiling (0%->100%) on a CLEAN uniform-depth taxonomy (2 domains, 1 root, every member depth 3
/// — see <see cref="NavigatorMultiHopCeiling"/>). The mechanism: a LEARNED per-level goal-REGION = centroid of the
/// concepts at each graph DEPTH (genus=chain[0], domain=chain[1], root=chain[^1], climbing each leaf's is-a chain). The
/// suspected weakness on a messy graph is that the depth-pooled regions blur:
///   (a) UNEVEN DEPTHS — "domain" = chain[1] is the ROOT for a shallow (depth-2) leaf but a mid-graph node for a deep
///       (depth-4) leaf, so the domain-region centroid pools heterogeneous concepts (domains AND roots together);
///   (b) MULTIPLE ROOTS — the root-region centroid averages disjoint, far-apart tops;
///   (c) OVERLAPPING members — a member under two genera gives an ambiguous climb chain.
///
/// This planted taxonomy (see <see cref="NoisyTaxonomy"/>) has ALL THREE adversarial properties plus asymmetric
/// branching. We measure, on a HELD-OUT set the policy never trains on, per-cue (genus / domain / root) LANDING accuracy
/// for the NAVIGATOR cold->warm, AND two BASELINES on the SAME messy structure (the untrained-navigator floor + a
/// one-shot nearest-to-region lookup that does NO walk) — so the reported number is the LIFT, not just the absolute.
///
/// HONEST: the SMOKE budget (NAVNOISE_CYCLES default 2, &lt;30s) is NOT the verdict — it only confirms the path runs and
/// gives noisy early numbers. The real measurement is NAVNOISE_CYCLES&gt;=5 (run in the background). The MEASUREMENT is
/// identical at any budget; only the strict assertion is gated so the default smoke never blocks/flakes.
/// </summary>
public sealed class NavigatorNoisyGraphStress
{
    private readonly ITestOutputHelper _out;
    public NavigatorNoisyGraphStress(ITestOutputHelper o) => _out = o;

    // BOUNDED BY DEFAULT (smoke = 2 cycles). The full measurement is NAVNOISE_CYCLES=8 RUN_SLOW=1; the strong claim only
    // fires at >=5. The per-cue held-out walk is the SAME at any budget — only the assertion bar scales.
    private static int Cycles =>
        int.TryParse(Environment.GetEnvironmentVariable("NAVNOISE_CYCLES"), out var n) && n >= 1 ? n : 2;
    private static int Epochs =>
        int.TryParse(Environment.GetEnvironmentVariable("NAVNOISE_EPOCHS"), out var e) && e >= 1 ? e : 4;
    private static bool StrongClaim => Cycles >= 5;

    private static GenesisNovaConfig BuildConfig(string dir) =>
        new GenesisNovaConfig(
            HiddenSize: 64, Backend: ComputeBackend.Cpu, AutoPersist: false, AutoResume: false, LocalStateDirectory: dir)
        .WithProductionMechanisms() with { FieldTicks = false, MeaningOps = false };

    [SlowFact]
    public async Task HeldOutMultiHopLanding_OnNoisyGraph_Measured()
    {
        await Task.Yield();
        var dir = Path.Combine(Path.GetTempPath(), "gn-navnoise-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(BuildConfig(dir));
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);

            // ── PLANT the noisy taxonomy (adjacency-only, auto-padded for a monotone degree climb) + register held-out ──
            var (edges, fillers, heldOut) = NoisyTaxonomy.Build();
            runtime.PlantNavigatorTaxonomy(edges, reinforce: 5);
            runtime.RegisterNavigatorHeldOut(heldOut.SelectMany(h => new[]
            {
                (h.Member, NavCue.Genus,  h.Genus),
                (h.Member, NavCue.Domain, h.Domain),
                (h.Member, NavCue.Root,   h.Root),
            }));

            Assert.True(runtime.NavHeldOutCount > 0, "no held-out queries registered");
            foreach (var h in heldOut)
                Assert.True(ds.ContainsConcept(h.Member), $"held-out member '{h.Member}' was not planted");

            // The LEARNED depth-pooled regions exist (derived from the messy graph's depth — no hardcoding).
            var gRegion = runtime.NavLevelRegion(NavCue.Genus);
            var dRegion = runtime.NavLevelRegion(NavCue.Domain);
            var rRegion = runtime.NavLevelRegion(NavCue.Root);

            _out.WriteLine($"=== NOISY-GRAPH STRESS: {ds.NodeCount} concepts ({edges.Count} edges, {fillers.Count} fillers) | " +
                           $"navigator {runtime.State.Navigator.Dim}/{runtime.State.Navigator.Hidden} | {runtime.NavHeldOutCount} held-out queries | " +
                           $"{Cycles} cycles x {Epochs} epochs (NAVNOISE_CYCLES/NAVNOISE_EPOCHS) | strongClaim={StrongClaim} ===");
            _out.WriteLine("    adversarial properties: UNEVEN DEPTHS (chains len 2/3/4) | MULTIPLE ROOTS (entity, matter, notion) | " +
                           "OVERLAP (coral -> mineral AND reptile) | ASYMMETRIC branching (mammal=4 vs insect=1)");
            _out.WriteLine($"    learned regions present: genus={gRegion is not null} domain={dRegion is not null} root={rRegion is not null}");
            Assert.NotNull(dRegion);
            Assert.NotNull(rRegion);

            // What does the depth-pooled DOMAIN region's nearest concept look like? (diagnostic for the smear.)
            _out.WriteLine($"    domain-region nearest concept = '{NearestConcept(ds, dRegion!, exclude: null)}'  " +
                           $"root-region nearest = '{NearestConcept(ds, rRegion!, exclude: null)}'");

            // ── ONE-SHOT BASELINE (no walk): for each held-out (member,cue), answer = the concept globally nearest the
            //    cue's learned region centroid. Isolates region QUALITY — if the smear were a clean prototype this would
            //    already land the ancestor; the navigator's job is to beat it by WALKING the graph to the right one. ──
            var baseRegion = new Dictionary<NavCue, (int correct, int n)>();
            foreach (var cue in Cues) baseRegion[cue] = (0, 0);
            foreach (var h in heldOut)
            {
                foreach (var (cue, ancestor) in new[] { (NavCue.Genus, h.Genus), (NavCue.Domain, h.Domain), (NavCue.Root, h.Root) })
                {
                    var region = runtime.NavLevelRegion(cue);
                    if (region is null) continue;
                    var ans = NearestConcept(ds, region, exclude: h.Member);
                    var (c, n) = baseRegion[cue];
                    baseRegion[cue] = (c + (string.Equals(ans, ancestor, StringComparison.Ordinal) ? 1 : 0), n + 1);
                }
            }

            // ── THE PER-CUE COLD->WARM NAVIGATOR CURVE on the held-out set ──
            var series = new Dictionary<NavCue, List<double>> { [NavCue.Genus] = new(), [NavCue.Domain] = new(), [NavCue.Root] = new() };
            void Checkpoint(int c)
            {
                var pts = runtime.EvaluateNavigatorHeldOutPerCue();
                foreach (var cue in Cues)
                {
                    var p = pts.FirstOrDefault(x => x.Cycle == (int)cue);
                    series[cue].Add(p.Count > 0 ? p.AccuracyPct : 0.0);
                }
                var g = pts.FirstOrDefault(x => x.Cycle == (int)NavCue.Genus);
                var d = pts.FirstOrDefault(x => x.Cycle == (int)NavCue.Domain);
                var r = pts.FirstOrDefault(x => x.Cycle == (int)NavCue.Root);
                _out.WriteLine($"    [nav held-out] cyc {c,3} | genus {g.AccuracyPct,6:P0} (res {g.ResolvePct,4:P0}) | " +
                               $"domain {d.AccuracyPct,6:P0} (res {d.ResolvePct,4:P0}) | root {r.AccuracyPct,6:P0} (res {r.ResolvePct,4:P0})");
            }

            Checkpoint(0); // COLD (untrained policy) — the navigator-floor baseline
            for (var c = 1; c <= Cycles; c++)
            {
                runtime.TrainNavigatorCycle(maxMembers: 64, epochs: Epochs);
                Checkpoint(c);
            }

            double Cold(NavCue cue) => series[cue].First();
            double Warm(NavCue cue) => series[cue].Skip(Math.Max(1, series[cue].Count - 3)).Average();
            double Peak(NavCue cue) => series[cue].Max();

            _out.WriteLine("=== SUMMARY (HONEST — smoke numbers are noisy; verdict needs NAVNOISE_CYCLES>=5) ===");
            foreach (var cue in Cues)
            {
                var (bc, bn) = baseRegion[cue];
                var basePct = bn > 0 ? (double)bc / bn : 0.0;
                _out.WriteLine($"    {cue,-7} | nav: {string.Join(" ", series[cue].Select(v => $"{v:P0}"))}  " +
                               $"| cold {Cold(cue):P0} -> warm {Warm(cue):P0} (peak {Peak(cue):P0})  " +
                               $"|| BASELINE one-shot nearest-region {basePct:P0}");
            }

            var multiHopWarm = (Warm(NavCue.Domain) + Warm(NavCue.Root)) / 2.0;
            var multiHopPeak = Math.Max(Peak(NavCue.Domain), Peak(NavCue.Root));
            var multiHopCold = (Cold(NavCue.Domain) + Cold(NavCue.Root)) / 2.0;
            var baseMulti = (Frac(baseRegion[NavCue.Domain]) + Frac(baseRegion[NavCue.Root])) / 2.0;
            _out.WriteLine($"    MULTI-HOP (domain+root) nav: cold {multiHopCold:P1} -> warm {multiHopWarm:P1} (peak {multiHopPeak:P1}) " +
                           $"|| one-shot-region baseline {baseMulti:P1}");
            // Verdict PER CUE — an aggregate would let one strong cue mask another's collapse (the honest failure mode is
            // expected to be cue-SPECIFIC: the DOMAIN region = chain[1] is the one that pools heterogeneous depths).
            string Verdict(NavCue cue) => Peak(cue) >= 0.40 ? "HELD" : Peak(cue) >= 0.15 ? "SOFTENED" : "BROKE DOWN";
            _out.WriteLine($"    VERDICT per cue (vs clean 100%): genus={Verdict(NavCue.Genus)} domain={Verdict(NavCue.Domain)} root={Verdict(NavCue.Root)} " +
                           $"(genus {Peak(NavCue.Genus):P0} | domain {Peak(NavCue.Domain):P0} | root {Peak(NavCue.Root):P0})" +
                           $"{(StrongClaim ? "" : "  [SMOKE budget — NOT the verdict; run NAVNOISE_CYCLES=8]")}");

            // ── ASSERTIONS — only cheap always-true invariants fire at the smoke budget so it never blocks/flakes. ──
            Assert.True(series[NavCue.Domain].Count == Cycles + 1, "the per-cue held-out series was not populated each cycle");
            // The strong measurement claim (gated, NAVNOISE_CYCLES>=5): does the lift SURVIVE the noisy graph at all, and
            // does the WALK beat the one-shot region lookup? This is a MEASUREMENT, so the bar is deliberately weak (any
            // generalization above the cold floor AND any edge over the static baseline) — we are documenting, not gating.
            if (StrongClaim)
            {
                Assert.True(multiHopPeak > multiHopCold,
                    $"multi-hop landing showed NO lift over the cold floor on the noisy graph (cold {multiHopCold:P1} -> peak {multiHopPeak:P1}) — region conditioning fully collapsed");
            }
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    private static readonly NavCue[] Cues = { NavCue.Genus, NavCue.Domain, NavCue.Root };
    private static double Frac((int correct, int n) x) => x.n > 0 ? (double)x.correct / x.n : 0.0;

    /// <summary>The active concept whose face is most cosine-aligned with <paramref name="region"/> (a one-shot lookup,
    /// no walk), skipping numbers and the excluded member. The baseline's "answer" to a region — what a static
    /// nearest-to-prototype would return.</summary>
    private static string NearestConcept(DialecticalSpace ds, IReadOnlyList<double> region, string? exclude)
    {
        string best = string.Empty; double bestSim = double.NegativeInfinity;
        double rn = 0; for (var i = 0; i < region.Count; i++) rn += region[i] * region[i];
        if (rn <= 1e-12) return best;
        foreach (var c in ds.ActiveConcepts)
        {
            if (string.IsNullOrWhiteSpace(c) || double.TryParse(c, out _)) continue;
            if (exclude is not null && string.Equals(c, exclude, StringComparison.Ordinal)) continue;
            if (!ds.TryGetConceptFace(c, out var f) || f.Length == 0) continue;
            double dot = 0, fn = 0; var n = Math.Min(f.Length, region.Count);
            for (var i = 0; i < n; i++) { dot += f[i] * region[i]; fn += f[i] * f[i]; }
            if (fn <= 1e-12) continue;
            var sim = dot / Math.Sqrt(fn);
            if (sim > bestSim) { bestSim = sim; best = c; }
        }
        return best;
    }
}

/// <summary>
/// THE NOISY TAXONOMY (DATA, not a dispatch table). Three DISJOINT roots; chains of length 2, 3 AND 4 (uneven depth);
/// one OVERLAPPING member; asymmetric branching. Real is-a adjacency edges (child -> immediate parent) are listed
/// explicitly; the structure is then AUTO-PADDED with filler leaves so each ancestor's relational degree strictly
/// exceeds its children's — keeping <c>ClimbAncestors</c>' degree-gradient monotone so the INTENDED chains are the
/// ground truth (the messiness then lives in the depth-POOLED region, which is the variable under test, not in a broken
/// climb). Numbers never appear (substrate hard rule). Held-out members span all three depths + the overlap.
/// </summary>
internal static class NoisyTaxonomy
{
    // ── REAL is-a adjacency (child -> immediate parent). Roots (entity/matter/notion) have no parent. ──
    private static readonly (string Child, string Parent)[] Real =
    {
        // ── TREE A: root "entity" — a NORMAL depth-3 branch (creature) + a DEEP depth-4 branch (artifact) ──
        ("creature", "entity"),
        ("mammal", "creature"), ("bird", "creature"), ("reptile", "creature"), ("insect", "creature"), ("fish", "creature"),
        ("dog", "mammal"), ("cat", "mammal"), ("wolf", "mammal"), ("fox", "mammal"),       // wolf = HELD-OUT (depth 3)
        ("robin", "bird"), ("sparrow", "bird"), ("eagle", "bird"), ("finch", "bird"),       // eagle = HELD-OUT (depth 3)
        ("lizard", "reptile"), ("snake", "reptile"),                                        // asymmetric: 2 members
        ("ant", "insect"),                                                                  // asymmetric: 1 member
        ("perch", "fish"),                                                                  // asymmetric: 1 member
        ("artifact", "entity"),
        ("machine", "artifact"),                                                            // the EXTRA level => depth 4
        ("engine", "machine"), ("tool", "machine"),
        ("piston", "engine"), ("valve", "engine"), ("rotor", "engine"), ("shaft", "engine"),// rotor = HELD-OUT (depth 4)
        ("hammer", "tool"), ("wrench", "tool"),

        // ── TREE B: root "matter" — SHALLOW depth-2 (genera sit directly under the root) ──
        ("mineral", "matter"), ("liquid", "matter"),
        ("iron", "mineral"), ("gold", "mineral"), ("copper", "mineral"), ("tin", "mineral"),// copper = HELD-OUT (depth 2)
        ("brine", "liquid"), ("milk", "liquid"), ("juice", "liquid"),

        // ── TREE C: root "notion" — SHALLOW depth-2, small ──
        ("value", "notion"), ("fact", "notion"),
        ("good", "value"), ("bad", "value"),
        ("truth", "fact"), ("lie", "fact"), ("maybe", "fact"),                              // maybe = HELD-OUT (depth 2)

        // ── OVERLAP: "coral" belongs to TWO genera in TWO DIFFERENT trees (mineral@matter AND reptile@entity) ──
        ("coral", "mineral"), ("coral", "reptile"),                                         // coral = HELD-OUT (overlap)
    };

    // Held-out members + their INTENDED (genus, domain, root) by the planted structure. For a depth-2 member the 2-hop
    // ancestor IS the root, so domain == root (the heterogeneity the pooled domain-region must contend with).
    private static readonly (string Member, string Genus, string Domain, string Root)[] HeldOutSet =
    {
        ("wolf",   "mammal",  "creature", "entity"),   // depth 3
        ("eagle",  "bird",    "creature", "entity"),   // depth 3
        ("rotor",  "engine",  "machine",  "entity"),   // depth 4 (domain is mid-graph, not the root)
        ("copper", "mineral", "matter",   "matter"),   // depth 2 (domain == root)
        ("maybe",  "fact",    "notion",   "notion"),   // depth 2 (domain == root)
        ("coral",  "mineral", "matter",   "matter"),   // overlap (climb resolves to the higher-degree parent mineral)
    };

    // Real-ish filler words used to auto-pad internal nodes' degree (kept distinct from every real concept).
    private static readonly string[] FillerPool =
    {
        "rock", "cloud", "fire", "dust", "glass", "stone", "sand", "mud", "ash", "clay",
        "wax", "ice", "steam", "foam", "myth", "rule", "law", "sign", "gravel", "pebble",
        "vapor", "smoke", "resin", "chalk", "slate", "amber", "coal", "peat", "loam", "silt",
        "grit", "husk", "bark", "moss", "reed", "fern", "soot", "lime",
    };

    public static IReadOnlyList<(string Member, string Genus, string Domain, string Root)> HeldOut() => HeldOutSet;

    /// <summary>Real edges + auto-pad fillers, the filler list, and the held-out set. Auto-padding makes each ancestor's
    /// degree strictly exceed its children's (bottom-up), so the degree-gradient climb is monotone and the intended
    /// chains are recoverable — the noise lives in the depth-pooled REGION (uneven depth / multiple roots / overlap),
    /// which is exactly the variable under test.</summary>
    public static (List<(string Child, string Parent)> Edges, List<string> Fillers,
                   IReadOnlyList<(string Member, string Genus, string Domain, string Root)> HeldOut) Build()
    {
        var children = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var parents = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var (c, p) in Real)
        {
            (children.TryGetValue(p, out var ks) ? ks : children[p] = new()).Add(c);
            (parents.TryGetValue(c, out var ps) ? ps : parents[c] = new()).Add(p);
        }

        // Padded degree, memoized bottom-up: max(realDegree, max child's padded degree + 1).
        var memo = new Dictionary<string, int>(StringComparer.Ordinal);
        int RealDeg(string n) => (children.TryGetValue(n, out var ks) ? ks.Count : 0) + (parents.TryGetValue(n, out var ps) ? ps.Count : 0);
        int Padded(string n)
        {
            if (memo.TryGetValue(n, out var v)) return v;
            var req = 0;
            if (children.TryGetValue(n, out var ks)) foreach (var k in ks) req = Math.Max(req, Padded(k) + 1);
            return memo[n] = Math.Max(RealDeg(n), req);
        }

        var edges = new List<(string, string)>(Real);
        var fillers = new List<string>();
        var fi = 0;
        foreach (var node in children.Keys.OrderBy(x => x, StringComparer.Ordinal)) // internal nodes only
        {
            var need = Padded(node) - RealDeg(node);
            for (var i = 0; i < need; i++)
            {
                var f = fi < FillerPool.Length ? FillerPool[fi] : $"pad{fi}";
                fi++;
                edges.Add((f, node));
                fillers.Add(f);
            }
        }
        return (edges, fillers, HeldOutSet);
    }
}
