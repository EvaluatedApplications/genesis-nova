using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// GOAL-EMERGENT, QUERY-CONDITIONED NAVIGATION — the axiomatic route into inference (PLATONIC_NAVIGATOR.md §2/§3/§6/§7;
/// PLATONIC_MIND.md §3 "an answer is wherever a query relaxes to"). The sibling <see cref="NavigatorRealDataValidation"/>
/// proves the navigator when the ANSWER coordinate is supplied (goal-conditioned). This test removes the answer: the
/// walker is given ONLY the query-context (anchorFace, cue) — cue ∈ {GENUS, DOMAIN, ROOT} — and must RELAX to the
/// cued ancestor without ever being told where it is. A query is a partial state; the answer is the emergent attractor.
///
/// THE HEADLINE: the SAME anchor with three different cues must land on three DIFFERENT correct answers, none supplied.
/// Reuses the real taxonomy of <see cref="NavigatorRealDataValidation"/> (organism → {animal,plant} → genera → members;
/// mammal/insect hubs; held-out amphibian/shrub genera + salmon leaves added post-training). Production sizing
/// (Dim 1024, Hidden 2048); SMALL epochs/eval sets to run in a few minutes on CUDA. Gated [SlowFact] behind RUN_SLOW=1.
/// HONEST MEASUREMENT — loose sanity floors only; the reported DATA is the result.
/// </summary>
public sealed class NavigatorQueryConditioned
{
    private readonly ITestOutputHelper _out;
    public NavigatorQueryConditioned(ITestOutputHelper o) => _out = o;

    private const int Dim = 1024;       // production FaceDimension
    private const int Hidden = 2048;    // production HiddenSize
    private const int K = 16;           // egocentric candidate fan-out
    private const int BcEpochs = 20;    // FAST
    private const int DaggerRounds = 1; // FAST
    private const int DaggerEpochs = 20;// FAST
    private const double Lr = 1e-3;
    private const int MaxSteps = 12;
    private const int ObservePerEdge = 4;

    private const string Root = "organism";
    private static readonly int CueGenus = (int)NavCue.Genus;
    private static readonly int CueDomain = (int)NavCue.Domain;
    private static readonly int CueRoot = (int)NavCue.Root;

    private static void Plant(DialecticalSpace s, Dictionary<string, string> parent, string child, string par)
    {
        parent[child] = par;
        for (var i = 0; i < ObservePerEdge; i++) s.ObserveContradiction(child, par, 0.0);
    }

    // ── The real taxonomy (copied from NavigatorRealDataValidation). ──
    private static readonly (string Domain, (string Genus, string[] Members)[] Genera)[] Taxonomy =
    {
        ("animal", new[]
        {
            ("mammal", new[] { "dog","cat","horse","cow","lion","tiger","wolf","bear","deer","fox","rabbit","sheep",
                               "goat","pig","mouse","elephant","giraffe","zebra","monkey","kangaroo","bat","whale",
                               "camel","donkey","hippo","rhino" }),
            ("bird", new[] { "robin","sparrow","eagle","owl","hawk","crow","finch","duck","goose","swan","pigeon",
                             "parrot","penguin","ostrich" }),
            ("fish", new[] { "salmon","trout","tuna","cod","bass","shark","herring","mackerel" }),
            ("reptile", new[] { "snake","lizard","turtle","crocodile","gecko","iguana" }),
            ("insect", new[] { "ant","bee","wasp","beetle","moth","fly","grasshopper","butterfly","cricket","dragonfly",
                               "ladybug","termite","aphid","flea","mosquito","weevil","earwig" }),
            ("arachnid", new[] { "spider","scorpion","tick","mite","harvestman" }),
            ("mollusk", new[] { "snail","clam","octopus","squid","oyster" }),
        }),
        ("plant", new[]
        {
            ("tree", new[] { "oak","pine","maple","birch","willow","elm","cedar","spruce","ash","poplar","fir","beech" }),
            ("flower", new[] { "rose","tulip","daisy","lily","orchid","sunflower","violet","daffodil","poppy" }),
            ("grass", new[] { "wheat","corn","rice","barley","oat","rye" }),
            ("vegetable", new[] { "carrot","potato","onion","tomato","pea","bean","cabbage","lettuce" }),
            ("berry", new[] { "strawberry","blueberry","raspberry","blackberry","gooseberry","cranberry" }),
        }),
    };

    private static readonly (string Genus, string Domain, string[] Members)[] HeldGenera =
    {
        ("amphibian", "animal", new[] { "frog","toad","newt","salamander","axolotl" }),
        ("shrub", "plant", new[] { "holly","hazel","privet","boxwood" }),
    };
    private static readonly string[] HeldSalmonLeaves = { "chinook","sockeye","coho" };

    private sealed record World(
        DialecticalSpace Space,
        Dictionary<string, string> Parent,   // child → immediate parent (trained only)
        HashSet<string> Genera,
        HashSet<string> Domains,
        List<string> Members);               // leaf members directly under a genus

    private static World BuildTrainedWorld()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var parent = new Dictionary<string, string>(StringComparer.Ordinal);
        var genera = new HashSet<string>(StringComparer.Ordinal);
        var domains = new HashSet<string>(StringComparer.Ordinal);
        var members = new List<string>();

        foreach (var (domain, generaArr) in Taxonomy)
        {
            domains.Add(domain);
            Plant(space, parent, domain, Root);
            foreach (var (genus, mem) in generaArr)
            {
                genera.Add(genus);
                Plant(space, parent, genus, domain);
                foreach (var m in mem) { Plant(space, parent, m, genus); members.Add(m); }
            }
        }
        space.FlushCloudBatch();
        return new World(space, parent, genera, domains, members);
    }

    private static (List<string> Held, Dictionary<string, string> HeldParent) AddHeldout(DialecticalSpace space)
    {
        var held = new List<string>();
        var heldParent = new Dictionary<string, string>(StringComparer.Ordinal);
        void PlantHeld(string child, string par)
        {
            heldParent[child] = par; held.Add(child);
            for (var i = 0; i < ObservePerEdge; i++) space.ObserveContradiction(child, par, 0.0);
        }
        foreach (var (genus, domain, members) in HeldGenera)
        {
            PlantHeld(genus, domain);
            foreach (var m in members) PlantHeld(m, genus);
        }
        foreach (var leaf in HeldSalmonLeaves) PlantHeld(leaf, "salmon");
        space.FlushCloudBatch();
        return (held, heldParent);
    }

    // Cued-ancestor resolvers over a (possibly merged trained+held) parent map.
    private static string Genus(Dictionary<string, string> parent, string m) => parent.TryGetValue(m, out var p) ? p : "";
    private static string DomainOf(Dictionary<string, string> parent, string m)
    {
        var cur = m; var guard = 0;
        while (guard++ < 32 && parent.TryGetValue(cur, out var p)) { if (p == Root) return cur; cur = p; }
        return "";
    }

    private static double Pct(int r, int t) => 100.0 * r / Math.Max(1, t);

    [SlowFact]
    public void QueryConditioned_GoalEmergent_SameAnchorDifferentCue_RelaxesToDifferentAnswers()
    {
        var sw = Stopwatch.StartNew();
        var device = cuda.is_available() ? CUDA : CPU;
        manual_seed(7);

        var w = BuildTrainedWorld();
        var space = w.Space;
        _out.WriteLine($"=== Curriculum (REAL vocab): {w.Parent.Count + 1} concepts | {w.Genera.Count} genera under " +
                       $"{w.Domains.Count} domains under '{Root}' | {w.Members.Count} leaf members | device={device.type} ===");

        using var net = new NavQueryPolicyNet(Dim, NavQueryFeatures.CueCount, Hidden, device);
        _out.WriteLine($"=== Net (query-conditioned): per-candidate F={net.FeatureLength}; cues={net.CueCount}; " +
                       $"params={net.ParameterCount():N0} | conditioned on (anchor, cue) — NO goal coordinate ===");

        // ── Build the cue curriculum: every member × {GENUS, DOMAIN, ROOT}. The cued ancestor is the only oracle. ──
        var trainQueries = new List<NavQueryDaggerTrainer.Query>();
        foreach (var m in w.Members)
        {
            var g = Genus(w.Parent, m); var d = DomainOf(w.Parent, m);
            if (g.Length == 0 || d.Length == 0) continue;
            trainQueries.Add(new NavQueryDaggerTrainer.Query(m, CueGenus, g));
            trainQueries.Add(new NavQueryDaggerTrainer.Query(m, CueDomain, d));
            trainQueries.Add(new NavQueryDaggerTrainer.Query(m, CueRoot, Root));
        }

        // ── BC warm-start on the cued flow fields ──
        var bc = NavQueryDaggerTrainer.BuildQueryTrajectories(space, trainQueries, K);
        Assert.NotEmpty(bc);
        var bcLoss = NavQueryDaggerTrainer.TrainQuery(net, bc, BcEpochs, Lr, device, K);
        _out.WriteLine($"=== BC warm-start: {bc.Count} query trajectories, {bc.Sum(t => t.Steps.Count)} steps | " +
                       $"CE={bcLoss.CrossEntropy:F4} haltBCE={bcLoss.HaltBce:F4} valueMSE={bcLoss.ValueMse:F4} ===");

        // ── 1 on-policy DAgger round (roll the net under each cue, label from the cued oracle) ──
        var aggregate = new List<NavQueryTrajectory>(bc);
        for (var round = 1; round <= DaggerRounds; round++)
        {
            var rollouts = NavQueryDaggerTrainer.RolloutQueryTrajectories(net, space, trainQueries, device, MaxSteps, K);
            aggregate.AddRange(rollouts);
            var loss = NavQueryDaggerTrainer.TrainQuery(net, aggregate, DaggerEpochs, Lr, device, K);
            _out.WriteLine($"=== DAgger round {round}: +{rollouts.Count} rollouts (agg {aggregate.Count}) | " +
                           $"CE={loss.CrossEntropy:F4} haltBCE={loss.HaltBce:F4} valueMSE={loss.ValueMse:F4} ===");
        }

        // ═══════════ PROOF 1 — GOAL-EMERGENT on TRAINED members (NO answer supplied), reached% per cue ═══════════
        // Sample to keep eval fast.
        var evalMembers = w.Members.OrderBy(x => x, StringComparer.Ordinal).Where((_, i) => i % 4 == 0).Take(30).ToList();
        var (g1r, g1t) = ReachedCue(net, space, evalMembers, CueGenus, m => Genus(w.Parent, m), device);
        var (d1r, d1t) = ReachedCue(net, space, evalMembers, CueDomain, m => DomainOf(w.Parent, m), device);
        var (r1r, r1t) = ReachedCue(net, space, evalMembers, CueRoot, _ => Root, device);
        _out.WriteLine("=== [1] GOAL-EMERGENT on trained members (no answer supplied) ===");
        _out.WriteLine($"    GENUS  = {Pct(g1r, g1t):F1}% ({g1r}/{g1t})");
        _out.WriteLine($"    DOMAIN = {Pct(d1r, d1t):F1}% ({d1r}/{d1t})");
        _out.WriteLine($"    ROOT   = {Pct(r1r, r1t):F1}% ({r1r}/{r1t})");

        // ═══════════ PROOF 2 — THE HEADLINE: same anchor, three cues → three DISTINCT correct answers ═══════════
        var headlineMembers = new[] { "dog", "robin", "salmon", "snake", "ant", "spider", "oak", "rose", "wheat", "carrot" }
            .Where(m => w.Parent.ContainsKey(m)).ToList();
        var allThree = 0;
        _out.WriteLine("=== [2] HEADLINE — same anchor, different cue → different answer (no answer supplied) ===");
        foreach (var m in headlineMembers)
        {
            var gExp = Genus(w.Parent, m); var dExp = DomainOf(w.Parent, m);
            var rg = NavQueryDaggerTrainer.WalkQuery(net, space, m, CueGenus, gExp, device, MaxSteps, K);
            var rd = NavQueryDaggerTrainer.WalkQuery(net, space, m, CueDomain, dExp, device, MaxSteps, K);
            var rr = NavQueryDaggerTrainer.WalkQuery(net, space, m, CueRoot, Root, device, MaxSteps, K);
            var distinct = new HashSet<string> { gExp, dExp, Root }.Count == 3; // the three cued answers are genuinely different
            var ok = rg.Reached && rd.Reached && rr.Reached && distinct;
            if (ok) allThree++;
            _out.WriteLine($"    {m,-10} GENUS→{rg.Final,-10}{(rg.Reached ? "OK" : "..")}  " +
                           $"DOMAIN→{rd.Final,-9}{(rd.Reached ? "OK" : "..")}  ROOT→{rr.Final,-9}{(rr.Reached ? "OK" : "..")}" +
                           $"   {(ok ? "[ALL THREE DISTINCT & CORRECT]" : "")}");
        }
        var headlineFrac = Pct(allThree, headlineMembers.Count);
        _out.WriteLine($"    >>> all-three-distinct-correct = {allThree}/{headlineMembers.Count} = {headlineFrac:F1}%");

        // ═══════════ PROOF 3 — HELD-OUT generalization (subtrees added post-training), reached% per cue ═══════════
        var (held, heldParent) = AddHeldout(space);
        var merged = new Dictionary<string, string>(w.Parent, StringComparer.Ordinal);
        foreach (var kv in heldParent) merged[kv.Key] = kv.Value;
        var heldMembers = held.Where(h => !HeldGenera.Any(g => g.Genus == h)).ToList(); // leaves only (members + salmon leaves)
        var (gHr, gHt) = ReachedCue(net, space, heldMembers, CueGenus, m => Genus(merged, m), device);
        var (dHr, dHt) = ReachedCue(net, space, heldMembers, CueDomain, m => DomainOf(merged, m), device);
        var (rHr, rHt) = ReachedCue(net, space, heldMembers, CueRoot, _ => Root, device);
        _out.WriteLine("=== [3] HELD-OUT generalization (no answer supplied; baseline goal-supplied held-out ≈ 78.6%) ===");
        _out.WriteLine($"    GENUS  = {Pct(gHr, gHt):F1}% ({gHr}/{gHt})");
        _out.WriteLine($"    DOMAIN = {Pct(dHr, dHt):F1}% ({dHr}/{dHt})");
        _out.WriteLine($"    ROOT   = {Pct(rHr, rHt):F1}% ({rHr}/{rHt})");

        // ═══════════ PROOF 4 — ABSTENTION: an ill-posed cue-target → no confident halt on a real answer ═══════════
        _out.WriteLine("=== [4] ABSTENTION — ill-posed cue (no such ancestor) → not reached, no hang ===");
        // (a) the ROOT itself queried for a GENUS/DOMAIN: nothing lies above it.
        var abs1 = NavQueryDaggerTrainer.WalkQuery(net, space, Root, CueGenus, "<none>", device, MaxSteps, K);
        var abs2 = NavQueryDaggerTrainer.WalkQuery(net, space, "animal", CueGenus, "<none>", device, MaxSteps, K);
        _out.WriteLine($"    (organism, GENUS): halted={abs1.Halted} final='{abs1.Final}' steps={abs1.Steps} reached={abs1.Reached}");
        _out.WriteLine($"    (animal,   GENUS): halted={abs2.Halted} final='{abs2.Final}' steps={abs2.Steps} reached={abs2.Reached}");

        sw.Stop();
        _out.WriteLine("=== SUMMARY (query relaxes to its answer with the goal EMERGENT, not supplied) ===");
        _out.WriteLine($"    wall time = {sw.Elapsed.TotalSeconds:F1}s on {device.type}");
        _out.WriteLine($"    [1] trained reached%:  GENUS={Pct(g1r, g1t):F1}  DOMAIN={Pct(d1r, d1t):F1}  ROOT={Pct(r1r, r1t):F1}");
        _out.WriteLine($"    [2] HEADLINE same-anchor-diff-cue all-distinct-correct = {headlineFrac:F1}% ({allThree}/{headlineMembers.Count})");
        _out.WriteLine($"    [3] held-out reached%: GENUS={Pct(gHr, gHt):F1}  DOMAIN={Pct(dHr, dHt):F1}  ROOT={Pct(rHr, rHt):F1}");
        _out.WriteLine($"    [4] abstention: organism/animal under GENUS did NOT reach a real ancestor " +
                       $"({(!abs1.Reached && !abs2.Reached ? "ABSTAINED" : "HALLUCINATED")})");

        // Loose sanity floors (HONEST MEASUREMENT — the DATA is the result):
        Assert.True(g1t > 0 && d1t > 0 && r1t > 0 && gHt > 0, "eval sets must be non-empty");
        Assert.True(!abs1.Reached, "querying the root for a genus must not reach a real ancestor (abstain)");
    }

    // Goal-EMERGENT reached% under a fixed cue: for each member, walk with (anchor, cue) and NO answer; reached iff it
    // confidently halts on the cue's expected ancestor.
    private static (int Reached, int Total) ReachedCue(
        NavQueryPolicyNet net, DialecticalSpace space, IEnumerable<string> members, int cue,
        Func<string, string> expected, Device device)
    {
        var reached = 0; var total = 0;
        foreach (var m in members)
        {
            var exp = expected(m);
            if (exp.Length == 0 || !space.TryGetConceptFace(m, out _)) continue;
            total++;
            if (NavQueryDaggerTrainer.WalkQuery(net, space, m, cue, exp, device, MaxSteps, K).Reached) reached++;
        }
        return (reached, total);
    }
}
