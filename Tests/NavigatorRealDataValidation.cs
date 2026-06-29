using System;
using System.Collections.Generic;
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
/// VALIDATE THE PLATONIC NAVIGATOR ON A REALISTIC SEMANTIC GRAPH (PLATONIC_NAVIGATOR.md §5/§6 the walk + differential
/// recogniser, §7 the flow-field oracle). The sibling <see cref="NavigatorDaggerTests"/> proves the mechanism on
/// SYNTHETIC nonce chains (disjoint CV-syllable chains, held-out reached ≈ 48–64%). This test asks the harder question:
/// does that generalization hold on REAL structure —
///   • real English vocabulary (meaningful spelling in the frozen char face, not nonce syllables),
///   • HUB categories with many members (mammal ≈ 27, insect ≈ 18 — above the K=16 egocentric fan-out → dilution risk),
///   • multi-hop taxonomies (member → genus → domain → root, depth 3–5) sharing those hubs, and
///   • settled distributional clouds (each is-a fact observed several times; clouds recomputed from relational context).
///
/// Held-out is REAL sub-structure: whole genus subtrees (amphibian/shrub and leaves under salmon) attached to TRAINED
/// hubs, never present during training (added to the SAME space only AFTER training, so there is NO leak into the dense
/// flow-field BC). The headline read is held-out reached% vs the ~48–64% synthetic baseline: does shared real structure
/// help or hurt generalization, and does a populous hub dilute navigation through it. Production sizing: Dim 1024,
/// Hidden 2048. Gated [SlowFact] behind RUN_SLOW=1. HONEST MEASUREMENT — the assertions are loose sanity floors; the
/// value is the reported DATA, not a forced bar.
/// </summary>
public sealed class NavigatorRealDataValidation
{
    private readonly ITestOutputHelper _out;
    public NavigatorRealDataValidation(ITestOutputHelper o) => _out = o;

    private const int Dim = 1024;       // production FaceDimension (orbital [416,1024) = 608)
    private const int Hidden = 2048;    // production HiddenSize
    private const int K = 16;           // egocentric candidate fan-out (the dilution threshold)
    private const int BcEpochs = 20;
    private const int DaggerRounds = 1;
    private const int DaggerEpochs = 20;
    private const double Lr = 1e-3;
    private const int MaxSteps = 16;
    private const int ObservePerEdge = 4; // observe each is-a fact several times so clouds settle (presence-based; no drown)

    private const string Root = "organism";

    private static void Plant(DialecticalSpace s, Dictionary<string, string> parent, string child, string par)
    {
        parent[child] = par;
        for (var i = 0; i < ObservePerEdge; i++) s.ObserveContradiction(child, par, 0.0); // is-a edge, perfect agreement
    }

    // ── The realistic taxonomy (real English vocabulary). Two domains under one root; genera of widely varying member
    //    count so hub size straddles K (mammal 26 + dog, insect 17 → degree > K; fish/reptile/arachnid/mollusk < K). ──
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

    // Depth-5 trained: dog breeds under "dog" (itself a mammal member) → breed→dog→mammal→animal→organism (4 hops).
    private static readonly string[] DogBreeds = { "spaniel","terrier","poodle","beagle" };

    // ── HELD-OUT real sub-structure (added AFTER training): whole genus subtrees under TRAINED hubs, plus held leaves
    //    under the trained member "salmon" (depth-5 held chain chinook→salmon→fish→animal→organism). ──
    private static readonly (string Genus, string Domain, string[] Members)[] HeldGenera =
    {
        ("amphibian", "animal", new[] { "frog","toad","newt","salamander","axolotl" }),
        ("shrub", "plant", new[] { "holly","hazel","privet","boxwood" }),
    };
    private static readonly string[] HeldSalmonLeaves = { "chinook","sockeye","coho" };

    private sealed record World(
        DialecticalSpace Space,
        Dictionary<string, string> Parent,        // child → immediate category (trained only)
        List<string> Genera,                      // trained genus names
        List<string> Domains,                     // trained domain names
        List<string> TrainAnswers,                // navigation goals the net trains toward
        List<(string Start, string Answer)> TrainStarts);

    private static IReadOnlyList<string> Ancestors(Dictionary<string, string> parent, string node)
    {
        var anc = new List<string>();
        var cur = node;
        var guard = 0;
        while (parent.TryGetValue(cur, out var p) && guard++ < 32) { anc.Add(p); cur = p; }
        return anc;
    }

    private World BuildTrainedWorld()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var parent = new Dictionary<string, string>(StringComparer.Ordinal);
        var genera = new List<string>();
        var domains = new List<string>();

        foreach (var (domain, generaArr) in Taxonomy)
        {
            domains.Add(domain);
            Plant(space, parent, domain, Root);
            foreach (var (genus, members) in generaArr)
            {
                genera.Add(genus);
                Plant(space, parent, genus, domain);
                foreach (var m in members) Plant(space, parent, m, genus);
            }
        }
        foreach (var b in DogBreeds) Plant(space, parent, b, "dog");
        space.FlushCloudBatch(); // settle clouds from the full relational context

        // Goals the net trains toward: roots/domains + three genera (mammal big-hub, bird mid, tree mid) for the 1-hop read.
        var trainAnswers = new List<string> { Root, "animal", "plant", "mammal", "bird", "tree" };

        // Natural ancestor (node → trained-answer) pairs — the DAgger rollout starts (kept on-taxonomy; BC covers the dense field).
        var trainStarts = new List<(string, string)>();
        foreach (var node in parent.Keys)
            foreach (var a in Ancestors(parent, node))
                if (trainAnswers.Contains(a) && !string.Equals(a, node, StringComparison.Ordinal))
                    trainStarts.Add((node, a));

        return new World(space, parent, genera, domains, trainAnswers, trainStarts);
    }

    // Add the held-out subtrees to the SAME trained space (post-training, so they never leaked into the flow-field BC).
    // Returns the held-out concepts (genus + members + salmon leaves) and their immediate-parent map.
    private static (List<string> HeldNodes, Dictionary<string, string> HeldParent) AddHeldout(DialecticalSpace space)
    {
        var heldNodes = new List<string>();
        var heldParent = new Dictionary<string, string>(StringComparer.Ordinal);
        void PlantHeld(string child, string par)
        {
            heldParent[child] = par; heldNodes.Add(child);
            for (var i = 0; i < ObservePerEdge; i++) space.ObserveContradiction(child, par, 0.0);
        }

        foreach (var (genus, domain, members) in HeldGenera)
        {
            PlantHeld(genus, domain);
            foreach (var m in members) PlantHeld(m, genus);
        }
        foreach (var leaf in HeldSalmonLeaves) PlantHeld(leaf, "salmon"); // salmon is a TRAINED fish member
        space.FlushCloudBatch();
        return (heldNodes, heldParent);
    }

    // Walk the net from each (start → answer) pair; return (reached, total).
    private static (int Reached, int Total) ReachedPairs(
        NavigatorPolicyNet net, DialecticalSpace space, IEnumerable<(string Start, string Answer)> pairs, Device device)
    {
        var walk = new NavigatorWalk();
        var reached = 0; var total = 0;
        foreach (var (start, answer) in pairs)
        {
            if (!space.TryGetConceptFace(answer, out var goal)) continue;
            using var policy = new NavNetPolicy(net, space, device, K);
            var res = walk.Walk(space, start, goal, answer, policy, new NavWalkOptions(MaxSteps: MaxSteps));
            total++;
            if (res.Reached) reached++;
        }
        return (reached, total);
    }

    private static double Pct(int r, int t) => 100.0 * r / Math.Max(1, t);

    [SlowFact]
    public void Navigator_OnRealSemanticGraph_GeneralizesAndSurvivesHubDilution()
    {
        var device = cuda.is_available() ? CUDA : CPU;
        manual_seed(7);

        var w = BuildTrainedWorld();
        var space = w.Space;
        var members = w.Parent.Keys.Where(n => !w.Genera.Contains(n) && !w.Domains.Contains(n)).ToList();

        _out.WriteLine($"=== Curriculum (REAL vocabulary): {w.Parent.Count + 1} trained concepts | " +
                       $"{w.Genera.Count} genera under {w.Domains.Count} domains under root '{Root}' | " +
                       $"{members.Count} leaf members (incl {DogBreeds.Length} depth-5 dog breeds) | device={device.type} ===");

        // ── HUB MECHANISM TABLE: for every genus, its relational degree and whether its parent survives the top-K cut. ──
        _out.WriteLine($"=== Hub structure (degree vs K={K}; parent-in-top-K = can the UP move even be a candidate?) ===");
        foreach (var g in w.Genera)
        {
            var degree = space.GetNeighbors(g, PlatonicNeighborhoodType.Relational, 64).Count;
            var topK = space.GetNeighbors(g, PlatonicNeighborhoodType.Relational, K).Select(n => n.Concept).ToHashSet(StringComparer.Ordinal);
            var par = w.Parent[g];
            _out.WriteLine($"    {g,-10} degree={degree,2}  parent='{par}'  parentInTopK={(topK.Contains(par) ? "YES" : "no ")}" +
                           (degree > K ? "   << populous hub (> K)" : ""));
        }

        // ── The net ──
        using var net = new NavigatorPolicyNet(Dim, Hidden, device);
        _out.WriteLine($"=== Net: per-candidate F={net.FeatureLength}; params={net.ParameterCount():N0} ===");

        // ── BC warm-start on the dense flow field (trained graph only — held-out is not in the space yet → no leak). ──
        var bc = NavigatorDaggerTrainer.BuildOracleTrajectories(space, w.TrainAnswers, K);
        Assert.NotEmpty(bc);
        var bcLoss = NavigatorDaggerTrainer.TrainOnTrajectories(net, bc, BcEpochs, Lr, device, K);
        _out.WriteLine($"=== BC warm-start: {bc.Count} oracle trajectories, {bc.Sum(t => t.Steps.Count)} steps | " +
                       $"CE={bcLoss.CrossEntropy:F4} haltBCE={bcLoss.HaltBce:F4} valueMSE={bcLoss.ValueMse:F4} ===");

        // 1-hop (entity → immediate category) for answer genera; multi-hop (member → root).
        var oneHopPairs = members.Where(m => new[] { "mammal", "bird", "tree" }.Contains(w.Parent[m]))
                                  .OrderBy(x => x, StringComparer.Ordinal).Take(30)
                                  .Select(m => (m, w.Parent[m])).ToList();
        // FAST: sample the trained multi-hop read (the per-round eval dominates wall time); held-out below stays FULL.
        var multiHopPairs = members.OrderBy(x => x, StringComparer.Ordinal).Take(40).Select(m => (m, Root)).ToList();

        var (oh0r, oh0t) = ReachedPairs(net, space, oneHopPairs, device);
        var (mh0r, mh0t) = ReachedPairs(net, space, multiHopPairs, device);
        _out.WriteLine($"=== AFTER BC-ONLY: 1-hop={Pct(oh0r, oh0t):F1}% ({oh0r}/{oh0t})  multi-hop(member→root)={Pct(mh0r, mh0t):F1}% ({mh0r}/{mh0t}) ===");

        // ── ON-POLICY DAgger ──
        var aggregate = new List<NavTrajectory>(bc);
        var oneHopByRound = new List<double> { Pct(oh0r, oh0t) };
        var multiByRound = new List<double> { Pct(mh0r, mh0t) };
        for (var round = 1; round <= DaggerRounds; round++)
        {
            var rollouts = NavigatorDaggerTrainer.RolloutDaggerTrajectories(net, space, w.TrainStarts, device, MaxSteps, K);
            aggregate.AddRange(rollouts);
            var loss = NavigatorDaggerTrainer.TrainOnTrajectories(net, aggregate, DaggerEpochs, Lr, device, K);

            var (ohr, oht) = ReachedPairs(net, space, oneHopPairs, device);
            var (mhr, mht) = ReachedPairs(net, space, multiHopPairs, device);
            oneHopByRound.Add(Pct(ohr, oht));
            multiByRound.Add(Pct(mhr, mht));
            _out.WriteLine($"=== DAgger round {round}: +{rollouts.Count} rollouts (agg {aggregate.Count}) | CE={loss.CrossEntropy:F4} | " +
                           $"1-hop={Pct(ohr, oht):F1}%  multi-hop={Pct(mhr, mht):F1}% ({mhr}/{mht}) ===");
        }

        var finalOneHop = oneHopByRound[^1];
        var finalMulti = multiByRound[^1];

        // ── HUB STRESS (trained): member→root reached%, partitioned by the member's genus size (big > K vs small < K). ──
        var bigHubGenera = w.Genera.Where(g => space.GetNeighbors(g, PlatonicNeighborhoodType.Relational, 64).Count > K).ToHashSet(StringComparer.Ordinal);
        var smallStressGenera = new[] { "fish", "reptile", "arachnid", "mollusk" }.ToHashSet(StringComparer.Ordinal);
        var bigHubPairs = members.Where(m => bigHubGenera.Contains(w.Parent[m]) || (w.Parent[m] == "dog")).OrderBy(x => x, StringComparer.Ordinal).Take(25).Select(m => (m, Root)).ToList();
        var smallHubPairs = members.Where(m => smallStressGenera.Contains(w.Parent[m])).OrderBy(x => x, StringComparer.Ordinal).Take(25).Select(m => (m, Root)).ToList();
        var (bhr, bht) = ReachedPairs(net, space, bigHubPairs, device);
        var (shr, sht) = ReachedPairs(net, space, smallHubPairs, device);

        // ── ADD HELD-OUT real sub-structure to the SAME space, then evaluate generalization. ──
        var (heldNodes, heldParent) = AddHeldout(space);
        var heldMembers = heldNodes; // every held node (genus + members + salmon leaves) → root
        var heldPairs = heldMembers.Select(h => (h, Root)).ToList();
        var (hr, ht) = ReachedPairs(net, space, heldPairs, device);
        var finalHeld = Pct(hr, ht);

        // Held-out leaves under a SMALL trained hub (amphibian/shrub members, salmon leaves) vs the held genus nodes.
        var heldLeafPairs = heldMembers.Where(h => heldParent.ContainsKey(h) && !HeldGenera.Any(g => g.Genus == h)).Select(h => (h, Root)).ToList();
        var (hlr, hlt) = ReachedPairs(net, space, heldLeafPairs, device);

        // ── Per-step oracle agreement on held-out (does the net's emitted target land on the oracle's Next toward root?). ──
        var field = FlowFieldOracle.Compute(space, Root);
        if (!space.TryGetConceptFace(Root, out var rootFace)) throw new InvalidOperationException("root face missing");
        var agree = 0; var agreeTotal = 0;
        using (var policy = new NavNetPolicy(net, space, device, K))
        {
            foreach (var h in heldNodes)
            {
                if (!field.TryNext(h, out var oracleNext) || !space.TryGetConceptFace(h, out var hf)) continue;
                agreeTotal++;
                var d = policy.Decide(new NavState(h, hf, rootFace, Root, 0));
                if (!d.Halt && d.Target.Length > 0 &&
                    space.TryLand(d.Target, out var landed, out _, out _, out _) &&
                    string.Equals(landed, oracleNext, StringComparison.Ordinal))
                    agree++;
            }
        }
        var agreePct = Pct(agree, agreeTotal);

        // ── SUMMARY ──
        _out.WriteLine("=== SUMMARY (real semantic graph) ===");
        _out.WriteLine($"  1-hop by stage:     BC={oneHopByRound[0]:F1}%  " + string.Join("  ", Enumerable.Range(1, DaggerRounds).Select(r => $"D{r}={oneHopByRound[r]:F1}%")));
        _out.WriteLine($"  multi-hop by stage: BC={multiByRound[0]:F1}%  " + string.Join("  ", Enumerable.Range(1, DaggerRounds).Select(r => $"D{r}={multiByRound[r]:F1}%")));
        _out.WriteLine($"  [1] 1-hop entity→category (sanity, real vocab) = {finalOneHop:F1}% ({oh0t} probes)");
        _out.WriteLine($"  [2] trained multi-hop member→root             = {finalMulti:F1}% ({mh0t} probes)");
        _out.WriteLine($"  [3] HELD-OUT subtree node→root (HEADLINE)      = {finalHeld:F1}% ({hr}/{ht})   [synthetic nonce baseline ≈ 48–64%]");
        _out.WriteLine($"        held-out LEAVES only (members, not genus) = {Pct(hlr, hlt):F1}% ({hlr}/{hlt})");
        _out.WriteLine($"  [4] HUB STRESS member→root: BIG hubs (>K: {string.Join(",", bigHubGenera)}) = {Pct(bhr, bht):F1}% ({bhr}/{bht})   " +
                       $"vs SMALL hubs ({string.Join(",", smallStressGenera)}) = {Pct(shr, sht):F1}% ({shr}/{sht})");
        _out.WriteLine($"  per-step oracle agreement on held-out          = {agree}/{agreeTotal} = {agreePct:F1}%");
        _out.WriteLine($"  VERDICT: held-out {finalHeld:F1}% vs synthetic ~48–64%; hub dilution Δ = {Pct(bhr, bht) - Pct(shr, sht):F1} pts (small−big = {Pct(shr, sht) - Pct(bhr, bht):F1}).");

        // Loose sanity floors only (HONEST MEASUREMENT — the DATA above is the result, not a bar):
        Assert.True(oh0t > 0 && mh0t > 0 && ht > 0, "evaluation sets must be non-empty");
        Assert.True(finalOneHop >= 80.0, $"1-hop entity→category should be near-saturated on real vocab, got {finalOneHop:F1}%");
    }
}
