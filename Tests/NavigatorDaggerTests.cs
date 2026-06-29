using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE NAVIGATOR AS A LOCAL DIFFERENTIAL RECOGNISER, trained BC-warm-start → ON-POLICY DAgger (PLATONIC_NAVIGATOR.md
/// §2 egocentric observation, §3 the self threads the walk, §6 thin recogniser, §7 flow-field oracle = supervision).
///
/// The prior BC policy memorised a per-node lookup (held-out reached ≈ 10%). This rebuild makes the policy READ the
/// LOCAL relational structure — for each candidate neighbour, its contrast to the goal and to here (NavFeatures) — and
/// reinforces every routing decision on the NN's OWN multi-step trajectory (DAgger), the oracle correcting wherever the
/// learner actually goes. The PRIMARY read is held-out generalization: chains the net never trained on. Production
/// sizing: FaceDimension 512, hidden 2048. Gated [SlowFact] behind RUN_SLOW=1.
/// </summary>
public sealed class NavigatorDaggerTests
{
    private readonly ITestOutputHelper _out;
    public NavigatorDaggerTests(ITestOutputHelper o) => _out = o;

    private const int Dim = 512;        // FaceDimension (the decodable address space)
    private const int Hidden = 2048;    // production HiddenSize — the thin trunk width
    private const int K = 16;           // egocentric candidate fan-out
    // Deliberately LIGHT BC warm-start: enough to orient the net, NOT enough to solve the training graph — so the net
    // genuinely STRAYS on-policy and DAgger's oracle-correction-where-the-NN-actually-goes has something to fix (the
    // climb across rounds is the on-policy reinforcement made visible). A heavy BC on the dense field already saturates
    // (the flow field labels every node = "DAgger for free"), which hides the mechanism.
    private const int BcEpochs = 30;
    private const int DaggerRounds = 4;
    private const int DaggerEpochs = 35;
    private const double Lr = 1e-3;
    private const int MaxSteps = 16;

    // PRIMARY BAR: held-out reached% must rise FAR above the 10% BC-lookup baseline. (Reported honestly whatever it is.)
    private const double HeldoutBar = 40.0;

    private static void Relate(DialecticalSpace s, string a, string b) => s.ObserveContradiction(a, b, 0.0);

    // Deterministic distinct pseudo-words → distinct char-face identities (so TryLand separates hundreds of concepts).
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
        List<(string Start, string Answer)> TrainStarts);

    private static Curriculum BuildWorld()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var words = Pseudowords(123).GetEnumerator();
        string NextWord() { words.MoveNext(); return words.Current!; }
        var rng = new Random(7);

        List<string[]> MakeChains(int count)
        {
            var chains = new List<string[]>(count);
            for (var i = 0; i < count; i++)
            {
                var len = rng.Next(3, 6); // 3..5 nodes → multi-hop (Cost[start] up to 4)
                var chain = new string[len];
                for (var j = 0; j < len; j++) chain[j] = NextWord();
                for (var j = 0; j < len - 1; j++) Relate(space, chain[j], chain[j + 1]);
                chains.Add(chain);
            }
            return chains;
        }

        var trainChains = MakeChains(50);   // ~200 concepts across 50 multi-hop chains
        var holdoutChains = MakeChains(12);  // ~48 concepts, held out from ALL training

        var factStars = new List<(string, string[])>();
        for (var c = 0; c < 5; c++)
        {
            var category = NextWord();
            var entities = new string[6];
            for (var e = 0; e < 6; e++) { entities[e] = NextWord(); Relate(space, entities[e], category); }
            factStars.Add((category, entities));
        }

        var trainAnswers = trainChains.Select(ch => ch[^1]).Concat(factStars.Select(f => f.Item1)).ToList();

        // Every non-answer training node, paired with its chain/star answer — the DAgger rollout starts.
        var trainStarts = new List<(string, string)>();
        foreach (var ch in trainChains)
            for (var j = 0; j < ch.Length - 1; j++) trainStarts.Add((ch[j], ch[^1]));
        foreach (var (cat, ents) in factStars)
            foreach (var e in ents) trainStarts.Add((e, cat));

        return new Curriculum(space, trainChains, factStars, holdoutChains, trainAnswers, trainStarts);
    }

    // Fact stars as 1-hop "chains" (entity -> category) for the reached-% evaluator.
    private static IEnumerable<string[]> StarChains(IEnumerable<(string Category, string[] Entities)> stars)
        => stars.SelectMany(s => s.Entities.Select(e => new[] { e, s.Category }));

    private double TrainedReachedPct(NavigatorPolicyNet net, Curriculum w, Device device)
    {
        var (r1, t1) = NavigatorDaggerTrainer.CountReached(net, w.Space, w.TrainChains, device, MaxSteps, K);
        var (r2, t2) = NavigatorDaggerTrainer.CountReached(net, w.Space, StarChains(w.FactStars), device, MaxSteps, K);
        return 100.0 * (r1 + r2) / Math.Max(1, t1 + t2);
    }

    private double HeldoutReachedPct(NavigatorPolicyNet net, Curriculum w, Device device, out int reached, out int total)
    {
        var (r, t) = NavigatorDaggerTrainer.CountReached(net, w.Space, w.HoldoutChains, device, MaxSteps, K);
        reached = r; total = t;
        return 100.0 * r / Math.Max(1, t);
    }

    [SlowFact]
    public void DifferentialRecogniser_OnPolicyDagger_GeneralizesToHeldOutChains()
    {
        var device = cuda.is_available() ? CUDA : CPU;
        manual_seed(7);

        var world = BuildWorld();
        var space = world.Space;

        var conceptCount = world.TrainChains.Sum(c => c.Length)
                         + world.FactStars.Sum(f => f.Entities.Length + 1)
                         + world.HoldoutChains.Sum(c => c.Length);
        _out.WriteLine($"=== Curriculum: {conceptCount} concepts | {world.TrainChains.Count} train chains + " +
                       $"{world.FactStars.Count} fact stars + {world.HoldoutChains.Count} HELD-OUT chains | device={device.type} ===");

        using var net = new NavigatorPolicyNet(Dim, Hidden, device);
        _out.WriteLine($"=== Net (recurrent differential recogniser): per-candidate feature F={net.FeatureLength} " +
                       $"(=[cand−goal | cand−cur | cand]·{Dim} + κ); candEnc {net.FeatureLength}->{Hidden}; GRU self {Hidden}; " +
                       $"score MLP 2*{Hidden}->{Hidden}->1 per candidate; halt & value heads | params={net.ParameterCount():N0} ===");

        // ── BC WARM-START — clone the flow field as threaded oracle trajectories ──────────────────────────────────────
        var bc = NavigatorDaggerTrainer.BuildOracleTrajectories(space, world.TrainAnswers, K);
        Assert.NotEmpty(bc);
        var bcLoss = NavigatorDaggerTrainer.TrainOnTrajectories(net, bc, BcEpochs, Lr, device, K);
        _out.WriteLine($"=== BC warm-start: {bc.Count} oracle trajectories, {bc.Sum(t => t.Steps.Count)} steps, {BcEpochs} epochs | " +
                       $"CE={bcLoss.CrossEntropy:F4} haltBCE={bcLoss.HaltBce:F4} valueMSE={bcLoss.ValueMse:F4} ===");

        var trainedBc = TrainedReachedPct(net, world, device);
        var heldBc = HeldoutReachedPct(net, world, device, out var hr0, out var ht0);
        _out.WriteLine($"=== AFTER BC-ONLY:  trained reached = {trainedBc:F1}%  |  HELD-OUT reached = {heldBc:F1}% ({hr0}/{ht0}) ===");

        // ── ON-POLICY DAgger — roll the CURRENT net, label every visited state from the oracle, aggregate, retrain ────
        var aggregate = new List<NavTrajectory>(bc);
        var heldoutByRound = new List<double> { heldBc };
        var trainedByRound = new List<double> { trainedBc };

        for (var round = 1; round <= DaggerRounds; round++)
        {
            var rollouts = NavigatorDaggerTrainer.RolloutDaggerTrajectories(net, space, world.TrainStarts, device, MaxSteps, K);
            aggregate.AddRange(rollouts);
            var loss = NavigatorDaggerTrainer.TrainOnTrajectories(net, aggregate, DaggerEpochs, Lr, device, K);

            var trained = TrainedReachedPct(net, world, device);
            var held = HeldoutReachedPct(net, world, device, out var hr, out var ht);
            heldoutByRound.Add(held);
            trainedByRound.Add(trained);
            _out.WriteLine($"=== DAgger round {round}: +{rollouts.Count} on-policy trajectories (agg {aggregate.Count}) | " +
                           $"CE={loss.CrossEntropy:F4} haltBCE={loss.HaltBce:F4} valueMSE={loss.ValueMse:F4} | " +
                           $"trained = {trained:F1}%  |  HELD-OUT = {held:F1}% ({hr}/{ht}) ===");
        }

        // ── per-step oracle agreement on held-out (does the net's emitted target land on the oracle's Next?) ──────────
        var (agree, totalNodes) = HeldoutAgreement(net, world, device);
        var agreePct = 100.0 * agree / Math.Max(1, totalNodes);

        var finalTrained = trainedByRound[^1];
        var finalHeld = heldoutByRound[^1];
        _out.WriteLine("=== SUMMARY ===");
        _out.WriteLine($"  held-out by stage: BC-only={heldoutByRound[0]:F1}%  " +
                       string.Join("  ", Enumerable.Range(1, DaggerRounds).Select(r => $"DAgger{r}={heldoutByRound[r]:F1}%")));
        _out.WriteLine($"  trained graph by stage: BC-only={trainedByRound[0]:F1}%  " +
                       string.Join("  ", Enumerable.Range(1, DaggerRounds).Select(r => $"DAgger{r}={trainedByRound[r]:F1}%")));
        _out.WriteLine($"  held-out per-step oracle agreement: {agree}/{totalNodes} = {agreePct:F1}%");
        _out.WriteLine($"  DAgger LIFT (held-out): {heldoutByRound[0]:F1}% -> {finalHeld:F1}%");

        // TEST 1 (PRIMARY): held-out generalization is FAR above the 10% BC-lookup baseline.
        Assert.True(finalHeld >= HeldoutBar,
            $"held-out reached must be >> 10% (bar {HeldoutBar:F0}%), got {finalHeld:F1}%");
        // TEST 2: the trained graph is essentially solved.
        Assert.True(finalTrained >= 90.0,
            $"trained-graph reached must be ~100%, got {finalTrained:F1}%");
    }

    // Per-step agreement on HELD-OUT chains: from each non-answer node, does the net's argmax-candidate land on the
    // oracle's Next[node]? (A clean read of "reads structure" independent of the multi-step walk.)
    private (int Agree, int Total) HeldoutAgreement(NavigatorPolicyNet net, Curriculum w, Device device)
    {
        var agree = 0; var total = 0;
        foreach (var chain in w.HoldoutChains)
        {
            var answer = chain[^1];
            if (!w.Space.TryGetConceptFace(answer, out var goalFace)) continue;
            var field = FlowFieldOracle.Compute(w.Space, answer);
            using var policy = new NavNetPolicy(net, w.Space, device, K);
            for (var j = 0; j < chain.Length - 1; j++)
            {
                var start = chain[j];
                if (!field.TryNext(start, out var oracleNext) || !w.Space.TryGetConceptFace(start, out var sf)) continue;
                total++;
                var d = policy.Decide(new NavState(start, sf, goalFace, answer, 0));
                if (!d.Halt && d.Target.Length > 0 &&
                    w.Space.TryLand(d.Target, out var landed, out _, out _, out _) &&
                    string.Equals(landed, oracleNext, StringComparison.Ordinal))
                    agree++;
            }
        }
        return (agree, total);
    }
}
