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
/// THE NAVIGATOR'S LEARNED POLICY (PLATONIC_NAVIGATOR.md §6 thin policy/value net, §7 behavioural cloning on the
/// flow-field oracle, §10 continuous target-emit + lattice snap). Proves the NN LEARNS TO WALK the platonic space —
/// it reproduces the deterministic oracle on a TRAINED multi-hop fact graph, so navigation is genuinely step-by-step
/// (Next[cur] is an INTERMEDIATE node, not the goal), not a one-hop "emit the answer".
///
/// PRODUCTION SIZING (nova-production-standard): policy-net hidden width = 2048, FaceDimension = 512. The curriculum
/// is hundreds of concepts across many multi-hop chains; a held-out set of chains gives a real generalization read.
/// Trains the net → gated [SlowFact] behind RUN_SLOW=1.
/// </summary>
public sealed class NavigatorPolicyTests
{
    private readonly ITestOutputHelper _out;
    public NavigatorPolicyTests(ITestOutputHelper o) => _out = o;

    // PRODUCTION sizing.
    private const int Dim = 512;       // FaceDimension (the decodable address space)
    private const int Hidden = 2048;   // production HiddenSize — the thin trunk width
    private const int Epochs = 600;
    private const double Lr = 1e-3;

    private static void Relate(DialecticalSpace s, string a, string b) => s.ObserveContradiction(a, b, 0.0);

    // Deterministic, distinct pseudo-words (single tokens → distinct char-face identities, so TryLand can separate
    // hundreds of concepts). CV syllables; 2–3 per word; deduped.
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

    // The hand-built world: many disjoint multi-hop chains (so the oracle's Next is an intermediate node), a few
    // one-hop fact stars, and a HELD-OUT set of chains never shown to BC (generalization read).
    private sealed record Curriculum(
        DialecticalSpace Space,
        List<string[]> TrainChains,      // each: node0 -> node1 -> ... -> answer
        List<(string Category, string[] Entities)> FactStars, // entity -> category (1-hop)
        List<string[]> HoldoutChains,
        List<string> TrainAnswers);

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
        var holdoutChains = MakeChains(12);  // ~48 concepts, held out from BC

        // One-hop fact stars: 5 categories, 6 entities each (Cost 1, Next = category).
        var factStars = new List<(string, string[])>();
        for (var c = 0; c < 5; c++)
        {
            var category = NextWord();
            var entities = new string[6];
            for (var e = 0; e < 6; e++) { entities[e] = NextWord(); Relate(space, entities[e], category); }
            factStars.Add((category, entities));
        }

        var trainAnswers = trainChains.Select(ch => ch[^1])
            .Concat(factStars.Select(f => f.Item1))
            .ToList();

        return new Curriculum(space, trainChains, factStars, holdoutChains, trainAnswers);
    }

    [SlowFact]
    public void LearnedPolicy_ClonesFlowField_AndWalksTheSpace()
    {
        var device = cuda.is_available() ? CUDA : CPU;
        manual_seed(7);

        var world = BuildWorld();
        var space = world.Space;

        var conceptCount = world.TrainChains.Sum(c => c.Length)
                         + world.FactStars.Sum(f => f.Entities.Length + 1)
                         + world.HoldoutChains.Sum(c => c.Length);
        _out.WriteLine($"=== Curriculum: {conceptCount} concepts | {world.TrainChains.Count} train chains + " +
                       $"{world.FactStars.Count} fact stars + {world.HoldoutChains.Count} holdout chains | device={device.type} ===");

        // ── Build the BC dataset from the flow-field oracle and train ───────────────────────────────────────────────
        var dataset = NavigatorBcTrainer.BuildDataset(space, world.TrainAnswers);
        Assert.NotEmpty(dataset);

        using var net = new NavigatorPolicyNet(Dim, Hidden, device);
        var paramCount = net.ParameterCount();
        _out.WriteLine($"=== Net: in={2 * Dim} -> {Hidden} -> {Hidden} -> heads(target={Dim}, value=1, halt=1) | " +
                       $"params={paramCount:N0} | BC examples={dataset.Count}, epochs={Epochs}, lr={Lr} ===");

        var losses = NavigatorBcTrainer.Train(net, dataset, Epochs, Lr, device);
        _out.WriteLine($"=== BC losses: targetMSE={losses.TargetMse:F5} valueMSE={losses.ValueMse:F5} haltBCE={losses.HaltBce:F5} ===");

        // TEST 1 — BC convergence: masked target-MSE and halt-BCE are low.
        Assert.True(losses.TargetMse < 0.02, $"target MSE must be low (cloned the field), got {losses.TargetMse:F5}");
        Assert.True(losses.HaltBce < 0.05, $"halt BCE must be low, got {losses.HaltBce:F5}");

        // ── TEST 2 (PRIMARY) — the learned policy WALKS to the answers on the trained graph ────────────────────────
        var policy = new NavNetPolicy(net, device);
        var reached = 0; var totalStarts = 0;
        var agree = 0; var totalNodes = 0;

        void Eval(string[] chain)
        {
            var answer = chain[^1];
            space.TryGetConceptFace(answer, out var goalFace);
            var field = FlowFieldOracle.Compute(space, answer);
            for (var j = 0; j < chain.Length - 1; j++) // every non-answer start
            {
                var start = chain[j];
                var result = new NavigatorWalk().Walk(space, start, goalFace, answer, policy, new NavWalkOptions(MaxSteps: 16));
                totalStarts++;
                if (result.Reached) reached++;

                // per-step agreement: does the net's emitted target land on the oracle's Next[start]?
                if (field.TryNext(start, out var oracleNext) && space.TryGetConceptFace(start, out var sf))
                {
                    totalNodes++;
                    var decision = policy.Decide(new NavState(start, sf, goalFace, answer, 0));
                    if (!decision.Halt && decision.Target.Length > 0 &&
                        space.TryLand(decision.Target, out var landed, out _, out _, out _) &&
                        string.Equals(landed, oracleNext, StringComparison.Ordinal))
                        agree++;
                }
            }
        }

        foreach (var chain in world.TrainChains) Eval(chain);
        // fact stars as 1-hop "chains" (entity -> category)
        foreach (var (category, entities) in world.FactStars)
            foreach (var e in entities) Eval(new[] { e, category });

        var reachedPct = 100.0 * reached / Math.Max(1, totalStarts);
        var agreePct = 100.0 * agree / Math.Max(1, totalNodes);
        _out.WriteLine($"=== PRIMARY: learned policy reached {reached}/{totalStarts} starts = {reachedPct:F1}% ===");
        _out.WriteLine($"=== per-step agreement (net target lands on oracle Next): {agree}/{totalNodes} = {agreePct:F1}% ===");

        // ── TEST 3 (STRETCH, report only — NOT gated) — held-out chains generalization ────────────────────────────
        var hReached = 0; var hStarts = 0;
        foreach (var chain in world.HoldoutChains)
        {
            var answer = chain[^1];
            space.TryGetConceptFace(answer, out var goalFace);
            for (var j = 0; j < chain.Length - 1; j++)
            {
                var result = new NavigatorWalk().Walk(space, chain[j], goalFace, answer, policy, new NavWalkOptions(MaxSteps: 16));
                hStarts++;
                if (result.Reached) hReached++;
            }
        }
        var hPct = 100.0 * hReached / Math.Max(1, hStarts);
        _out.WriteLine($"=== STRETCH (held-out, ungated): reached {hReached}/{hStarts} = {hPct:F1}% ===");

        // PRIMARY BAR: ≥ 90% of trained starts reached.
        Assert.True(reachedPct >= 90.0, $"learned policy must walk to the answer on >=90% of trained starts, got {reachedPct:F1}%");
    }
}
