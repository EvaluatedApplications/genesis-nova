using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  TRAINING THE DIFFERENTIAL RECOGNISER — BC WARM-START → ON-POLICY DAgger  (PLATONIC_NAVIGATOR.md §7).
//
//  The prior BC policy memorised a per-node lookup (10% held-out). Two changes fix the generalization gap:
//    (1) the net is a DIFFERENTIAL RECOGNISER over the local candidate set (NavFeatures) — it reads structure, and
//    (2) it is reinforced ON ITS OWN TRAJECTORY by DAgger: roll out the CURRENT policy's actual multi-step walk, and at
//        EVERY visited state query the flow-field oracle for the correct candidate (Next[visited]) — the field covers
//        the WHOLE graph, so even when the learner strays off-path the right correction is already there (DAgger for
//        free). The right decision is reinforced exactly where the NN actually goes.
//
//  The training unit is a TRAJECTORY (a sequence of decisions) so the recurrent self h_t is threaded faithfully, the
//  same way the live walk threads it. BC trajectories follow the ORACLE (Next[] from each start to the answer); DAgger
//  trajectories follow the NET (its real walk) with oracle labels stapled on at each visited node.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>One decision in a training trajectory: the egocentric differential features (K·F), the candidate mask (K),
/// the (cur−goal) context (dim), the oracle's correct candidate INDEX (−1 = ignore: at the answer, or a strayed node
/// the field can't correct), the halt target (1 at the answer), and the cost-to-go value target.</summary>
public readonly record struct NavStep(float[] FeaturesFlat, float[] Mask, float[] Context, int TargetIdx, float Halt, float Value);

/// <summary>A training trajectory: the goal face (seeds h₀) and an ordered list of decisions (so the GRU self threads
/// across hops exactly as the live walk threads it).</summary>
public sealed class NavTrajectory
{
    public required float[] GoalFace { get; init; }
    public required List<NavStep> Steps { get; init; }
}

/// <summary>Final-epoch composite losses (the convergence read).</summary>
public readonly record struct NavTrainLosses(double CrossEntropy, double HaltBce, double ValueMse);

/// <summary>
/// Behavioural-cloning warm-start + on-policy DAgger trainer for the recurrent differential navigator
/// (PLATONIC_NAVIGATOR.md §7). <see cref="BuildOracleTrajectories"/> distils the flow field into BC sequences;
/// <see cref="RolloutDaggerTrajectories"/> rolls the current net and labels each visited state from the oracle;
/// <see cref="TrainOnTrajectories"/> fits the net (Adam) under masked candidate-CE + halt-BCE + value-MSE, threading
/// the self h_t across each trajectory and bucketing equal-length trajectories into batches.
/// </summary>
public static class NavigatorDaggerTrainer
{
    public const int DefaultK = 16;

    // ──────────────────────────────────────────────────────────── BC warm-start: clone the oracle's whole flow field.

    /// <summary>
    /// Build the BC warm-start trajectories. For each answer, compute the flow field once, then from EVERY reachable
    /// non-answer node follow <c>Next[]</c> to the answer, emitting one trajectory (each node a decision, plus a
    /// terminal HALT decision on the answer). The dense field means every node is covered as a start — the
    /// DAgger-for-free property, exploited even in the warm start.
    /// </summary>
    public static IReadOnlyList<NavTrajectory> BuildOracleTrajectories(
        DialecticalSpace space, IEnumerable<string> answers, int k = DefaultK, double minConfidence = 0.0)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(answers);

        var trajectories = new List<NavTrajectory>();
        foreach (var answer in answers.Distinct(StringComparer.Ordinal))
        {
            if (string.IsNullOrWhiteSpace(answer) || !space.TryGetConceptFace(answer, out var goalFace)) continue;
            var field = FlowFieldOracle.Compute(space, answer);
            var goalF = ToFloat(goalFace);

            foreach (var start in field.Cost.Keys)
            {
                if (string.Equals(start, answer, StringComparison.Ordinal)) continue;

                // Follow the oracle from `start` to the answer (guard against pathological cycles with a visited set).
                var path = new List<string>();
                var seen = new HashSet<string>(StringComparer.Ordinal);
                var cur = start;
                while (seen.Add(cur))
                {
                    path.Add(cur);
                    if (string.Equals(cur, answer, StringComparison.Ordinal)) break;
                    if (!field.TryNext(cur, out var next)) break;
                    cur = next;
                }
                if (path.Count == 0 || !string.Equals(path[^1], answer, StringComparison.Ordinal)) continue; // didn't terminate on answer

                var steps = BuildSteps(space, path, answer, goalFace, field, k, minConfidence);
                if (steps.Count > 0) trajectories.Add(new NavTrajectory { GoalFace = goalF, Steps = steps });
            }
        }
        return trajectories;
    }

    // ──────────────────────────────────────────────────────────── DAgger: roll the NET, label from the oracle.

    /// <summary>
    /// Roll out the CURRENT policy from each given start and label every visited state with the oracle's correct move —
    /// the on-policy reinforcement (PLATONIC_NAVIGATOR.md §7). The walk is the net's REAL multi-step trajectory (it may
    /// stray); the flow field still has the correct <c>Next[]</c> at every reachable strayed node, so the right decision
    /// is reinforced exactly where the NN goes. Starts are grouped by their answer (one field per answer).
    /// </summary>
    public static IReadOnlyList<NavTrajectory> RolloutDaggerTrajectories(
        NavigatorPolicyNet net, DialecticalSpace space, IEnumerable<(string Start, string Answer)> starts,
        Device device, int maxSteps = 16, int k = DefaultK, double minConfidence = 0.0)
    {
        ArgumentNullException.ThrowIfNull(net);
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(starts);

        var trajectories = new List<NavTrajectory>();
        var fields = new Dictionary<string, PlatonicFlowField>(StringComparer.Ordinal);
        var walk = new NavigatorWalk();

        foreach (var (start, answer) in starts)
        {
            if (!space.TryGetConceptFace(answer, out var goalFace)) continue;
            if (!fields.TryGetValue(answer, out var field))
            {
                field = FlowFieldOracle.Compute(space, answer);
                fields[answer] = field;
            }

            using var policy = new NavNetPolicy(net, space, device, k, minConfidence);
            var result = walk.Walk(space, start, goalFace, answer, policy, new NavWalkOptions(MaxSteps: maxSteps));

            var steps = BuildSteps(space, result.Trajectory, answer, goalFace, field, k, minConfidence);
            if (steps.Count > 0)
                trajectories.Add(new NavTrajectory { GoalFace = ToFloat(goalFace), Steps = steps });
        }
        return trajectories;
    }

    // Build the per-node decision steps along a visited path, labelling each from the oracle field.
    private static List<NavStep> BuildSteps(
        DialecticalSpace space, IReadOnlyList<string> path, string answer, double[] goalFace,
        PlatonicFlowField field, int k, double minConfidence)
    {
        var dim = goalFace.Length;
        var maxCost = field.Cost.Count == 0 ? 1.0 : field.Cost.Values.Max() + 1.0; // fallback value for off-graph strays
        var steps = new List<NavStep>(path.Count);
        foreach (var node in path)
        {
            if (!space.TryGetConceptFace(node, out var nodeFace) || nodeFace.Length != dim) continue;
            var obs = NavFeatures.Build(space, node, nodeFace, goalFace, k, minConfidence);

            var isAnswer = string.Equals(node, answer, StringComparison.Ordinal);
            var targetIdx = -1;
            if (!isAnswer && field.TryNext(node, out var next))
            {
                for (var i = 0; i < obs.CandidateSymbols.Count; i++)
                    if (string.Equals(obs.CandidateSymbols[i], next, StringComparison.Ordinal)) { targetIdx = i; break; }
            }

            var value = field.TryCost(node, out var c) ? (float)c : (float)maxCost;
            var context = new float[dim];
            for (var d = 0; d < dim; d++) context[d] = (float)(nodeFace[d] - goalFace[d]);

            steps.Add(new NavStep(obs.FeaturesFlat, obs.Mask, context, targetIdx, isAnswer ? 1f : 0f, value));
        }
        return steps;
    }

    // ──────────────────────────────────────────────────────────── Train: thread h_t, bucket equal-length trajectories.

    /// <summary>
    /// Fit the net on the trajectories (Adam). Loss per step = masked candidate cross-entropy (ignore_index −1) +
    /// halt BCE + value MSE; the self h_t is threaded across each trajectory (seeded from the goal). Trajectories are
    /// bucketed by length so each bucket batches into dense [B,T,…] tensors. Returns final-epoch mean losses.
    /// </summary>
    public static NavTrainLosses TrainOnTrajectories(
        NavigatorPolicyNet net, IReadOnlyList<NavTrajectory> trajectories, int epochs, double lr, Device device, int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(net);
        ArgumentNullException.ThrowIfNull(trajectories);
        if (trajectories.Count == 0) throw new ArgumentException("empty trajectory set", nameof(trajectories));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        var dim = net.Dim;
        var f = net.FeatureLength;

        // Bucket by trajectory length (number of decision steps) → uniform [B,T] batches with no padding.
        var buckets = trajectories.GroupBy(t => t.Steps.Count).Select(g => PackBucket(g.ToList(), dim, f, k, device)).ToList();

        net.train();
        net.to(device);
        using var opt = torch.optim.Adam(net.parameters(), lr);

        double finalCe = 0, finalHalt = 0, finalValue = 0;
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            double ceSum = 0, haltSum = 0, valueSum = 0; var ceTerms = 0;
            foreach (var b in buckets)
            {
                using var scope = NewDisposeScope();
                opt.zero_grad();

                var h = net.SeedHidden(b.Goal); // [B,H]
                Tensor loss = zeros(1, device: device);
                double bCe = 0, bHalt = 0, bValue = 0; var bCeTerms = 0;

                for (var t = 0; t < b.T; t++)
                {
                    using var featT = b.Features.select(1, t);   // [B,K,F]
                    using var maskT = b.Mask.select(1, t);       // [B,K]
                    using var ctxT = b.Context.select(1, t);     // [B,dim]

                    var (logits, haltLogit, value, hNext) = net.Step(featT, maskT, ctxT, h);
                    h = hNext;

                    using var targetT = b.Target.select(1, t);   // [B] long (−1 = ignore)
                    using var haltT = b.Halt.select(1, t).reshape(b.B, 1);
                    using var valueT = b.Value.select(1, t).reshape(b.B, 1);

                    var haltLoss = nn.functional.binary_cross_entropy_with_logits(haltLogit, haltT);
                    var valueLoss = nn.functional.mse_loss(value, valueT);
                    loss = loss + haltLoss + valueLoss;
                    bHalt += haltLoss.detach().cpu().item<float>();
                    bValue += valueLoss.detach().cpu().item<float>();

                    if (b.HasTarget[t]) // at least one non-ignored candidate label at this step
                    {
                        var ceLoss = nn.functional.cross_entropy(logits, targetT, ignore_index: -1);
                        loss = loss + ceLoss;
                        bCe += ceLoss.detach().cpu().item<float>();
                        bCeTerms++;
                    }
                }

                loss.backward();
                opt.step();

                ceSum += bCe; haltSum += bHalt; valueSum += bValue; ceTerms += bCeTerms;
            }

            if (epoch == epochs - 1)
            {
                finalCe = ceTerms > 0 ? ceSum / ceTerms : 0.0;
                var tSteps = buckets.Sum(b => b.T);
                finalHalt = tSteps > 0 ? haltSum / tSteps : 0.0;
                finalValue = tSteps > 0 ? valueSum / tSteps : 0.0;
            }
        }

        foreach (var b in buckets) b.Dispose();
        return new NavTrainLosses(finalCe, finalHalt, finalValue);
    }

    // ──────────────────────────────────────────────────────────── Evaluation helper (reached% over a set of chains).

    /// <summary>Walk the net from each non-answer start of each chain; return (reached, totalStarts). A chain is
    /// <c>[node0, …, answer]</c>; the answer is the last element.</summary>
    public static (int Reached, int Total) CountReached(
        NavigatorPolicyNet net, DialecticalSpace space, IEnumerable<string[]> chains, Device device,
        int maxSteps = 16, int k = DefaultK, double minConfidence = 0.0)
    {
        var walk = new NavigatorWalk();
        var reached = 0; var total = 0;
        foreach (var chain in chains)
        {
            if (chain.Length < 2) continue;
            var answer = chain[^1];
            if (!space.TryGetConceptFace(answer, out var goalFace)) continue;
            using var policy = new NavNetPolicy(net, space, device, k, minConfidence);
            for (var j = 0; j < chain.Length - 1; j++)
            {
                var r = walk.Walk(space, chain[j], goalFace, answer, policy, new NavWalkOptions(MaxSteps: maxSteps));
                total++;
                if (r.Reached) reached++;
            }
        }
        return (reached, total);
    }

    // ──────────────────────────────────────────────────────────── Packing.

    private sealed class Bucket : IDisposable
    {
        public required int B { get; init; }
        public required int T { get; init; }
        public required Tensor Goal { get; init; }      // [B, dim]
        public required Tensor Features { get; init; }  // [B, T, K, F]
        public required Tensor Mask { get; init; }      // [B, T, K]
        public required Tensor Context { get; init; }   // [B, T, dim]
        public required Tensor Target { get; init; }    // [B, T] long
        public required Tensor Halt { get; init; }      // [B, T]
        public required Tensor Value { get; init; }     // [B, T]
        public required bool[] HasTarget { get; init; } // length T

        public void Dispose()
        {
            Goal.Dispose(); Features.Dispose(); Mask.Dispose(); Context.Dispose();
            Target.Dispose(); Halt.Dispose(); Value.Dispose();
        }
    }

    private static Bucket PackBucket(List<NavTrajectory> trajs, int dim, int f, int k, Device device)
    {
        var b = trajs.Count;
        var t = trajs[0].Steps.Count;

        var goal = new float[b * dim];
        var feat = new float[b * t * k * f];
        var mask = new float[b * t * k];
        var ctx = new float[b * t * dim];
        var target = new long[b * t];
        var halt = new float[b * t];
        var value = new float[b * t];
        var hasTarget = new bool[t];

        for (var bi = 0; bi < b; bi++)
        {
            Array.Copy(trajs[bi].GoalFace, 0, goal, bi * dim, dim);
            for (var ti = 0; ti < t; ti++)
            {
                var s = trajs[bi].Steps[ti];
                Array.Copy(s.FeaturesFlat, 0, feat, (bi * t + ti) * k * f, k * f);
                Array.Copy(s.Mask, 0, mask, (bi * t + ti) * k, k);
                Array.Copy(s.Context, 0, ctx, (bi * t + ti) * dim, dim);
                target[bi * t + ti] = s.TargetIdx;
                halt[bi * t + ti] = s.Halt;
                value[bi * t + ti] = s.Value;
                if (s.TargetIdx >= 0) hasTarget[ti] = true;
            }
        }

        return new Bucket
        {
            B = b,
            T = t,
            Goal = tensor(goal, new long[] { b, dim }, device: device),
            Features = tensor(feat, new long[] { b, t, k, f }, device: device),
            Mask = tensor(mask, new long[] { b, t, k }, device: device),
            Context = tensor(ctx, new long[] { b, t, dim }, device: device),
            Target = tensor(target, new long[] { b, t }, device: device),
            Halt = tensor(halt, new long[] { b, t }, device: device),
            Value = tensor(value, new long[] { b, t }, device: device),
            HasTarget = hasTarget,
        };
    }

    private static float[] ToFloat(double[] x)
    {
        var r = new float[x.Length];
        for (var i = 0; i < x.Length; i++) r[i] = (float)x[i];
        return r;
    }
}
