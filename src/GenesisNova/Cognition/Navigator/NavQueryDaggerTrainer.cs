using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  TRAINING THE QUERY-CONDITIONED NAVIGATOR — BC warm-start → on-policy DAgger  (PLATONIC_NAVIGATOR.md §7).
//
//  A training unit is a QUERY (member, cue, cued-ancestor). The cued ancestor is the ONLY thing that supplies the
//  oracle: GENUS = the member's immediate parent, DOMAIN = the domain ancestor, ROOT = the root. We compute the
//  flow-field oracle to that ancestor ONCE (cached per ancestor) and clone it — but the FEATURES the net trains on are
//  answer-free (anchored on the member + the cue), so train-time and inference-time inputs match exactly. The answer
//  appears ONLY in the labels (which candidate to step to; halt=1 AT the ancestor), never in the net's input.
//
//  The headline mechanism lives in the labels: the SAME member, with three different cues, yields three trajectories of
//  different length to three different halts — so the cue embedding (mixed into the GRU self + the halt head) is the
//  only thing that can disambiguate where to stop. That is what forces "same anchor, different cue → different answer".
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>One decision in a query trajectory: answer-free differential features (K·F), the candidate mask (K), the
/// (cur−anchor) displacement context (dim), the oracle's correct candidate INDEX (−1 = ignore), the halt target (1 at
/// the cued ancestor), and the cost-to-go value target.</summary>
public readonly record struct NavQueryStep(float[] FeaturesFlat, float[] Mask, float[] Context, int TargetIdx, float Halt, float Value);

/// <summary>A query trajectory: the anchor face + cue (seed the self h₀ and the per-hop question-tension) and an
/// ordered list of decisions (so the GRU self threads across hops exactly as the live walk threads it).</summary>
public sealed class NavQueryTrajectory
{
    public required float[] AnchorFace { get; init; }
    public required int Cue { get; init; }
    public required List<NavQueryStep> Steps { get; init; }
}

/// <summary>
/// BC warm-start + on-policy DAgger trainer for the query-conditioned navigator (PLATONIC_NAVIGATOR.md §7).
/// <see cref="BuildQueryTrajectories"/> distils the cued flow fields into BC sequences (oracle path member→cued
/// ancestor); <see cref="RolloutQueryTrajectories"/> rolls the current net from each (member, cue) and labels each
/// visited state from the oracle; <see cref="TrainQuery"/> fits the net threading the self + cue across each trajectory.
/// </summary>
public static class NavQueryDaggerTrainer
{
    public const int DefaultK = 16;

    /// <summary>A query the navigator must resolve: the <paramref name="Member"/> being asked about (the anchor), the
    /// <paramref name="Cue"/> (GENUS/DOMAIN/ROOT as an int), and the <paramref name="Ancestor"/> the cue points to.</summary>
    public readonly record struct Query(string Member, int Cue, string Ancestor);

    // ──────────────────────────────────────────────────────────── BC warm-start: clone the cued flow fields.

    /// <summary>
    /// Build BC trajectories. For each (member, cue, ancestor) compute (cache) the flow field toward the ancestor, then
    /// follow <c>Next[]</c> from the member to the ancestor, emitting one trajectory whose feature rows are anchored on
    /// the member + cue (answer-free) and whose labels come from the oracle (next candidate; halt=1 at the ancestor).
    /// </summary>
    public static IReadOnlyList<NavQueryTrajectory> BuildQueryTrajectories(
        DialecticalSpace space, IEnumerable<Query> queries, int k = DefaultK, double minConfidence = 0.0)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(queries);

        var fields = new Dictionary<string, PlatonicFlowField>(StringComparer.Ordinal);
        var trajectories = new List<NavQueryTrajectory>();
        foreach (var (member, cue, ancestor) in queries)
        {
            if (string.IsNullOrWhiteSpace(member) || string.IsNullOrWhiteSpace(ancestor)) continue;
            if (!space.TryGetConceptFace(member, out var anchorFace)) continue;
            var field = Field(space, ancestor, fields);

            var path = OraclePath(field, member, ancestor);
            if (path.Count == 0) continue;

            var steps = BuildQuerySteps(space, path, ancestor, anchorFace, field, k, minConfidence);
            if (steps.Count > 0)
                trajectories.Add(new NavQueryTrajectory { AnchorFace = ToFloat(anchorFace), Cue = cue, Steps = steps });
        }
        return trajectories;
    }

    // ──────────────────────────────────────────────────────────── DAgger: roll the NET, label from the cued oracle.

    /// <summary>
    /// Roll out the CURRENT query policy from each (member, cue) and label every visited state with the cued oracle's
    /// correct move (PLATONIC_NAVIGATOR.md §7). The walk is the net's REAL multi-step trajectory (goalSymbol=null → it
    /// relies on its own HALT and may stray); the cued flow field still has the correct <c>Next[]</c> at every reachable
    /// strayed node, so the right decision is reinforced exactly where the NN goes.
    /// </summary>
    public static IReadOnlyList<NavQueryTrajectory> RolloutQueryTrajectories(
        NavQueryPolicyNet net, DialecticalSpace space, IEnumerable<Query> queries,
        Device device, int maxSteps = 16, int k = DefaultK, double minConfidence = 0.0)
    {
        ArgumentNullException.ThrowIfNull(net);
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(queries);

        var fields = new Dictionary<string, PlatonicFlowField>(StringComparer.Ordinal);
        var trajectories = new List<NavQueryTrajectory>();
        var walk = new NavigatorWalk();

        foreach (var (member, cue, ancestor) in queries)
        {
            if (!space.TryGetConceptFace(member, out var anchorFace)) continue;
            var field = Field(space, ancestor, fields);

            using var policy = new QueryNavPolicy(net, space, anchorFace, cue, device, k, minConfidence);
            // goalFace is a harmless dummy (the policy ignores it); goalSymbol=null → the loop relies on the net's HALT.
            var result = walk.Walk(space, member, anchorFace, null, policy, new NavWalkOptions(MaxSteps: maxSteps));

            var steps = BuildQuerySteps(space, result.Trajectory, ancestor, anchorFace, field, k, minConfidence);
            if (steps.Count > 0)
                trajectories.Add(new NavQueryTrajectory { AnchorFace = ToFloat(anchorFace), Cue = cue, Steps = steps });
        }
        return trajectories;
    }

    private static PlatonicFlowField Field(DialecticalSpace space, string ancestor, Dictionary<string, PlatonicFlowField> cache)
    {
        if (!cache.TryGetValue(ancestor, out var field)) { field = FlowFieldOracle.Compute(space, ancestor); cache[ancestor] = field; }
        return field;
    }

    // Follow the oracle's Next[] from `start` to `ancestor` (guard pathological cycles). Empty if it never terminates there.
    private static List<string> OraclePath(PlatonicFlowField field, string start, string ancestor)
    {
        var path = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var cur = start;
        while (seen.Add(cur))
        {
            path.Add(cur);
            if (string.Equals(cur, ancestor, StringComparison.Ordinal)) return path;
            if (!field.TryNext(cur, out var next)) break;
            cur = next;
        }
        return new List<string>();
    }

    // Build the per-node decision steps along a visited path, labelling each from the cued oracle field.
    private static List<NavQueryStep> BuildQuerySteps(
        DialecticalSpace space, IReadOnlyList<string> path, string ancestor, double[] anchorFace,
        PlatonicFlowField field, int k, double minConfidence)
    {
        var dim = anchorFace.Length;
        var maxCost = field.Cost.Count == 0 ? 1.0 : field.Cost.Values.Max() + 1.0; // fallback value for off-graph strays
        var steps = new List<NavQueryStep>(path.Count);
        foreach (var node in path)
        {
            if (!space.TryGetConceptFace(node, out var nodeFace) || nodeFace.Length != dim) continue;
            var obs = NavQueryFeatures.Build(space, node, nodeFace, anchorFace, k, minConfidence);

            var isAncestor = string.Equals(node, ancestor, StringComparison.Ordinal);
            var targetIdx = -1;
            if (!isAncestor && field.TryNext(node, out var next))
            {
                for (var i = 0; i < obs.CandidateSymbols.Count; i++)
                    if (string.Equals(obs.CandidateSymbols[i], next, StringComparison.Ordinal)) { targetIdx = i; break; }
            }

            var value = field.TryCost(node, out var c) ? (float)c : (float)maxCost;
            var context = new float[dim];
            for (var d = 0; d < dim; d++) context[d] = (float)(nodeFace[d] - anchorFace[d]);

            steps.Add(new NavQueryStep(obs.FeaturesFlat, obs.Mask, context, targetIdx, isAncestor ? 1f : 0f, value));
        }
        return steps;
    }

    // ──────────────────────────────────────────────────────────── Train: thread self+cue, bucket equal-length trajectories.

    /// <summary>
    /// Fit the net on the query trajectories (Adam). Per-step loss = masked candidate cross-entropy (ignore −1) + halt
    /// BCE + value MSE; the self h_t is threaded across each trajectory (seeded from anchor⊕cue, cue mixed every hop).
    /// Trajectories bucket by length into dense [B,T,…] tensors. Returns final-epoch mean losses.
    /// </summary>
    public static NavTrainLosses TrainQuery(
        NavQueryPolicyNet net, IReadOnlyList<NavQueryTrajectory> trajectories, int epochs, double lr, Device device, int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(net);
        ArgumentNullException.ThrowIfNull(trajectories);
        if (trajectories.Count == 0) throw new ArgumentException("empty trajectory set", nameof(trajectories));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        var dim = net.Dim;
        var f = net.FeatureLength;

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

                var h = net.SeedHidden(b.Anchor, b.Cue); // [B,H]
                Tensor loss = zeros(1, device: device);
                double bCe = 0, bHalt = 0, bValue = 0; var bCeTerms = 0;

                for (var t = 0; t < b.T; t++)
                {
                    using var featT = b.Features.select(1, t);   // [B,K,F]
                    using var maskT = b.Mask.select(1, t);       // [B,K]
                    using var ctxT = b.Context.select(1, t);     // [B,dim]

                    var (logits, haltLogit, value, hNext) = net.Step(featT, maskT, ctxT, b.Cue, h);
                    h = hNext;

                    using var targetT = b.Target.select(1, t);   // [B] long (−1 = ignore)
                    using var haltT = b.Halt.select(1, t).reshape(b.B, 1);
                    using var valueT = b.Value.select(1, t).reshape(b.B, 1);

                    var haltLoss = nn.functional.binary_cross_entropy_with_logits(haltLogit, haltT);
                    var valueLoss = nn.functional.mse_loss(value, valueT);
                    loss = loss + haltLoss + valueLoss;
                    bHalt += haltLoss.detach().cpu().item<float>();
                    bValue += valueLoss.detach().cpu().item<float>();

                    if (b.HasTarget[t])
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

    // ──────────────────────────────────────────────────────────── Evaluation: goal-EMERGENT walk (NO answer supplied).

    /// <summary>The outcome of one query-conditioned walk: did it confidently HALT on the expected cued ancestor (with
    /// NO answer ever supplied), and where did it actually stand.</summary>
    public readonly record struct QueryWalkResult(bool Reached, bool Halted, string Final, int Steps);

    /// <summary>
    /// Walk the net from <paramref name="member"/> under <paramref name="cue"/> with NO answer supplied — only the
    /// query-context (anchor + cue). Runs the walk with goalSymbol=null so the agent halts by its own learned halt head.
    /// <see cref="QueryWalkResult.Reached"/> = it halted AND the final symbol equals <paramref name="expected"/> (the
    /// cued ancestor) — the answer EMERGED. Budget exhaustion leaves Halted=false → an honest abstain.
    /// </summary>
    public static QueryWalkResult WalkQuery(
        NavQueryPolicyNet net, DialecticalSpace space, string member, int cue, string expected,
        Device device, int maxSteps = 16, int k = DefaultK, double minConfidence = 0.0)
    {
        if (!space.TryGetConceptFace(member, out var anchorFace)) return new QueryWalkResult(false, false, member, 0);
        using var policy = new QueryNavPolicy(net, space, anchorFace, cue, device, k, minConfidence);
        var walk = new NavigatorWalk();
        var res = walk.Walk(space, member, anchorFace, null, policy, new NavWalkOptions(MaxSteps: maxSteps));
        var reached = policy.LastHalt && string.Equals(res.FinalSymbol, expected, StringComparison.Ordinal);
        return new QueryWalkResult(reached, policy.LastHalt, res.FinalSymbol, res.Steps);
    }

    // ──────────────────────────────────────────────────────────── Packing.

    private sealed class Bucket : IDisposable
    {
        public required int B { get; init; }
        public required int T { get; init; }
        public required Tensor Anchor { get; init; }    // [B, dim]
        public required Tensor Cue { get; init; }       // [B] long
        public required Tensor Features { get; init; }  // [B, T, K, F]
        public required Tensor Mask { get; init; }      // [B, T, K]
        public required Tensor Context { get; init; }   // [B, T, dim]
        public required Tensor Target { get; init; }    // [B, T] long
        public required Tensor Halt { get; init; }      // [B, T]
        public required Tensor Value { get; init; }     // [B, T]
        public required bool[] HasTarget { get; init; } // length T

        public void Dispose()
        {
            Anchor.Dispose(); Cue.Dispose(); Features.Dispose(); Mask.Dispose(); Context.Dispose();
            Target.Dispose(); Halt.Dispose(); Value.Dispose();
        }
    }

    private static Bucket PackBucket(List<NavQueryTrajectory> trajs, int dim, int f, int k, Device device)
    {
        var b = trajs.Count;
        var t = trajs[0].Steps.Count;

        var anchor = new float[b * dim];
        var cue = new long[b];
        var feat = new float[b * t * k * f];
        var mask = new float[b * t * k];
        var ctx = new float[b * t * dim];
        var target = new long[b * t];
        var halt = new float[b * t];
        var value = new float[b * t];
        var hasTarget = new bool[t];

        for (var bi = 0; bi < b; bi++)
        {
            Array.Copy(trajs[bi].AnchorFace, 0, anchor, bi * dim, dim);
            cue[bi] = trajs[bi].Cue;
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
            Anchor = tensor(anchor, new long[] { b, dim }, device: device),
            Cue = tensor(cue, new long[] { b }, device: device),
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
